import torch
from torchmetrics.functional.clustering import adjusted_rand_score
from sklearn.metrics import adjusted_rand_score as sk_ari
import time, math, numpy as np, os
from pathlib import Path

# ============================================================
# (Optional) synthetic data generator
# ============================================================
def generate_synthetic_data_highdim(
    n: int,
    K: int,
    d: int,
    sigma0_max: float,
    sigma0_min: float,
    sigma1_max: float,
    sigma1_min: float,
    device: str = "cpu",
    seed: int = 42,
):
    torch.manual_seed(seed)
    sigma0_diag = torch.linspace(sigma0_max, sigma0_min, d, device=device)  # (d,)
    sigma1_diag = torch.linspace(sigma1_max, sigma1_min, d, device=device)  # (d,)
    mu = sigma1_diag * torch.randn(K, d, device=device)                      # (K,d)
    z = torch.randint(low=0, high=K, size=(n,), device=device)               # (n,)
    y = mu[z] + sigma0_diag * torch.randn(n, d, device=device)               # (n,d)
    return y, z, mu, sigma0_diag, sigma1_diag


# ============================================================
# utils: pack/unpack global lambda_0 = [m0, rho0]
# ============================================================
def pack_lambda(m0: torch.Tensor, rho0: torch.Tensor) -> torch.Tensor:
    # m0, rho0: (K,d)
    return torch.cat([m0.reshape(-1), rho0.reshape(-1)], dim=0)

def unpack_lambda(lambda_0: torch.Tensor, K: int, d: int, device=None):
    if device is None:
        device = lambda_0.device
    lambda_0 = lambda_0.to(device)
    assert lambda_0.numel() == 2 * K * d, f"lambda_0 must have shape (2*K*d,), got {lambda_0.shape}"
    m0 = lambda_0[: K * d].view(K, d)          # (K,d)
    rho0 = lambda_0[K * d :].view(K, d)        # (K,d)
    return m0, rho0


# ============================================================
# Newton solver for rho (elementwise)
# ============================================================
def newton_rho(rho, A, gamma, rho0, inv_eta, n, num_newton=5):
    """
    rho, A, gamma, rho0: (K,d)
    inv_eta: scalar (1/eta_s)
    """
    c = gamma - 0.5 / n
    for _ in range(num_newton):
        exp_rho = torch.exp(rho)
        f = A * exp_rho + c + inv_eta * (rho - rho0)
        fp = A * exp_rho + inv_eta
        rho = rho - f / fp
    return rho


# ============================================================
# ELBO pieces (same as your high-dim diagonal version)
# ============================================================
def compute_const_term_diag(y, sigma0_diag, sigma1_diag, K):
    n, d = y.shape
    inv_sigma0_2 = 1.0 / (sigma0_diag ** 2)
    y_sq_weighted_sum = (y ** 2 * inv_sigma0_2.view(1, d)).sum().item()

    logdet_sigma0_2 = (2.0 * torch.log(sigma0_diag)).sum().item()
    logdet_sigma1_2 = (2.0 * torch.log(sigma1_diag)).sum().item()

    const1 = n * (0.5 * d * math.log(2.0 * math.pi) + 0.5 * logdet_sigma0_2 + math.log(K))
    const2 = 0.5 * y_sq_weighted_sum
    const3 = (K / 2.0) * (logdet_sigma1_2 - d)
    return const1 + const2 + const3


def elbo_batch_diag(alpha_all, lambda_0, y, sigma0_diag, sigma1_diag, n):
    """
    alpha_all: (n,K)
    lambda_0: (2*K*d,)
    y: (n,d)
    returns: scalar sum_i L_i (no const_term)
    """
    device = y.device
    n_all, K = alpha_all.shape
    d = y.shape[1]

    m0, rho0 = unpack_lambda(lambda_0, K, d, device=device)  # (K,d),(K,d)
    s2 = torch.exp(rho0)                                     # (K,d)

    phi = torch.softmax(alpha_all, dim=-1)                   # (n,K)
    log_phi = torch.log_softmax(alpha_all, dim=-1)           # (n,K)

    inv_sigma0_2 = (1.0 / (sigma0_diag ** 2)).view(1, 1, d)  # (1,1,d)
    inv_sigma1_2 = (1.0 / (sigma1_diag ** 2)).view(1, d)     # (1,d)

    y_ = y.view(n_all, 1, d)            # (n,1,d)
    m_ = m0.view(1, K, d)               # (1,K,d)
    s2_ = s2.view(1, K, d)              # (1,K,d)

    data_term = 0.5 * ((m_ * m_ + s2_ - 2.0 * y_ * m_) * inv_sigma0_2).sum(dim=-1)  # (n,K)

    vi_term = (-0.5 / n) * rho0.sum(dim=-1) + (0.5 / n) * ((m0 * m0 + s2) * inv_sigma1_2).sum(dim=-1)  # (K,)
    vi_term = vi_term.view(1, K)  # (1,K)

    inside = phi * (log_phi + data_term) + vi_term
    return inside.sum()  # scalar


# ============================================================
# label remap + empirical sigma estimate (same as your version)
# ============================================================
def remap_labels_to_0K(labels: torch.Tensor):
    uniq, inv = torch.unique(labels, return_inverse=True)
    return inv, uniq.numel()

@torch.no_grad()
def estimate_sigma0_sigma1_from_initlabels(y: torch.Tensor, initial_labels: torch.Tensor, eps=1e-12):
    """
    returns:
      sigma0_diag: (d,) average within-cluster std (per-dim)
      sigma1_diag: (d,) std across cluster means (per-dim)
    """
    device = y.device
    n, d = y.shape
    labels, K = remap_labels_to_0K(initial_labels.to(device))

    ones = torch.ones(n, device=device, dtype=y.dtype)
    counts = torch.zeros(K, device=device, dtype=y.dtype).index_add_(0, labels, ones)  # (K,)
    counts_clamped = counts.clamp_min(1.0)

    sum_y = torch.zeros(K, d, device=device, dtype=y.dtype)
    sum_y.index_add_(0, labels, y)
    mean_y = sum_y / counts_clamped.unsqueeze(1)  # (K,d)

    sum_y2 = torch.zeros(K, d, device=device, dtype=y.dtype)
    sum_y2.index_add_(0, labels, y * y)
    Ey2 = sum_y2 / counts_clamped.unsqueeze(1)
    var_within = (Ey2 - mean_y * mean_y).clamp_min(0.0)
    std_within = torch.sqrt(var_within + eps)  # (K,d)

    sigma0_diag = std_within.mean(dim=0)            # (d,)
    sigma1_diag = mean_y.std(dim=0, unbiased=True)  # (d,)

    return sigma0_diag, sigma1_diag


# ============================================================
# build patches (your code, wrapped as a function)
# ============================================================
@torch.no_grad()
def build_patch_indices(coords: torch.Tensor, nx: int, ny: int):
    """
    coords: (n,2) on same device as y.
    returns patch_indices: list length B=nx*ny, each is 1D LongTensor of indices
    """
    device = coords.device
    n = coords.shape[0]
    B = nx * ny

    x_edges = torch.linspace(coords[:, 0].min(), coords[:, 0].max(), steps=nx + 1, device=device)
    y_edges = torch.linspace(coords[:, 1].min(), coords[:, 1].max(), steps=ny + 1, device=device)
    all_idx = torch.arange(n, device=device, dtype=torch.long)

    patch_indices = [None] * B
    for ix in range(nx):
        x_lo, x_hi = x_edges[ix], x_edges[ix + 1]
        in_x = (coords[:, 0] >= x_lo) & (coords[:, 0] < x_hi if ix < nx - 1 else coords[:, 0] <= x_hi)
        for iy in range(ny):
            y_lo, y_hi = y_edges[iy], y_edges[iy + 1]
            in_y = (coords[:, 1] >= y_lo) & (coords[:, 1] < y_hi if iy < ny - 1 else coords[:, 1] <= y_hi)

            mask = in_x & in_y
            b = ix * ny + iy
            patch_indices[b] = all_idx[mask]
    return patch_indices



# ============================================================
# (0.45) patch-local update: minimize L_b
# ============================================================

def patch_objective_045_diag(
    alpha: torch.Tensor,      # (n_b,K)
    y_patch: torch.Tensor,    # (n_b,d)
    m: torch.Tensor,          # (K,d)
    rho: torch.Tensor,        # (K,d)
    m0: torch.Tensor,         # (K,d)
    rho0: torch.Tensor,       # (K,d)
    mu_m_b: torch.Tensor,     # (K,d)
    gamma_b: torch.Tensor,    # (K,d)
    eta_m: float,
    eta_s: float,
    sigma0_diag: torch.Tensor,# (d,)
    sigma1_diag: torch.Tensor,# (d,)
    n: int,
    eps: float = 1e-12,
):
    """
    Compute patch augmented Lagrangian objective corresponding to eq (0.45),
    adapted to diagonal sigma0, sigma1 (per-dimension std vectors).

    Returns a scalar tensor.
    """
    n_b, K = alpha.shape
    d = y_patch.shape[1]
    device = y_patch.device

    # phi, log_phi
    phi = torch.softmax(alpha, dim=-1)                # (n_b,K)
    log_phi = torch.log_softmax(alpha, dim=-1)        # (n_b,K)

    inv_sigma0_2 = (1.0 / (sigma0_diag ** 2 + eps)).view(1, 1, d)  # (1,1,d)
    inv_sigma1_2 = (1.0 / (sigma1_diag ** 2 + eps)).view(1, d)     # (1,d)

    exp_rho = torch.exp(rho)                          # (K,d)

    # ----- data term: C_ik = 0.5 * sum_j (m_kj^2 + exp(rho_kj) - 2 y_ij m_kj)/sigma0_j^2
    y_ = y_patch.view(n_b, 1, d)                      # (n_b,1,d)
    m_ = m.view(1, K, d)                              # (1,K,d)
    exp_rho_ = exp_rho.view(1, K, d)                  # (1,K,d)

    C = 0.5 * ((m_ * m_ + exp_rho_ - 2.0 * y_ * m_) * inv_sigma0_2).sum(dim=-1)  # (n_b,K)

    # ----- sum_i sum_k phi_ik (log phi_ik + C_ik)
    term_data_entropy = (phi * (log_phi + C)).sum()   # scalar

    # ----- prior/VI term per k:  (-1/(2n)) sum_j rho_kj + (1/(2n)) sum_j (m_kj^2+exp(rho_kj))/sigma1_j^2
    term_vi = (-0.5 / n) * rho.sum() + (0.5 / n) * ((m * m + exp_rho) * inv_sigma1_2).sum()

    # ----- dual linear terms: <mu_m_b, m-m0> + <gamma_b, rho-rho0>
    dm = m - m0
    dr = rho - rho0
    term_dual = (mu_m_b * dm).sum() + (gamma_b * dr).sum()

    # ----- quadratic augmented terms: (1/(2*eta_m))||m-m0||^2 + (1/(2*eta_s))||rho-rho0||^2
    term_aug = 0.5 * (dm * dm).sum() / eta_m + 0.5 * (dr * dr).sum() / eta_s

    return term_data_entropy + term_vi + term_dual + term_aug


def local_update_patch_diag(
    mu_b: torch.Tensor,                 # (2*K*d,)
    y_patch: torch.Tensor,              # (n_b,d)
    alpha_patch: torch.Tensor,          # (n_b,K)
    lambda_0: torch.Tensor,             # (2*K*d,)
    K: int,
    eta_m: float,
    eta_s: float,
    n: int,
    device: str,
    sigma0_diag: torch.Tensor,          # (d,)
    sigma1_diag: torch.Tensor,          # (d,)
    num_steps: int = 10,
    newton_steps: int = 5,
):
    """
    Clear/accurate implementation of eq (0.45) for ONE patch b.
    Local variables are (m_b, rho_b) shared within patch.
    Alpha (and phi) are per-point in the patch.
    """
    y_patch = y_patch.to(device)
    alpha = alpha_patch.to(device).clone()

    n_b, d = y_patch.shape

    # unpack patch duals: mu_b = [mu_m_b, gamma_b]
    mu_m_b = mu_b[: K * d].view(K, d)          # (K,d)
    gamma_b = mu_b[K * d :].view(K, d)         # (K,d)

    # unpack global lambda_0
    m0, rho0 = unpack_lambda(lambda_0, K, d, device=y_patch.device)  # (K,d)

    # init patch locals
    m = m0.clone()      # (K,d)
    rho = rho0.clone()  # (K,d)

    inv_sigma0_2 = (1.0 / (sigma0_diag ** 2)).view(1, d)  # (1,d)
    inv_sigma1_2 = (1.0 / (sigma1_diag ** 2)).view(1, d)  # (1,d)
    inv_eta_m = 1.0 / eta_m
    inv_eta_s = 1.0 / eta_s

    for step in range(num_steps):
        # ---------- (A) alpha/phi closed form ----------
        exp_rho = torch.exp(rho)                      # (K,d)

        y_ = y_patch.unsqueeze(1)                     # (n_b,1,d)
        m_ = m.unsqueeze(0)                           # (1,K,d)
        exp_rho_ = exp_rho.unsqueeze(0)               # (1,K,d)

        C = 0.5 * ((m_ * m_ + exp_rho_ - 2.0 * y_ * m_) * inv_sigma0_2.view(1, 1, d)).sum(dim=-1)  # (n_b,K)
        alpha = -C
        phi = torch.softmax(alpha, dim=-1)            # (n_b,K)

        # Sufficient stats in this patch
        Nk = phi.sum(dim=0).view(K, 1)                # (K,1)
        Sy = (phi.unsqueeze(-1) * y_patch.unsqueeze(1)).sum(dim=0)  # (K,d)

        # ---------- (B) m_b closed form ----------
        denom_m = Nk * inv_sigma0_2 + (1.0 / n) * inv_sigma1_2 + inv_eta_m  # (K,d)
        numer_m = Sy * inv_sigma0_2 + m0 * inv_eta_m - mu_m_b               # (K,d)
        m = numer_m / denom_m

        # ---------- (C) rho_b Newton update ----------
        A = Nk * (0.5 * inv_sigma0_2) + 0.5 * (1.0 / n) * inv_sigma1_2      # (K,d)
        rho = newton_rho(
            rho=rho,
            A=A,
            gamma=gamma_b,
            rho0=rho0,
            inv_eta=inv_eta_s,
            n=n,
            num_newton=newton_steps
        )

        # obj = patch_objective_045_diag(
        #         alpha=alpha,
        #         y_patch=y_patch,
        #         m=m,
        #         rho=rho,
        #         m0=m0,
        #         rho0=rho0,
        #         mu_m_b=mu_m_b,
        #         gamma_b=gamma_b,
        #         eta_m=eta_m,
        #         eta_s=eta_s,
        #         sigma0_diag=sigma0_diag,
        #         sigma1_diag=sigma1_diag,
        #         n=n,
        #     )
        # print(f"    [local step {step+1:02d}/{num_steps}] obj = {obj.item():.6e}")

    lambda_b = pack_lambda(m, rho)  # (2*K*d,)
    return alpha, lambda_b, m, rho


# ============================================================
# main: patch-consistent training loop with your h_t update
# ============================================================
if __name__ == "__main__":
    # ---------------- settings ----------------
    eta_m = 1e-4
    eta_s = 1
    max_iter = 10000
    log_every = 10

    # patch grid
    nx, ny = 10, 10
    B = nx * ny

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ------------- load -------------
    root = Path(__file__).resolve().parents[3]
    processed_dir = root / "experiments" / "real_data" / "data" / "processed"
    results_dir = root / "experiments" / "real_data" / "results" / "mfvi_highdim_patch"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(processed_dir / "tensor_pca_xy_true_initial_E165_E1S3.pt", map_location=device)
    y = data["pca_data"].to(device)           # (n,d)
    coords = data["spatial"].to(device)       # (n,2)
    z_true = data["true_labels"].to(device)   # (n,)
    initial_labels = data["initial_labels"].to(device)

    n, d = y.shape
    K = torch.unique(z_true).numel()

    # ------------- patches -------------
    patch_indices = build_patch_indices(coords, nx, ny)
    # (optional) filter empty patches for sampling
    nonempty_patches = [b for b in range(B) if patch_indices[b].numel() > 0]
    if len(nonempty_patches) == 0:
        raise RuntimeError("All patches are empty. Check coords / nx,ny.")

    # ------------- sigma estimates -------------
    sigma0_diag, sigma1_diag = estimate_sigma0_sigma1_from_initlabels(y, initial_labels)
    const_term = compute_const_term_diag(y, sigma0_diag, sigma1_diag, K)

    # ------------- init alpha, (m0,rho0), duals -------------
    alpha_all = 1e-2 * torch.randn(n, K, device=device)

    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0)
    y_var = y.var(dim=0)

    m0 = y_mean.view(1, d) + 0.5 * y_std.view(1, d) * torch.randn(K, d, device=device)      # (K,d)
    rho0 = torch.log(y_var + 1e-12).view(1, d) + 1e-2 * torch.randn(K, d, device=device)    # (K,d)

    # Global variables stored as tensors (K,d); lambda_0 is just a packed view when needed
    h_t_m = torch.zeros(K, d, device=device)
    h_t_rho = torch.zeros(K, d, device=device)

    # dual per patch (eq 0.45 uses mu_b, gamma_b): (B, 2*K*d)
    mu_patch = torch.zeros(B, 2 * K * d, device=device)

    # logs
    loss_list, ari_list, time_list, residual_list = [], [], [], []
    time_start = time.time()

    primal_residual = torch.tensor(1e6, device=device)

    for it in range(max_iter + 1):
        # ---------- logging ----------
        if it % log_every == 0:
            lambda_0 = pack_lambda(m0, rho0)
            elbo = elbo_batch_diag(alpha_all, lambda_0, y, sigma0_diag, sigma1_diag, n) + const_term
            phi_all = torch.softmax(alpha_all, dim=-1)
            z_pred = torch.argmax(phi_all, dim=-1)

            ari = float(adjusted_rand_score(z_pred.detach().cpu().to(torch.int64),
                                            z_true.detach().cpu().to(torch.int64)))
            ari_sk = sk_ari(z_true.detach().cpu().numpy(),
                            z_pred.detach().cpu().numpy())

            loss_list.append(elbo.item())
            ari_list.append(ari)
            residual_list.append(primal_residual.item())
            time_list.append(time.time() - time_start)

            print(
                f"iter {it}/{max_iter}, elbo = {elbo:.6e}, ari = {ari:.6e}, "
                f"ari_sk = {ari_sk:.6e}, residual = {primal_residual:.6e}, time = {time_list[-1]:.3f}s"
            )

        # ---------- sample P=5 non-empty patches WITHOUT replacement ----------
        P = 10
        M = len(nonempty_patches)
        P_eff = min(P, M)  # in case there are fewer than 5 non-empty patches

        perm = torch.randperm(M, device=device)  # without replacement
        bs = [nonempty_patches[i.item()] for i in perm[:P_eff]]

        # Freeze the current global lambda_0 for this outer iteration:
        # every sampled patch compares against the SAME (m0, rho0)
        lambda_0 = pack_lambda(m0, rho0)

        # Accumulators for patch-aggregated global updates
        m_diff_acc = torch.zeros_like(m0)      # (K, d)
        rho_diff_acc = torch.zeros_like(rho0)  # (K, d)
        w_sum = 0.0

        # Optional: residual over the sampled patches
        res2 = 0.0

        for b in bs:
            idx = patch_indices[b]            # 1D LongTensor of point indices in patch b
            n_b = idx.numel()
            if n_b == 0:
                continue

            y_patch = y[idx]                  # (n_b, d)
            alpha_patch = alpha_all[idx]      # (n_b, K)
            mu_b = mu_patch[b]                # (2*K*d,)

            # ---- local patch solve (Eq. 0.45) ----
            alpha_patch_new, lambda_b, m_b, rho_b = local_update_patch_diag(
                mu_b=mu_b,
                y_patch=y_patch,
                alpha_patch=alpha_patch,
                lambda_0=lambda_0,
                K=K,
                eta_m=eta_m,
                eta_s=eta_s,
                n=n,
                device=device,
                sigma0_diag=sigma0_diag,
                sigma1_diag=sigma1_diag,
                num_steps=30,
                newton_steps=10,
            )

            # Write back per-point alpha for this patch
            alpha_all[idx] = alpha_patch_new

            # ---- primal residual contribution (for monitoring) ----
            lambda_diff_b = lambda_b - lambda_0              # (2*K*d,)
            res2 += float(lambda_diff_b.pow(2).sum().item())

            # ---- dual update (per patch) ----
            mu_patch[b, : K * d] += lambda_diff_b[: K * d] / eta_m
            mu_patch[b, K * d :] += lambda_diff_b[K * d :] / eta_s

            # ---- patch-consistent weighting ----
            # Eq. (0.45) sums over i in S_b, so weight patch contributions by |S_b| / n
            m_diff_acc += (m_b - m0)
            rho_diff_acc += (rho_b - rho0)

        # Optional: a combined residual over the sampled patches
        primal_residual = torch.tensor(math.sqrt(res2), device=device)

        # ---- global update using the aggregated P patches ----
        # Your h_t accumulation (aggregated across sampled patches)
        h_t_m += m_diff_acc/B
        h_t_rho += rho_diff_acc/B

        # Keep your preferred form: m0 = m_bar + h_t_m, rho0 = rho_bar + h_t_rho
        # where (m_bar, rho_bar) is the weighted average of the sampled patch solutions.
        m_bar = m_diff_acc / P_eff
        rho_bar = rho_diff_acc / P_eff

        m0 = m_bar + h_t_m
        rho0 = rho_bar + h_t_rho

    # ---------------- save outputs (optional) ----------------
    np.save(results_dir / "loss.npy", np.array(loss_list))
    np.save(results_dir / "ari.npy", np.array(ari_list))
    np.save(results_dir / "time.npy", np.array(time_list))
    np.save(results_dir / "residual.npy", np.array(residual_list))
