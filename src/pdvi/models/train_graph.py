import torch
from typing import Optional, Tuple
import time
from sklearn.metrics import adjusted_rand_score as sk_ari
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
import argparse
from pathlib import Path

try:
    from pdvi.utils.config import deep_update, load_yaml
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from pdvi.utils.config import deep_update, load_yaml


@torch.no_grad()
def global_neg_elbo_with_patch_graph(
    x: torch.Tensor,                 # (n, d)
    alpha: torch.Tensor,             # (n, K)
    m: torch.Tensor,                 # (K, d)
    rho: torch.Tensor,               # (K, d)
    sigma0_2: torch.Tensor,          # (d,)
    sigma1_2: torch.Tensor,          # (d,)
    patch_indices: list[torch.Tensor],          # len=B, each (n_b,) global ids
    patch_edge_index_local: list[torch.Tensor], # len=B, each (2, E_b) local ids in [0,n_b)
    patch_edge_weight: list[torch.Tensor],      # len=B, each (E_b,)
    xi: torch.Tensor | None = None,
    T_for_phi: float | None = None,  # If you want softmax(alpha/T) for objective, pass T_cur; else None -> softmax(alpha)
    eps: float = 1e-12,
    chunk_size: int = 4096,
) -> torch.Tensor:
    # base = entropy + data + prior (without graph)
    base = global_neg_elbo_without_graph(
        x=x, alpha=alpha, m=m, rho=rho,
        sigma0_2=sigma0_2, sigma1_2=sigma1_2,
        xi=xi, T=T_for_phi, eps=eps, chunk_size=chunk_size
    )

    device = x.device
    dtype = x.dtype

    # phi_all: choose temperature consistent with your evaluation if needed
    if T_for_phi is None:
        phi_all = torch.softmax(alpha, dim=-1)  # (n, K)
    else:
        phi_all = torch.softmax(alpha / float(T_for_phi), dim=-1)

    term_patch_graph = x.new_zeros(())
    B = len(patch_indices)

    for b in range(B):
        St = patch_indices[b]
        if St.numel() == 0:
            continue

        edge_l = patch_edge_index_local[b]
        w = patch_edge_weight[b].to(device=device, dtype=dtype)

        if edge_l.numel() == 0:
            continue

        # local -> pick phi inside patch
        phi_b = phi_all[St]  # (n_b, K)

        src = edge_l[0].to(device)
        dst = edge_l[1].to(device)

        # dot per edge: <phi_u, phi_v>
        dots = (phi_b[src] * phi_b[dst]).sum(dim=1)  # (E_b,)
        term_patch_graph = term_patch_graph - 0.5 * (w * dots).sum()

    return base + term_patch_graph


@torch.no_grad()
def global_neg_elbo_without_graph(
    x: torch.Tensor,          # (n, d)
    alpha: torch.Tensor,      # (n, K) logits
    m: torch.Tensor,          # (K, d)
    rho: torch.Tensor,        # (K, d) rho = log s^2
    sigma0_2: torch.Tensor,   # (d,)
    sigma1_2: torch.Tensor,   # (d,)
    xi: torch.Tensor | None = None,   # (d,) or None -> 0
    T: float = 1.0,
    eps: float = 1e-12,
    chunk_size: int = 4096,   # Control memory: compute data/entropy terms in chunks
) -> torch.Tensor:
    """
    Compute global negative ELBO (the objective in your formula) with:
      rho_kj = log s_kj^2, s_kj^2 = exp(rho_kj),
      phi = softmax(alpha/T).

    Returns: scalar tensor (on same device as x).
    """
    device = x.device
    n, d = x.shape
    K = alpha.shape[1]

    # xi
    if xi is None:
        xi = torch.zeros(d, device=device, dtype=x.dtype)
    else:
        xi = xi.to(device=device, dtype=x.dtype)

    # constants / precompute
    inv_sigma0_2 = (1.0 / sigma0_2.to(device=device, dtype=x.dtype)).view(1, d)  # (1, d)
    inv_sigma1_2 = (1.0 / sigma1_2.to(device=device, dtype=x.dtype)).view(1, d)  # (1, d)

    exp_rho = torch.exp(rho)            # (K, d) = s^2
    m2_plus_s2 = m * m + exp_rho        # (K, d)

    # ---- (A) entropy + data term
    term_entropy = x.new_zeros(())
    term_data = x.new_zeros(())

    for st in range(0, n, chunk_size):
        ed = min(n, st + chunk_size)
        x_blk = x[st:ed]                 # (b, d)
        alpha_blk = alpha[st:ed]         # (b, K)
        phi_blk = torch.softmax(alpha_blk / T, dim=-1)  # (b, K)
        phi_safe = phi_blk.clamp_min(eps)

        term_entropy = term_entropy + (phi_safe * phi_safe.log()).sum()

        const_k = 0.5 * (m2_plus_s2 * inv_sigma0_2).sum(dim=1)         # (K,)
        xm = torch.einsum("bd,kd->bk", x_blk * inv_sigma0_2, m)        # (b, K)
        C = const_k.view(1, K) - xm                                    # (b, K)
        term_data = term_data + (phi_blk * C).sum()

    # ---- (B) prior term
    prior_per_k = (((m2_plus_s2 - 2.0 * (xi.view(1, d) * m)) * inv_sigma1_2) - rho).sum(dim=1)  # (K,)
    term_prior = 0.5 * prior_per_k.sum()
    return term_entropy + term_data + term_prior


def local_objective_Lb(
    x: torch.Tensor,                 # (n_b, d)
    phi: torch.Tensor,               # (n_b, K)
    m: torch.Tensor,                 # (K, d) or (1, K, d)
    rho: torch.Tensor,               # (K, d) or (1, K, d)
    sigma0_2: torch.Tensor,          # (d,)
    sigma1_2: torch.Tensor,          # (d,)
    n: int,                          # global n
    edge_index: torch.Tensor,        # (2, E) local indices in [0, n_b)
    edge_weight: torch.Tensor,       # (E,)
    mu: torch.Tensor,                # (K, d)
    gamma: torch.Tensor,             # (K, d)
    m0: torch.Tensor,                # (K, d)
    rho0: torch.Tensor,              # (K, d)
    eta_m: float,
    eta_s: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    device = x.device
    n_b, d = x.shape
    K = phi.shape[1]

    if m.dim() == 3:
        m_kd = m.squeeze(0)
    else:
        m_kd = m
    if rho.dim() == 3:
        rho_kd = rho.squeeze(0)
    else:
        rho_kd = rho

    inv_sigma0_2 = (1.0 / sigma0_2.to(device)).view(1, d)
    inv_sigma1_2 = (1.0 / sigma1_2.to(device)).view(1, d)
    exp_rho = torch.exp(rho_kd)

    phi_safe = phi.clamp_min(eps)
    term_entropy = (phi_safe * phi_safe.log()).sum()

    x_ = x.view(n_b, 1, d)
    m_ = m_kd.view(1, K, d)
    exp_rho_ = exp_rho.view(1, K, d)
    inner = (m_ * m_ + exp_rho_ - 2.0 * x_ * m_) * inv_sigma0_2.view(1, 1, d)
    C_ik = 0.5 * inner.sum(dim=-1)
    term_data = (phi * C_ik).sum()

    prior_per_k = ((m_kd * m_kd + exp_rho) * inv_sigma1_2 - rho_kd).sum(dim=-1)
    term_prior = (n_b / (2.0 * float(n))) * prior_per_k.sum()

    src = edge_index[0].to(device)
    dst = edge_index[1].to(device)
    w = edge_weight.to(device)
    term_graph = -0.5 * (w * (phi[src] * phi[dst]).sum(dim=1)).sum()

    dm = m_kd - m0
    dr = rho_kd - rho0
    term_dual_aug = (
        (mu * dm).sum() + (gamma * dr).sum()
        + 0.5 * (dm * dm).sum() / eta_m
        + 0.5 * (dr * dr).sum() / eta_s
    )

    return term_entropy + term_data + term_prior + term_graph + term_dual_aug


def build_patch_edges_from_csr(St, indptr, indices, weights, N):
    device = St.device

    in_patch = torch.zeros(N, dtype=torch.bool, device=device)
    in_patch[St] = True

    starts = indptr[St]
    ends = indptr[St + 1]
    deg = ends - starts
    total_deg = int(deg.sum().item())

    base = torch.repeat_interleave(starts, deg)
    offset = torch.arange(total_deg, device=device) - torch.repeat_interleave(
        torch.cumsum(deg, dim=0) - deg, deg
    )
    nbr_pos = base + offset
    nbr = indices[nbr_pos]
    w = weights[nbr_pos]

    src = torch.repeat_interleave(St, deg)

    keep = in_patch[nbr]
    src = src[keep]
    dst = nbr[keep]
    w = w[keep]

    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, w


def remap_edge_index_global_to_local(
    St: torch.Tensor,
    edge_index_global: torch.Tensor,
    N: int,
) -> torch.Tensor:
    device = St.device
    g2l = torch.full((N,), -1, dtype=torch.long, device=device)
    g2l[St] = torch.arange(St.numel(), device=device)
    edge_index_local = g2l[edge_index_global]
    return edge_index_local


def alpha_update(data_batch, m, rho, sigma0_2, alpha, edge_index, edge_weight, T, inner_fp_steps=10, eps=1e-12):
    device = data_batch.device
    x = data_batch
    n_b, d = x.shape
    K = alpha.shape[1]

    inv_sigma0_2 = (1.0 / sigma0_2.to(device)).view(1, 1, d)
    x_ = x.view(n_b, 1, d)
    m_ = m.view(1, K, d)
    rho_ = rho.view(1, K, d)
    C = 0.5 * torch.sum(
        (m_ * m_ + torch.exp(rho_) - 2.0 * x_ * m_) * inv_sigma0_2,
        dim=-1
    )
    logits = -C

    phi = torch.softmax(alpha / T, dim=-1)
    s = torch.zeros_like(phi)
    src = edge_index[0].to(device)
    dst = edge_index[1].to(device)
    w = edge_weight.to(device)

    w_col = w.view(-1, 1)
    for _ in range(inner_fp_steps):
        s.zero_()
        s.index_add_(0, dst, phi[src] * w_col)
        phi = torch.softmax((logits + 0.5 * s) / T, dim=-1)

    alpha_new = (logits + 0.5 * s).detach()
    phi = torch.softmax(alpha_new / T, dim=-1)
    eps_phi = 1e-2
    phi = (1 - eps_phi) * phi + eps_phi / K
    alpha_new = (T * phi.log()).detach()

    return alpha_new, phi


def local_update(mu_batch, gamma_batch, data_batch, alpha_batch, m0, rho0, edge_index, edge_weight, n, eta_m, eta_s, device, sigma0_2, sigma1_2, T, num_steps=10, newton_steps=5):
    K = m0.shape[0]
    m = m0.unsqueeze(0).clone()
    d = data_batch.shape[1]
    n_b = data_batch.shape[0]
    rho = rho0.unsqueeze(0).clone()
    alpha = alpha_batch.clone()
    x = data_batch

    inv_sigma0_2 = (1.0 / sigma0_2).view(1, d)
    inv_eta_m = 1.0 / eta_m
    inv_eta_s = 1.0 / eta_s

    inv_sigma1_2 = (1.0 / sigma1_2).view(1, d)
    scale = float(n_b) / float(n)
    inv_nb_over_n_sigma1_2 = (scale * inv_sigma1_2)  # (1, d)

    for it in range(num_steps):
        alpha, phi = alpha_update(data_batch, m, rho, sigma0_2, alpha, edge_index, edge_weight, T)

        S_k = phi.sum(dim=0)
        T_kd = phi.t() @ x

        denom = (S_k.view(K, 1) * inv_sigma0_2) + inv_nb_over_n_sigma1_2 + inv_eta_m
        numer = (T_kd * inv_sigma0_2) + (m0 * inv_eta_m) - mu_batch
        m = (numer / denom).unsqueeze(0)

        Sk = S_k.view(K, 1)
        A = 0.5 * (Sk * inv_sigma0_2 + inv_nb_over_n_sigma1_2)  # (K, d)
        const = -0.5 * scale   # = - n_b/(2n)
        rho_kd = rho.squeeze(0)
        for _ in range(newton_steps):
            exp_rho = torch.exp(rho_kd)
            g = exp_rho * A + const + gamma_batch + (rho_kd - rho0) * inv_eta_s
            gp = exp_rho * A + inv_eta_s
            rho_kd = rho_kd - g / gp
        rho = rho_kd.unsqueeze(0)

    return alpha, m.squeeze(0), rho.squeeze(0)


def remap_labels_to_0K(labels: torch.Tensor):
    uniq = torch.unique(labels)
    inv = torch.bucketize(labels, uniq)
    return inv, uniq.numel()


@torch.no_grad()
def estimate_sigma0_sigma1_from_initlabels(y: torch.Tensor, initial_labels: torch.Tensor, eps=1e-12):
    device = y.device
    n, d = y.shape

    labels, K = remap_labels_to_0K(initial_labels.to(device))

    ones = torch.ones(n, device=device, dtype=y.dtype)
    counts = torch.zeros(K, device=device, dtype=y.dtype).index_add_(0, labels, ones)
    counts_clamped = counts.clamp_min(1.0)

    sum_y = torch.zeros(K, d, device=device, dtype=y.dtype)
    sum_y.index_add_(0, labels, y)

    mean_y = sum_y / counts_clamped.unsqueeze(1)

    sum_y2 = torch.zeros(K, d, device=device, dtype=y.dtype)
    sum_y2.index_add_(0, labels, y * y)
    Ey2 = sum_y2 / counts_clamped.unsqueeze(1)
    var_within = (Ey2 - mean_y * mean_y).clamp_min(0.0)
    std_within = torch.sqrt(var_within + eps)

    sigma0_diag = std_within.median(dim=0).values
    sigma1_diag = mean_y.std(dim=0, unbiased=True)

    return sigma0_diag * 10, sigma1_diag, mean_y, var_within, counts


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(root / "experiments" / "real_data" / "configs" / "train_graph.yaml"),
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg_default = {
        "algorithm": "ours",
        "seed": 1,
        "optimization": {
            "eta_m": 2e-2,
            "eta_s": 2e-1,
            "inner_seed_offset": 20,
            "max_iter": 20000,
            "local_num_steps": 15,
            "local_newton_steps": 5,
        },
        "patch": {"nx": 5, "ny": 5},
        "temperature": {"T": 1e-4, "T_init": 1.0, "anneal_iters": 1},
        "graph": {"neighbors_num": 25, "lambda_similarity": 1.0, "weights_scale": 0.005},
        "logging": {"print_every": 10, "grad_every": 100},
        "runtime": {"device": "cuda", "eps": 1e-12},
        "data": {
            "tensor_template": "experiments/real_data/data/processed/tensor_pca_xy_true_initial_E165_E1S3_seed_{seed}.pt",
            "weights_template": "experiments/real_data/data/intermediate/weights_{neighbors_num}_lambda_{lambda_similarity}_E165_E1S3_seed_{seed}.pt",
        },
        "output": {
            "run_dir_template": "experiments/real_data/results/preconditioned_{max_iter}_seed_{seed}",
            "output_prefix_template": "results_ours_{max_iter}_seed_{seed}",
            "labels_fig_name": "final_labels.png",
        },
    }
    cfg_file = load_yaml(args.config)
    cfg = deep_update(cfg_default, cfg_file)

    seed = int(args.seed if args.seed is not None else cfg["seed"])
    algorithm = cfg["algorithm"]
    eta_m = float(cfg["optimization"]["eta_m"])
    eta_s = float(cfg["optimization"]["eta_s"])
    inner_seed = int(seed + int(cfg["optimization"]["inner_seed_offset"]))
    max_iter = int(cfg["optimization"]["max_iter"])
    local_num_steps = int(cfg["optimization"]["local_num_steps"])
    local_newton_steps = int(cfg["optimization"]["local_newton_steps"])

    nx = int(cfg["patch"]["nx"])
    ny = int(cfg["patch"]["ny"])

    T = float(cfg["temperature"]["T"])
    T_init = float(cfg["temperature"]["T_init"])
    anneal_iters = int(cfg["temperature"]["anneal_iters"])

    neighbors_num = int(cfg["graph"]["neighbors_num"])
    lambda_similarity = float(cfg["graph"]["lambda_similarity"])
    weights_scale = float(cfg["graph"]["weights_scale"])

    print_every = int(cfg["logging"]["print_every"])
    grad_every = int(cfg["logging"]["grad_every"])

    device = str(cfg["runtime"]["device"])
    eps = float(cfg["runtime"]["eps"])

    data_path = root / cfg["data"]["tensor_template"].format(seed=seed)
    graph_path = root / cfg["data"]["weights_template"].format(
        seed=seed,
        neighbors_num=neighbors_num,
        lambda_similarity=lambda_similarity,
    )
    output_folder = root / cfg["output"]["run_dir_template"].format(max_iter=max_iter, seed=seed)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_prefix = cfg["output"]["output_prefix_template"].format(max_iter=max_iter, seed=seed)
    labels_fig_name = cfg["output"]["labels_fig_name"]

    B = nx * ny

    torch.manual_seed(inner_seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    data = torch.load(data_path, map_location=device)
    coords = data["spatial"]
    data_pca = data["pca_data"]
    true_labels = data["true_labels"]
    initial_labels = data["initial_labels"]
    initial_labels_cpu = initial_labels.detach().cpu()

    n = data_pca.shape[0]
    d = data_pca.shape[1]
    K = torch.unique(true_labels).numel()

    sigma0, sigma1, mean_k, var_k, counts = estimate_sigma0_sigma1_from_initlabels(data_pca, initial_labels)
    sigma0_2 = sigma0 ** 2
    sigma1_2 = sigma1 ** 2

    graph = torch.load(graph_path, map_location=device)
    indptr = graph["indptr"].to(device)
    indices = graph["indices"].to(device)
    weights = graph["weights"].to(device) * weights_scale
    meta = graph["meta"]

    x_edges = torch.linspace(coords[:, 0].min(), coords[:, 0].max(), steps=nx + 1, device=coords.device)
    y_edges = torch.linspace(coords[:, 1].min(), coords[:, 1].max(), steps=ny + 1, device=coords.device)
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

    patch_edge_index_local = [None] * B
    patch_edge_weight = [None] * B
    for b in range(B):
        St = patch_indices[b]
        edge_g, w = build_patch_edges_from_csr(St, indptr, indices, weights, n)
        edge_l = remap_edge_index_global_to_local(St, edge_g, N=n)
        patch_edge_index_local[b] = edge_l
        patch_edge_weight[b] = w

    loss_list = []
    ari_list = []
    time_list = []
    grad_norm_list = []
    time_start = time.time()

    # ---- metric logs (raw values) ----
    log_iter = []
    log_objective = []
    log_ari = []
    log_residual = []
    log_delta_consensus = []
    log_time_passed = []
    log_grad_norm = []
    # ---- new: final clustering output ----
    final_z_pred = None
    # ----------------------------------

    if algorithm == "ours":
        global_residual_list = []
        alpha_all = 1e-2 * torch.randn(n, K, device=device)
        m0 = mean_k
        rho0 = torch.log(var_k + 1e-12)

        mu = torch.zeros(B, K, d, device=device)
        gamma = torch.zeros(B, K, d, device=device)
        h_t_m = torch.zeros(K, d, device=device)
        h_t_rho = torch.zeros(K, d, device=device)

        primal_residual = torch.tensor(1000000, device=device)
        m = m0.clone()
        rho = rho0.clone()
        m0_prev = m0.clone() + 1000
        rho0_prev = rho0.clone() + 1000

        for iter in range(max_iter + 1):
            # ---- compute current temperature ----
            t = min(1.0, iter / anneal_iters)
            T_cur = T_init * (1.0 - t) + T * t
            # ------------------------------------

            if iter % print_every == 0:
                neg_elbo = global_neg_elbo_with_patch_graph(
                    x=data_pca, alpha=alpha_all, m=m0, rho=rho0,
                    sigma0_2=sigma0_2, sigma1_2=sigma1_2,
                    patch_indices=patch_indices,
                    patch_edge_index_local=patch_edge_index_local,
                    patch_edge_weight=patch_edge_weight,
                    xi=None,
                    T_for_phi=T_cur,
                )

                phi = torch.softmax(alpha_all / T_cur, dim=-1)
                z_pred = torch.argmax(phi, dim=-1)
                final_z_pred = z_pred.detach().clone()
                ari = sk_ari(true_labels.detach().cpu().numpy(), z_pred.detach().cpu().numpy())
                ari_list.append(ari)

                rel_m = torch.linalg.norm(m - m0) / (torch.linalg.norm(m0) + eps)
                rel_r = torch.linalg.norm(rho - rho0) / (torch.linalg.norm(rho0) + eps)
                primal_residual = (rel_m + rel_r).item()
                global_residual_list.append(primal_residual)

                delta_m0 = torch.linalg.norm(m0 - m0_prev) / (torch.linalg.norm(m0_prev) + eps)
                delta_r0 = torch.linalg.norm(rho0 - rho0_prev) / (torch.linalg.norm(rho0_prev) + eps)
                delta = (delta_m0 + delta_r0).item()

                time_passed = time.time() - time_start
                time_list.append(time_passed)

                # ---- grad norm (computed only every grad_every iters; else NaN) ----
                if iter % grad_every == 0:
                    with torch.enable_grad():
                        m0_tmp = m0.detach().clone().requires_grad_(True)
                        rho0_tmp = rho0.detach().clone().requires_grad_(True)

                        loss_wo_graph = global_neg_elbo_without_graph.__wrapped__(
                            x=data_pca,
                            alpha=alpha_all,
                            m=m0_tmp,
                            rho=rho0_tmp,
                            sigma0_2=sigma0_2,
                            sigma1_2=sigma1_2,
                            xi=None,
                            T=T_cur,
                            eps=eps,
                        )
                        loss_wo_graph.backward()

                        g_m = m0_tmp.grad
                        g_r = rho0_tmp.grad
                        grad_norm = torch.sqrt(g_m.pow(2).sum() + g_r.pow(2).sum()).item()
                else:
                    grad_norm = float("nan")
                # ------------------------------------------------------------------

                print(
                    f"iter {iter}/{max_iter},objective = {neg_elbo.item():.6e}, ari = {ari:.6e}, "
                    f"residual = {primal_residual:.6e}, delta_consensus = {delta:.6e}, time_passed = {time_passed:.6f}, T = {T_cur:.6f}"
                )
                if iter % grad_every == 0:
                    print(f"grad norm of (m0, rho0) for objective WITHOUT graph: {grad_norm:.6e}")

                # ---- append metrics to logs (raw values) ----
                log_iter.append(int(iter))
                log_objective.append(float(neg_elbo.item()))
                log_ari.append(float(ari))
                log_residual.append(float(primal_residual))
                log_delta_consensus.append(float(delta))
                log_time_passed.append(float(time_passed))
                log_grad_norm.append(float(grad_norm))
                # -------------------------------------------

                m0_prev = m0.clone()
                rho0_prev = rho0.clone()

            b = torch.randint(0, B, (1,), device=device).item()
            mu_batch = mu[b]
            gamma_batch = gamma[b]
            St = patch_indices[b]
            data_batch = data_pca[St]
            alpha_batch = alpha_all[St]
            edge_index = patch_edge_index_local[b]
            edge_weight = patch_edge_weight[b]

            alpha, m, rho = local_update(
                mu_batch, gamma_batch, data_batch, alpha_batch,
                m0, rho0, edge_index, edge_weight, n,
                eta_m, eta_s, device, sigma0_2, sigma1_2, T_cur,
                num_steps=local_num_steps, newton_steps=local_newton_steps
            )

            alpha_all[St] = alpha
            m_diff = m - m0
            rho_diff = rho - rho0

            mu[b] += m_diff / eta_m
            gamma[b] += rho_diff / eta_s

            m_diff_mean = m_diff / B
            rho_diff_mean = rho_diff / B
            h_t_m += m_diff_mean
            h_t_rho += rho_diff_mean
            m0 = m + h_t_m
            rho0 = rho + h_t_rho

    # ---- save logs for future comparison ----
    results = {
        "eta_m": float(eta_m),
        "eta_s": float(eta_s),
        "nx": int(nx),
        "ny": int(ny),
        "B": int(B),
        "max_iter": int(max_iter),
        "neighbors_num": int(neighbors_num),
        "lambda_similarity": float(lambda_similarity),
        "weights_scale": float(weights_scale),
        "T": float(T),
        "T_init": float(T_init),
        "anneal_iters": int(anneal_iters),
        "print_every": int(print_every),
        "grad_every": int(grad_every),
        "data_path": str(data_path),
        "graph_path": str(graph_path),
        "log_iter": log_iter,
        "objective": log_objective,
        "ari": log_ari,
        "residual": log_residual,
        "delta_consensus": log_delta_consensus,
        "time_passed": log_time_passed,
        "grad_norm_wo_graph_wrt_m0_rho0": log_grad_norm,
        "final_z_pred": final_z_pred.detach().cpu() if final_z_pred is not None else None,
        "final_m0": m0.detach().cpu(),
        "final_rho0": rho0.detach().cpu(),
    }
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(
        output_folder,
        f"{output_prefix}_eta_m_{eta_m:.0e}_eta_s_{eta_s:.0e}.pt"
    )
    torch.save(results, save_path)
    print(f"Saved results to: {save_path}")
    # ----------------------------------------

    # Use the final clustering result (if available) for summary + plot
    if final_z_pred is None:
        phi_final = torch.softmax(alpha_all / T_cur, dim=-1)
        final_z_pred = torch.argmax(phi_final, dim=-1)

    coords_cpu = coords.detach().cpu().numpy()
    z_pred_cpu = final_z_pred.detach().cpu().numpy().astype(int)
    print(f"Number of unique labels: {len(np.unique(z_pred_cpu))}")
    for i in range(K):
        print(f"Cluster {i}: {len(np.where(z_pred_cpu == i)[0])}")

    c1 = plt.get_cmap("tab20").colors
    c2 = plt.get_cmap("tab20b").colors
    colors = list(c1) + list(c2)
    cmap23 = ListedColormap(colors[:23])
    plt.figure()
    plt.scatter(coords_cpu[:, 0], coords_cpu[:, 1], s=5, c=z_pred_cpu, cmap=cmap23)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Final labels (Ours - Preconditioned)")
    plt.savefig(os.path.join(output_folder, labels_fig_name))

    # print the initial labels as well
    plt.figure()
    plt.scatter(coords_cpu[:, 0], coords_cpu[:, 1], s=5, c=initial_labels_cpu, cmap=cmap23)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Initial labels")
    plt.savefig(os.path.join(output_folder, "initial_labels.png"))


    
