import torch
from torchmetrics.functional.clustering import adjusted_rand_score
from sklearn.metrics import adjusted_rand_score as sk_ari
import time
import math
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path


# =========================
# utils: lambda pack/unpack
# =========================
def unpack_lambda(lambda_0: torch.Tensor, K: int, d: int, device=None):
    if device is None:
        device = lambda_0.device
    lambda_0 = lambda_0.to(device)
    assert lambda_0.numel() == 2 * K * d, f"lambda_0 must have shape (2*K*d,), got {lambda_0.shape}"
    m0 = lambda_0[: K * d].view(K, d)          # (K,d)
    rho0 = lambda_0[K * d :].view(K, d)        # (K,d)
    return m0, rho0


# =========================
# Newton solver for rho
# =========================
def newton_rho(rho, A, gamma, rho0, inv_eta, n, num_newton=5):
    """
    Elementwise Newton. Works for shapes like (B,K,d).
    """
    c = gamma - 0.5 / n
    for _ in range(num_newton):
        exp_rho = torch.exp(rho)
        f = A * exp_rho + c + inv_eta * (rho - rho0)
        fp = A * exp_rho + inv_eta
        rho = rho - f / fp
    return rho


# =========================
# constants and ELBO
# =========================
def compute_const_term_diag(y, sigma0_diag, sigma1_diag, K):
    """
    y: (n,d)
    sigma0_diag: (d,) std
    sigma1_diag: (d,) std

    This is the high-dim analog of your 1D compute_const_term,
    following the same decomposition pattern.
    """
    n, d = y.shape
    inv_sigma0_2 = 1.0 / (sigma0_diag ** 2)  # (d,)
    y_sq_weighted_sum = (y ** 2 * inv_sigma0_2.view(1, d)).sum().item()

    # log det of diagonal covariance in terms of std
    logdet_sigma0_2 = (2.0 * torch.log(sigma0_diag)).sum().item()  # sum_j log sigma0_j^2
    logdet_sigma1_2 = (2.0 * torch.log(sigma1_diag)).sum().item()  # sum_j log sigma1_j^2

    const1 = n * (0.5 * d * math.log(2.0 * math.pi) + 0.5 * logdet_sigma0_2 + math.log(K))
    const2 = 0.5 * y_sq_weighted_sum
    const3 = (K / 2.0) * (logdet_sigma1_2 - d)

    return const1 + const2 + const3


def elbo_batch_diag(alpha_batch, lambda_0, y, sigma0_diag, sigma1_diag, n):
    """
    alpha_batch: (B,K)
    lambda_0: (2*K*d,)
    y: (B,d)
    sigma0_diag,sigma1_diag: (d,) std
    """
    device = y.device
    B, K = alpha_batch.shape
    assert y.ndim == 2, f"y must be (B,d), got {y.shape}"
    d = y.shape[1]

    m0, rho0 = unpack_lambda(lambda_0, K, d, device=device)  # (K,d),(K,d)
    s2 = torch.exp(rho0)                                     # (K,d)

    phi = torch.softmax(alpha_batch / T, dim=-1)                 # (B,K)
    log_phi = torch.log_softmax(alpha_batch / T, dim=-1)         # (B,K)

    inv_sigma0_2 = (1.0 / (sigma0_diag ** 2)).view(1, 1, d)  # (1,1,d)
    inv_sigma1_2 = (1.0 / (sigma1_diag ** 2)).view(1, d)     # (1,d)

    y_ = y.view(B, 1, d)             # (B,1,d)
    m_ = m0.view(1, K, d)            # (1,K,d)
    s2_ = s2.view(1, K, d)           # (1,K,d)

    # data_term: (B,K)
    data_term = 0.5 * ((m_ * m_ + s2_ - 2.0 * y_ * m_) * inv_sigma0_2).sum(dim=-1)

    # vi_term per k: (K,)
    vi_term = (-0.5 / n) * rho0.sum(dim=-1) + (0.5 / n) * ((m0 * m0 + s2) * inv_sigma1_2).sum(dim=-1)
    vi_term = vi_term.view(1, K)  # (1,K)

    inside = phi * (log_phi + data_term) + vi_term
    fun = inside.sum(dim=-1)  # (B,)
    return fun.sum()


# =========================
# local update (high-dim)
# =========================
def local_update_diag(
    mu, y, alpha_batch, lambda_0,
    B, K, d_lambda, eta_m, eta_s, n, device,
    sigma0_diag, sigma1_diag,
    T,
    num_steps=10,
    newton_steps=5
):
    """
    mu: (B, 2*K*d) -> split into mu_m and gamma, each reshaped to (B,K,d)
    y: (B,d)
    alpha_batch: (B,K)
    lambda_0: (2*K*d,)
    """
    assert y.ndim == 2, f"y must be (B,d), got {y.shape}"
    d = y.shape[1]
    assert d_lambda == 2 * K * d, f"d_lambda must be 2*K*d. got {d_lambda}, expected {2*K*d}"

    # unpack mu
    mu_m = mu[:, : K * d].view(B, K, d)          # (B,K,d)
    gamma = mu[:, K * d :].view(B, K, d)         # (B,K,d)

    # unpack global lambda_0
    m0, rho0 = unpack_lambda(lambda_0, K, d, device=device)   # (K,d)

    # init locals
    m = m0.view(1, K, d).expand(B, K, d).clone()
    rho = rho0.view(1, K, d).expand(B, K, d).clone()
    alpha = alpha_batch.to(device).clone()

    # constants
    inv_sigma0_2 = (1.0 / (sigma0_diag ** 2)).view(1, 1, d)   # (1,1,d)
    inv_sigma1_2 = (1.0 / (sigma1_diag ** 2)).view(1, 1, d)   # (1,1,d)
    inv_eta_m = 1.0 / eta_m
    inv_eta_s = 1.0 / eta_s

    y_ = y.view(B, 1, d)  # (B,1,d)

    for _ in range(num_steps):
        # ===== (A) alpha/phi closed form =====
        exp_rho = torch.exp(rho)  # (B,K,d)
        C = 0.5 * ((m * m + exp_rho - 2.0 * y_ * m) * inv_sigma0_2).sum(dim=-1)  # (B,K)
        alpha = -C
        phi = torch.softmax(alpha / T, dim=-1)  # (B,K)

        # ===== (B) m closed form (per-dim) =====
        phi_ = phi.unsqueeze(-1)  # (B,K,1)
        denom_m = phi_ * inv_sigma0_2 + (1.0 / n) * inv_sigma1_2 + inv_eta_m  # (B,K,d)
        numer_m = phi_ * (y_ * inv_sigma0_2) + (m0.view(1, K, d) * inv_eta_m) - mu_m
        m = numer_m / denom_m

        # ===== (C) rho newton update (per-dim) =====
        A = phi_ * (0.5 * inv_sigma0_2) + 0.5 * (1.0 / n) * inv_sigma1_2  # (B,K,d)
        rho = newton_rho(
            rho, A, gamma,
            rho0.view(1, K, d),
            inv_eta_s, n,
            num_newton=newton_steps
        )

    # pack lambda per sample: (B,2*K*d)
    lambda_batch = torch.cat([m.reshape(B, K * d), rho.reshape(B, K * d)], dim=1)

    # gradients (phi treated as constant)
    exp_rho = torch.exp(rho)
    grad_m = phi.unsqueeze(-1) * ((m - y_) * inv_sigma0_2) + (1.0 / n) * m * inv_sigma1_2  # (B,K,d)
    grad_rho = exp_rho * (phi.unsqueeze(-1) * (0.5 * inv_sigma0_2) + 0.5 * (1.0 / n) * inv_sigma1_2) - 0.5 / n  # (B,K,d)

    # Hessians (diagonal / elementwise in this factorization)
    hess_m = phi.unsqueeze(-1) * inv_sigma0_2 + (1.0 / n) * inv_sigma1_2  # (B,K,d)
    hess_rho = exp_rho * (phi.unsqueeze(-1) * (0.5 * inv_sigma0_2) + 0.5 * (1.0 / n) * inv_sigma1_2)            # (B,K,d)

    # gradient of local augmented Lagrangian
    gradLm = grad_m + mu_m + (m - m0.view(1, K, d)) * inv_eta_m
    gradLrho = grad_rho + gamma + (rho - rho0.view(1, K, d)) * inv_eta_s

    gradL = torch.cat([gradLm.reshape(B, -1), gradLrho.reshape(B, -1)], dim=1)  # (B,2*K*d)
    grad_norm_per_sample = torch.linalg.norm(gradL, dim=1)
    grad_norm_mean = grad_norm_per_sample.mean()

    return alpha, lambda_batch, grad_m, grad_rho, hess_m, hess_rho, grad_norm_mean

def remap_labels_to_0K(labels: torch.Tensor):
    # labels: (n,)
    uniq = torch.unique(labels)
    inv = torch.bucketize(labels, uniq)  # works if uniq sorted; torch.unique is sorted by default
    # safer alternative:
    # uniq, inv = torch.unique(labels, return_inverse=True)
    return inv, uniq.numel()

@torch.no_grad()
def estimate_sigma0_sigma1_from_initlabels(y: torch.Tensor, initial_labels: torch.Tensor, eps=1e-12):
    """
    y: (n,d)
    initial_labels: (n,) possibly not 0..K-1
    returns:
      sigma0_diag: (d,)  average within-cluster std (per-dimension)
      sigma1_diag: (d,)  std across cluster means (per-dimension)
    """
    device = y.device
    n, d = y.shape

    # --- ensure labels are 0..K-1
    labels, K = remap_labels_to_0K(initial_labels.to(device))

    # --- counts per cluster: (K,)
    ones = torch.ones(n, device=device, dtype=y.dtype)
    counts = torch.zeros(K, device=device, dtype=y.dtype).index_add_(0, labels, ones)  # (K,)
    counts_clamped = counts.clamp_min(1.0)  # avoid div0

    # --- cluster sums: (K,d)
    sum_y = torch.zeros(K, d, device=device, dtype=y.dtype)
    sum_y.index_add_(0, labels, y)

    # --- cluster means: (K,d)
    mean_y = sum_y / counts_clamped.unsqueeze(1)

    # --- within-cluster variance:
    # E[y^2] - (E[y])^2
    sum_y2 = torch.zeros(K, d, device=device, dtype=y.dtype)
    sum_y2.index_add_(0, labels, y * y)
    Ey2 = sum_y2 / counts_clamped.unsqueeze(1)
    var_within = (Ey2 - mean_y * mean_y).clamp_min(0.0)  # numeric safety
    std_within = torch.sqrt(var_within + eps)            # (K,d)

    # (1) sigma0_diag: average within-cluster std over clusters
    # 你也可以用 counts 加权平均（见下方可选项）
    sigma0_diag_median = std_within.median(dim=0).values

    # (2) sigma1_diag: std of cluster means across clusters (per dimension)
    # 注意：这是“类别均值在类别维度上的离散程度”
    sigma1_diag = mean_y.std(dim=0, unbiased=True)       # (d,)

    # print the norm of sigma0_diag_median and sigma1_diag
    print(f"norm of sigma0_diag_median: {torch.linalg.norm(sigma0_diag_median)}")
    print(f"norm of sigma1_diag: {torch.linalg.norm(sigma1_diag)}")

    return sigma0_diag_median*2, sigma1_diag, mean_y, std_within, counts

# =========================
# main
# =========================
if __name__ == "__main__":
    # ---------------- settings ----------------
    eta_m = 0.1
    eta_s = 10000
    max_iter = 1000
    sample_size = 1000
    algorithm = "ours"  # "ours"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    T = 1.0 # temperature parameter

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ------------- load -------------
    root = Path(__file__).resolve().parents[3]
    processed_dir = root / "experiments" / "real_data" / "data" / "processed"
    figures_dir = root / "experiments" / "real_data" / "figures"
    results_dir = root / "experiments" / "real_data" / "results" / "mfvi_highdim"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(processed_dir / "tensor_pca_xy_true_initial_E165_E1S3.pt", map_location=device)
    y = data["pca_data"] # (n, d)
    z_true = data["true_labels"]
    initial_labels = data["initial_labels"]
    coords = data["spatial"]
    # set model parameters
    n = y.shape[0]
    d = y.shape[1]
    K = torch.unique(z_true).numel()

    sigma0_diag, sigma1_diag, mean_k, std_k, counts = estimate_sigma0_sigma1_from_initlabels(y, initial_labels)
    print(f"sigma0_diag: {sigma0_diag.shape}, sigma1_diag: {sigma1_diag.shape}, mean_k: {mean_k.shape}, std_k: {std_k.shape}, counts: {counts.shape}")

    # ---------------- constants for init ----------------
    const_term = compute_const_term_diag(y, sigma0_diag, sigma1_diag, K)

    y_mean = y.mean(dim=0)  # (d,)
    y_std = y.std(dim=0)    # (d,)
    y_var = y.var(dim=0)    # (d,)

    # ---------------- logs ----------------
    loss_list = []
    ari_list = []
    time_list = []
    grad_norm_list = []
    residual_list = []

    time_start = time.time()

    # ---------------- algorithm ----------------
    if algorithm == "ours":
        # init alpha, lambda_0, mu, h_t
        alpha_all = 1e-2 * torch.randn(n, K, device=device)

        m0 = y_mean.view(1, d) + 0.5 * y_std.view(1, d) * torch.randn(K, d, device=device)  # (K,d)
        rho0 = torch.log(y_var + 1e-12).view(1, d) + 1e-2 * torch.randn(K, d, device=device)  # (K,d)

        lambda_0 = torch.cat([m0.reshape(-1), rho0.reshape(-1)], dim=0)  # (2*K*d,)
        d_lambda = 2 * K * d

        mu = torch.zeros(n, d_lambda, device=device)  # (n,2*K*d)
        h_t = torch.zeros(d_lambda, device=device)

        primal_residual = torch.tensor(1e6, device=device)

        for iter in range(max_iter + 1):
            if iter % 10 == 0:
                elbo = elbo_batch_diag(alpha_all, lambda_0, y, sigma0_diag, sigma1_diag, n) + const_term
                loss_list.append(elbo.item())

                phi = torch.softmax(alpha_all / T, dim=-1)
                z_pred = torch.argmax(phi, dim=-1)
                ari = adjusted_rand_score(z_pred.detach().to("cpu", dtype=torch.int64),
                                          z_true.detach().to("cpu", dtype=torch.int64))
                ari = float(ari)
                ari_list.append(ari)

                ari_sk = sk_ari(
                        z_true.detach().cpu().numpy(),
                        z_pred.detach().cpu().numpy(),
                    )

                residual_list.append(primal_residual.item())

                time_passed = time.time() - time_start
                time_list.append(time_passed)

                print(
                    f"iter {iter}/{max_iter}, elbo = {elbo:.6e}, ari = {ari:.6e}, "
                    f"ari_sk = {ari_sk:.6e}, residual = {primal_residual:.6e}, time_passed = {time_passed:.6f}"
                )

            # sample minibatch
            St = torch.randperm(n, device=device)[:sample_size]
            mu_batch = mu[St]                 # (B,2*K*d)
            y_batch = y[St]                   # (B,d)
            alpha_batch = alpha_all[St]       # (B,K)

            alpha_batch, lambda_batch, grad_m, grad_rho, hess_m, hess_rho, grad_norm_mean = local_update_diag(
                mu_batch, y_batch, alpha_batch, lambda_0,
                sample_size, K, d_lambda, eta_m, eta_s, n, device,
                sigma0_diag, sigma1_diag,
                T,
                num_steps=10, newton_steps=5
            )

            alpha_all[St] = alpha_batch

            # primal residual: ||lambda_batch - lambda_0||
            lambda0_row = lambda_0.view(1, -1)
            lambda_diff = lambda_batch - lambda0_row                    # (B,2*K*d)
            primal_residual = torch.linalg.norm(lambda_diff)

            # dual update for sampled indices
            mu[St, :K * d] += lambda_diff[:, :K * d] / eta_m
            mu[St, K * d:] += lambda_diff[:, K * d:] / eta_s

            # update global lambda_0 (your original scheme)
            diff_mean = lambda_diff.sum(dim=0) / n                      # (2*K*d,)
            lambda_local_mean = lambda_batch.mean(dim=0)                # (2*K*d,)
            h_t = h_t + diff_mean
            lambda_0 = lambda_local_mean + h_t

            # optional: autograd check of gradient norm
            if iter % 100 == 0:
                alpha_tmp = alpha_all.detach().clone().requires_grad_(True)
                lambda_0_tmp = lambda_0.detach().clone().requires_grad_(True)
                loss_no_const = elbo_batch_diag(alpha_tmp, lambda_0_tmp, y, sigma0_diag, sigma1_diag, n)
                grad_alpha, grad_lambda = torch.autograd.grad(
                    loss_no_const,
                    [alpha_tmp, lambda_0_tmp],
                    create_graph=False,
                    retain_graph=False
                )
                grad_norm = torch.sqrt(grad_alpha.pow(2).sum() + grad_lambda.pow(2).sum())
                grad_norm_list.append(grad_norm.item())
                print(f"iter {iter}/{max_iter}, grad_norm = {grad_norm:.6e}")

        # ---------------- save outputs (optional) ----------------
        # plot the final labels

        coords_cpu = coords.detach().cpu().numpy()
        z_pred_cpu = z_pred.detach().cpu().numpy().astype(int)

        c1 = plt.get_cmap("tab20").colors
        c2 = plt.get_cmap("tab20b").colors
        colors = list(c1) + list(c2)   # 40 colors
        cmap23 = ListedColormap(colors[:23])
        plt.figure()
        plt.scatter(coords_cpu[:, 0], coords_cpu[:, 1], s=5, c=z_pred_cpu, cmap=cmap23)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Predicted labels")
        plt.savefig(figures_dir / "highdim_labels.png")
        # print the number of unique labels
        print(f"Number of unique labels: {len(np.unique(z_pred_cpu))}")
        # print the number of each cluster
        for i in range(K):
            print(f"Cluster {i}: {len(np.where(z_pred_cpu == i)[0])}")

        np.save(results_dir / f"{algorithm}_samplesize_{sample_size}_loss.npy", np.array(loss_list))
        np.save(results_dir / f"{algorithm}_samplesize_{sample_size}_ari.npy", np.array(ari_list))
        np.save(results_dir / f"{algorithm}_samplesize_{sample_size}_time.npy", np.array(time_list))
        np.save(results_dir / f"{algorithm}_samplesize_{sample_size}_residual.npy", np.array(residual_list))
        np.save(results_dir / f"{algorithm}_samplesize_{sample_size}_grad_norm.npy", np.array(grad_norm_list))
