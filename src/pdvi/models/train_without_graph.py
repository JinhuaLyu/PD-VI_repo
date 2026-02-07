import torch
from torchmetrics.clustering import AdjustedRandScore
import time
import math
import numpy as np
from pathlib import Path

# =========================
# Utilities
# =========================

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def newton_rho(rho, A, gamma, rho0, inv_eta, n, num_newton=5):
    """
    Elementwise Newton for rho in (B,K,d).
    """
    c = gamma - 0.5 / n
    for _ in range(num_newton):
        exp_rho = torch.exp(rho)
        f  = A * exp_rho + c + inv_eta * (rho - rho0)
        fp = A * exp_rho + inv_eta
        rho = rho - f / fp
    return rho

def compute_const_term_md(x, sigma0_vec, sigma1_vec, K: int):
    """
    Constant term so that elbo_batch_md + const_term matches your convention.
    x: (n,d)
    sigma0_vec, sigma1_vec: (d,) standard deviations (diag cov fixed)
    """
    n, d = x.shape
    sigma0_vec = torch.as_tensor(sigma0_vec, device=x.device, dtype=x.dtype).reshape(-1)
    sigma1_vec = torch.as_tensor(sigma1_vec, device=x.device, dtype=x.dtype).reshape(-1)
    assert sigma0_vec.numel() == d and sigma1_vec.numel() == d

    # const1: sum over dimensions inside log N(·|·,Σ0) normalizer + log K
    const1 = n * (0.5 * torch.log(2.0 * math.pi * sigma0_vec.pow(2)).sum().item() + math.log(K))
    # const2: sum_i 0.5 x_i^T Σ0^{-1} x_i
    const2 = (x.pow(2) / (2.0 * sigma0_vec.pow(2)[None, :])).sum().item()
    # const3: prior normalizer-ish under your 1D convention; extend dimensionwise
    const3 = (K / 2.0) * (torch.log(sigma1_vec.pow(2)).sum().item() - d * 1.0)
    return const1 + const2 + const3

def elbo_batch_md(alpha_batch, lambda_0, x, sigma0_vec, sigma1_vec, n: int):
    """
    alpha_batch: (B,K)
    lambda_0: (2*K*d,) where first K*d are m0 and last K*d are rho0
    x: (B,d)  (can also pass full (n,d) with alpha_all)
    sigma0_vec, sigma1_vec: (d,) standard deviations
    returns scalar sum over batch
    """
    B, K = alpha_batch.shape
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    d = x.shape[1]

    sigma0_vec = torch.as_tensor(sigma0_vec, device=x.device, dtype=x.dtype).reshape(-1)
    sigma1_vec = torch.as_tensor(sigma1_vec, device=x.device, dtype=x.dtype).reshape(-1)
    assert sigma0_vec.numel() == d and sigma1_vec.numel() == d

    assert lambda_0.numel() == 2 * K * d, f"lambda_0 must have length 2*K*d={2*K*d}"
    m0   = lambda_0[:K*d].view(K, d)     # (K,d)
    rho0 = lambda_0[K*d:].view(K, d)     # (K,d)

    phi = torch.softmax(alpha_batch, dim=-1)          # (B,K)
    log_phi = torch.log_softmax(alpha_batch, dim=-1)  # (B,K)

    x_b   = x[:, None, :]          # (B,1,d)
    m_b   = m0[None, :, :]         # (1,K,d)
    rho_b = rho0[None, :, :]       # (1,K,d)
    s2_b  = torch.exp(rho_b)       # (1,K,d)

    inv_sigma0_2 = 1.0 / (sigma0_vec.pow(2))[None, None, :]  # (1,1,d)

    # data_term_{ik} = 0.5 * sum_j (m_kj^2 + s_kj^2 - 2 x_ij m_kj)/sigma0_j^2
    data_term = (0.5 * (m_b*m_b + s2_b - 2.0 * x_b * m_b) * inv_sigma0_2).sum(dim=-1)  # (B,K)

    # vi term: -0.5/n * sum_j rho_kj + 0.5/n * sum_j (m_kj^2 + s_kj^2)/sigma1_j^2
    inv_sigma1_2 = 1.0 / (sigma1_vec.pow(2))[None, :]  # (1,d)
    vi_term_k = (-0.5 / n) * rho0.sum(dim=-1) + (0.5 / n) * ((m0*m0 + torch.exp(rho0)) * inv_sigma1_2).sum(dim=-1)  # (K,)
    vi_term = vi_term_k[None, :]  # (1,K)

    inside = phi * (log_phi + data_term) + vi_term
    return inside.sum()

def local_update_md(mu, x, alpha_batch, lambda_0, B, K, d_lambda,
                    eta_m, eta_s, n, device,
                    sigma0_vec, sigma1_vec,
                    num_steps=10, newton_steps=5):
    """
    mu: (B, 2*K*d)
    x: (B,d)
    alpha_batch: (B,K)
    lambda_0: (2*K*d,)
    Returns:
      alpha: (B,K)
      lambda_batch: (B,2*K*d)
      grad_m, grad_rho, hess_m, hess_rho: (B,K,d)
      grad_norm_mean: scalar
    """
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    d = x.shape[1]

    sigma0_vec = torch.as_tensor(sigma0_vec, device=device, dtype=x.dtype).reshape(-1)
    sigma1_vec = torch.as_tensor(sigma1_vec, device=device, dtype=x.dtype).reshape(-1)
    assert sigma0_vec.numel() == d and sigma1_vec.numel() == d

    assert lambda_0.numel() == 2 * K * d
    m0   = lambda_0[:K*d].view(K, d).to(device)      # (K,d)
    rho0 = lambda_0[K*d:].view(K, d).to(device)      # (K,d)

    mu_m   = mu[:, :K*d].view(B, K, d)   # (B,K,d)
    mu_rho = mu[:, K*d:].view(B, K, d)   # (B,K,d)

    # init locals
    m   = m0[None, :, :].expand(B, K, d).clone()
    rho = rho0[None, :, :].expand(B, K, d).clone()
    alpha = alpha_batch.to(device).clone()

    inv_sigma0_2 = 1.0 / (sigma0_vec.pow(2))[None, None, :]  # (1,1,d)
    inv_sigma1_2 = 1.0 / (sigma1_vec.pow(2))[None, None, :]  # (1,1,d)
    inv_n_sigma1_2 = inv_sigma1_2 / n                        # (1,1,d)

    inv_eta_m = 1.0 / eta_m
    inv_eta_s = 1.0 / eta_s

    x_b = x[:, None, :]  # (B,1,d)

    for _ in range(num_steps):
        # (A) alpha/phi closed form
        C = (0.5 * (m*m + torch.exp(rho) - 2.0 * x_b * m) * inv_sigma0_2).sum(dim=-1)  # (B,K)
        alpha = -C
        phi = torch.softmax(alpha, dim=-1)  # (B,K)
        phi_b = phi[:, :, None]             # (B,K,1)

        # (B) m closed form (elementwise)
        denom_m = phi_b * inv_sigma0_2 + inv_n_sigma1_2 + inv_eta_m  # (B,K,d)
        numer_m = phi_b * (x_b * inv_sigma0_2) + (m0[None, :, :] * inv_eta_m) - mu_m
        m = numer_m / denom_m

        # (C) rho newton update (elementwise)
        A = phi_b * (0.5 * inv_sigma0_2) + (0.5 * inv_n_sigma1_2)  # (B,K,d)
        rho = newton_rho(rho, A, mu_rho, rho0[None, :, :], inv_eta_s, n, num_newton=newton_steps)

    # pack
    lambda_batch = torch.cat([m.reshape(B, K*d), rho.reshape(B, K*d)], dim=1)

    # gradients / Hessians of f (phi treated constant)
    phi_b = torch.softmax(alpha, dim=-1)[:, :, None]  # (B,K,1)
    grad_m = phi_b * ((m - x_b) * inv_sigma0_2) + m * inv_n_sigma1_2  # (B,K,d)

    exp_rho = torch.exp(rho)
    grad_rho = exp_rho * (phi_b * (0.5 * inv_sigma0_2) + 0.5 * inv_n_sigma1_2) - 0.5 / n  # (B,K,d)

    hess_m = phi_b * inv_sigma0_2 + inv_n_sigma1_2  # (B,K,d)
    hess_rho = exp_rho * (phi_b * (0.5 * inv_sigma0_2) + 0.5 * inv_n_sigma1_2)  # (B,K,d)

    # local augmented Lagrangian grads (for monitoring)
    gradLm = grad_m + mu_m + (m - m0[None, :, :]) * inv_eta_m
    gradLrho = grad_rho + mu_rho + (rho - rho0[None, :, :]) * inv_eta_s
    gradL = torch.cat([gradLm.reshape(B, K*d), gradLrho.reshape(B, K*d)], dim=1)

    grad_norm_mean = torch.linalg.norm(gradL, dim=1).mean()

    return alpha, lambda_batch, grad_m, grad_rho, hess_m, hess_rho, grad_norm_mean


# =========================
# Main
# =========================

if __name__ == "__main__":

    # -------- config --------
    root = Path(__file__).resolve().parents[3]
    DATA_PATH = root / "experiments" / "real_data" / "data" / "processed" / "tensor_pca_xy_true_initial_E165_E1S3.pt"

    algorithm = "ours"
    seed = 42
    torch.set_default_dtype(torch.float64)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # needs to be tuned
    eta_m = 1
    eta_s = 1
    max_iter = 10000
    sample_size = 5000

    # local solver hyperparams
    num_steps = 10
    newton_steps = 5

    log_every = 10
    autograd_every = 100

    # -------- load data --------
    data = torch.load(DATA_PATH, map_location=device)
    coords = data["spatial"].to(device)
    data_pca = data["pca_data"].to(device)          # (n,d)
    true_labels = data["true_labels"].to(device)    # (n,)
    initial_labels = data["initial_labels"].to(device)

    n = data_pca.shape[0]
    d = data_pca.shape[1]
    K = torch.unique(true_labels).numel()

    print(f"n={n}, d={d}, K={K}")

    # -------- fixed diagonal hyperparams Σ0, Σ1 --------
    # Choice 1 (simple & usually stable): set sigma0 per-dim to sample std of data_pca
    # and sigma1 a larger prior scale (e.g., 10x)
    x_mean = data_pca.mean(dim=0)  # (d,)
    x_std = data_pca.std(dim=0).clamp_min(1e-12)
    x_var = data_pca.var(dim=0).clamp_min(1e-12)

    sigma0_vec = 0.01 * x_std.clone()                   # (d,)  observation noise std (fixed)
    sigma1_vec = (0.05 * x_std).clone()          # (d,)  prior std (fixed) -- adjust if needed

    # If you prefer manual constants:
    # sigma0_vec = torch.ones(d, device=device) * 1.0
    # sigma1_vec = torch.ones(d, device=device) * 10.0

    const_term = compute_const_term_md(data_pca, sigma0_vec, sigma1_vec, K)

    # -------- init parameters --------
    alpha_all = 1e-2 * torch.randn(n, K, device=device)

    # Initialize m0, rho0 as (K,d)
    m0 = x_mean[None, :] + 0.5 * x_std[None, :] * torch.randn(K, d, device=device)
    rho0 = torch.log(x_var)[None, :] + 1e-2 * torch.randn(K, d, device=device)

    lambda_0 = torch.cat([m0.reshape(-1), rho0.reshape(-1)], dim=0)  # (2*K*d,)
    d_lambda = 2 * K * d

    mu = torch.zeros(n, d_lambda, device=device)
    h_t = torch.zeros(d_lambda, device=device)

    # -------- logs --------
    loss_list = []
    ari_list = []
    time_list = []
    residual_list = []
    grad_norm_list = []

    metric = AdjustedRandScore().to(device)
    time_start = time.time()

    primal_residual = torch.tensor(1e6, device=device, dtype=torch.float64)

    # -------- training loop --------
    assert algorithm == "ours"

    for it in range(max_iter + 1):

        if it % log_every == 0:
            elbo = elbo_batch_md(alpha_all, lambda_0, data_pca, sigma0_vec, sigma1_vec, n) + const_term
            loss_list.append(elbo.item())

            phi_all = torch.softmax(alpha_all, dim=-1)
            z_pred = torch.argmax(phi_all, dim=-1)
            ari = metric(z_pred, true_labels)
            ari_list.append(ari.item())

            residual_list.append(primal_residual.item())

            t = time.time() - time_start
            time_list.append(t)

            print(f"iter {it}/{max_iter} | elbo={elbo:.6e} | ari={ari.item():.6e} | residual={primal_residual:.6e} | time={t:.3f}s")

        # minibatch indices
        B = min(sample_size, n)
        St = torch.randperm(n, device=device)[:B]

        mu_batch = mu[St]              # (B,2Kd)
        x_batch = data_pca[St]         # (B,d)
        alpha_batch = alpha_all[St]    # (B,K)

        alpha_new, lambda_batch, grad_m, grad_rho, hess_m, hess_rho, grad_norm_mean = local_update_md(
            mu_batch, x_batch, alpha_batch, lambda_0,
            B, K, d_lambda,
            eta_m, eta_s, n, device,
            sigma0_vec=sigma0_vec, sigma1_vec=sigma1_vec,
            num_steps=num_steps, newton_steps=newton_steps
        )

        # write back alpha
        alpha_all[St] = alpha_new

        # primal residual
        lambda_diff = lambda_batch - lambda_0[None, :]  # (B,2Kd)
        primal_residual = torch.linalg.norm(lambda_diff)

        # dual update (blockwise)
        Kd = K * d
        mu[St, :Kd] += lambda_diff[:, :Kd] / eta_m
        mu[St, Kd:] += lambda_diff[:, Kd:] / eta_s

        # global aggregation
        diff_mean = lambda_diff.sum(dim=0) / n         # (2Kd,)
        lambda_local_mean = lambda_batch.mean(dim=0)   # (2Kd,)
        h_t = h_t + diff_mean
        lambda_0 = lambda_local_mean + h_t

        # optional autograd check
        if it % autograd_every == 0 and it > 0:
            alpha_tmp = alpha_all.detach().clone().requires_grad_(True)
            lambda_tmp = lambda_0.detach().clone().requires_grad_(True)
            loss = elbo_batch_md(alpha_tmp, lambda_tmp, data_pca, sigma0_vec, sigma1_vec, n)
            grad_alpha, grad_lambda = torch.autograd.grad(loss, [alpha_tmp, lambda_tmp], create_graph=False, retain_graph=False)
            grad_norm = torch.sqrt(grad_alpha.pow(2).sum() + grad_lambda.pow(2).sum())
            grad_norm_list.append(grad_norm.item())
            print(f"  [autograd] iter {it} | grad_norm={grad_norm:.6e} | local_gradL_mean={grad_norm_mean.item():.6e}")

    # -------- optional save --------
    # np.save("loss.npy", np.array(loss_list))
   
