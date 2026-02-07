import torch
from sklearn.neighbors import NearestNeighbors
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

seed = args.seed

# local fisher LDA
def local_fisher_tangents(coords, labels, neighbors, lam=1e-2, min_same=5, min_diff=5, eps=1e-12, enforce_positive_x=True):
    N = coords.shape[0]
    tangents = torch.zeros((N, 2), device=coords.device, dtype=coords.dtype)
    I = torch.eye(2, device=coords.device, dtype=coords.dtype)

    for i in range(N):
        nbr = neighbors[i]

        Xi = coords[nbr]                      # (k,2)
        yi = labels[i]
        y_nbr = labels[nbr]                   # (k,)

        mask_same = (y_nbr == yi)
        mask_diff = ~mask_same

        if mask_same.sum().item() < min_same or mask_diff.sum().item() < min_diff:
            tangents[i] = torch.zeros(2, device=coords.device, dtype=coords.dtype) # return 0 vector
            continue

        Xp = Xi[mask_same]                    # (k1,2)
        Xq = Xi[mask_diff]                    # (k0,2)

        mu_p = Xp.mean(dim=0)                 # (2,)
        mu_q = Xq.mean(dim=0)                 # (2,)

        # within-class scatter matrix Sw = sum (x-mu)(x-mu)^T
        Cp = Xp - mu_p                        # (k1,2)
        Cq = Xq - mu_q                        # (k0,2)
        Sw = Cp.T @ Cp + Cq.T @ Cq            # (2,2)

        # regularization to avoid singularity
        Sw_reg = Sw + lam * I

        dmu = (mu_p - mu_q)                   # (2,)

        try:
            w = torch.linalg.solve(Sw_reg, dmu)   # (2,)
        except RuntimeError:
            tangents[i] = torch.zeros(2, device=coords.device, dtype=coords.dtype) # return 0 vector
            continue

        norm_w = torch.linalg.norm(w)
        if norm_w < 1e-6:
            tangents[i] = torch.zeros(2, device=coords.device, dtype=coords.dtype) # return 0 vector
            continue

        # smooth direction is perpendicular to w
        t = torch.stack([-w[1], w[0]])         # (2,)
        t = t / (torch.linalg.norm(t) + eps)

        if enforce_positive_x and t[0] < 0:
            t = -t

        tangents[i] = t
    return tangents


# read the data
ROOT = Path(__file__).resolve().parents[3]
processed_dir = ROOT / "experiments" / "real_data" / "data" / "processed"
intermediate_dir = ROOT / "experiments" / "real_data" / "data" / "intermediate"

in_path = processed_dir / f"tensor_pca_xy_true_initial_E165_E1S3_seed_{seed}.pt"
data = torch.load(in_path)
pca_data = data["pca_data"]
coords = data["spatial"]
true_labels = data["true_labels"]
initial_labels = data["initial_labels"]

coords_np = coords.detach().cpu().numpy()
nn = NearestNeighbors(n_neighbors=100, algorithm='kd_tree').fit(coords_np)
_, ind = nn.kneighbors(coords_np)
nbr_idx = torch.from_numpy(ind[:, 1:]).long().to(coords.device)

tangents = local_fisher_tangents(coords, initial_labels, nbr_idx)

# save the tangents
os.makedirs(intermediate_dir, exist_ok=True)
out_path = intermediate_dir / f"tangents_E165_E1S3_seed_{seed}.pt"
torch.save(tangents, out_path)
print("Saved to:", out_path)

# n = pca_data.shape[0]
# weights = torch.zeros((n, n), device=pca_data.device, dtype=pca_data.dtype)
