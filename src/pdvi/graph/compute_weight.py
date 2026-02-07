import torch
import time
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

seed = args.seed

start_time = time.time()
neighbors_num = 25
eps = 1e-10

# ------------- load -------------
ROOT = Path(__file__).resolve().parents[3]
processed_dir = ROOT / "experiments" / "real_data" / "data" / "processed"
intermediate_dir = ROOT / "experiments" / "real_data" / "data" / "intermediate"

tangents = torch.load(intermediate_dir / f"tangents_E165_E1S3_seed_{seed}.pt", map_location="cpu")
data = torch.load(processed_dir / f"tensor_pca_xy_true_initial_E165_E1S3_seed_{seed}.pt", map_location="cpu")

coords = data["spatial"]
z = data["pca_data"]  
nbr = torch.load(intermediate_dir / f"neighbors_{neighbors_num}_E165_E1S3.pt", map_location="cpu")
indptr = nbr["indptr"]
indices = nbr["indices"]

# ------------- move to GPU -------------
device = "cuda"
coords = coords.to(device)
tangents = tangents.to(device)
z = z.to(device)
z = z / (torch.linalg.norm(z, dim=1, keepdim=True) + eps)

indptr_cpu = indptr
indices = indices.to(device)

N = coords.shape[0]
nnz = indices.numel()
print("N =", N, "nnz =", nnz)

# ------------- compute weights on edges only -------------
lambda_similarity = 1.0

values = torch.empty((nnz,), device=device, dtype=coords.dtype)

for i in range(N):
    start = int(indptr_cpu[i])
    end = int(indptr_cpu[i + 1])
    k = end - start
    nbr_i = indices[start:end]              # (k,)
    xi = coords[i]                          # (2,)
    ti = tangents[i]                        # (2,)
    zi = z[i]                        # (r,)

    xj = coords[nbr_i]                      # (k,2)
    tj = tangents[nbr_i]                    # (k,2)
    zj = z[nbr_i]                    # (k,r)


    e = xj - xi                             # (k,2)
    dist = torch.linalg.norm(e, dim=1).clamp_min(eps)  # (k,)
    u = e / dist[:, None]                   # (k,2)

    dot_tt = (ti[None, :] * tj).sum(dim=1)  # (k,)
    tan = torch.where(dot_tt[:, None] < 0, (ti - tj) * 0.5, (ti + tj) * 0.5)  # (k,2)

    w_geom = torch.abs((u * tan).sum(dim=1))            # (k,)

    # cosine similarity in data space
    cos_ij = (zj * zi[None, :]).sum(dim=1)                 # (k,)

    values[start:end] = w_geom + lambda_similarity * (cos_ij + 1.0)       # (k,)

# ------------- save CSR weights -------------
out_path = intermediate_dir / f"weights_{neighbors_num}_lambda_{lambda_similarity}_E165_E1S3_seed_{seed}.pt"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

torch.save(
    {
        "indptr":  indptr_cpu.detach().cpu(),
        "indices": indices.detach().cpu(),
        "weights": values.detach().cpu(),
        "meta": {
            "N": int(N),
            "lambda_similarity": float(lambda_similarity),
            "eps": float(eps),
            "format": "CSR (indptr, indices, values aligned)",
        },
    },
    out_path,
)

end_time = time.time()
print("Saved to:", out_path)
print(f"Time taken: {end_time - start_time:.2f} seconds")
