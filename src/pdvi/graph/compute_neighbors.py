import os
import numpy as np
import torch
import time
from pathlib import Path

start_time = time.time()
# -----------------------------
# Params
# -----------------------------
width = 8
height = 2
radius = 2
eps = 1e-12

half_w = width / 2
half_h = height / 2

ROOT = Path(__file__).resolve().parents[3]
processed_dir = ROOT / "experiments" / "real_data" / "data" / "processed"
intermediate_dir = ROOT / "experiments" / "real_data" / "data" / "intermediate"

tangents_path = intermediate_dir / "tangents_E165_E1S3.pt"
in_path = processed_dir / "tensor_pca_xy_true_initial_E165_E1S3.pt"
out_path = intermediate_dir / "neighbors_25_E165_E1S3.pt"

# -----------------------------
# Load
# -----------------------------
tangents = torch.load(tangents_path, map_location="cpu")  # (N,2)
data = torch.load(in_path, map_location="cpu")
coords = data["spatial"]  # (N,2)

N = coords.shape[0]
assert coords.shape == (N, 2) and tangents.shape == (N, 2)

# normal vectors
normal = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)  # (N,2)

# -----------------------------
# 1) Build uniform grid (cell list) to get candidates fast
# -----------------------------
# Choose cell size so that searching nearby cells covers the neighborhood.
# Safe choice: cell size = max(width, height, 2*radius)
cell = float(max(width, height, 2 * radius))
xy = coords.numpy()  # (N,2) float
xmin, ymin = xy.min(axis=0)

# integer cell coordinates
cx = np.floor((xy[:, 0] - xmin) / cell).astype(np.int32)
cy = np.floor((xy[:, 1] - ymin) / cell).astype(np.int32)

# sort points by (cx, cy) so each cell becomes a contiguous block
# key = (cx, cy) pair encoded into a single int64
key = (cx.astype(np.int64) << 32) ^ (cy.astype(np.int64) & 0xffffffff)
order = np.argsort(key)
key_sorted = key[order]

# identify cell boundaries
unique_keys, start_idx, counts = np.unique(key_sorted, return_index=True, return_counts=True)
# build dict: cell_key -> (start, end) in `order`
cell2range = {int(k): (int(s), int(s + c)) for k, s, c in zip(unique_keys, start_idx, counts)}

# helper: get candidate indices from neighboring cells
def gather_candidates(i):
    cxi, cyi = int(cx[i]), int(cy[i])
    cand = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            k = (np.int64(cxi + dx) << 32) ^ (np.int64(cyi + dy) & 0xffffffff)
            r = cell2range.get(int(k))
            if r is None:
                continue
            s, e = r
            cand.append(order[s:e])
    if len(cand) == 0:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(cand, axis=0).astype(np.int64)

# -----------------------------
# 2) Build directed edges (i -> j) using candidates + your geometry filter
# -----------------------------
src_list = []
dst_list = []

coords_t = coords  # torch (N,2)
for i in range(N):
    cand = gather_candidates(i)
    if cand.size == 0:
        continue

    # exclude self early
    cand = cand[cand != i]
    if cand.size == 0:
        continue

    cand_t = torch.from_numpy(cand)  # cpu int64
    diff = coords_t[cand_t] - coords_t[i]  # (M,2)

    if torch.linalg.norm(tangents[i]) < eps:
        # radius ball
        d = torch.linalg.norm(diff, dim=1)
        keep = d <= radius
    else:
        u = diff @ tangents[i]  # (M,)
        v = diff @ normal[i]    # (M,)
        keep = (torch.abs(u) <= half_w) & (torch.abs(v) <= half_h)

    if keep.any():
        nbr = cand_t[keep].numpy()
        src_list.append(np.full(nbr.shape[0], i, dtype=np.int64))
        dst_list.append(nbr.astype(np.int64))

# concatenate edges
if len(src_list) == 0:
    raise RuntimeError("No edges found. Check width/height/radius or coordinate scale.")

src = np.concatenate(src_list, axis=0)
dst = np.concatenate(dst_list, axis=0)

# -----------------------------
# 3) Symmetrize by union: add reverse edges and deduplicate
# -----------------------------
src2 = np.concatenate([src, dst], axis=0)
dst2 = np.concatenate([dst, src], axis=0)

# remove self-loops if any
mask = src2 != dst2
src2, dst2 = src2[mask], dst2[mask]

# deduplicate directed edges via hashing
key_e = src2 * np.int64(N) + dst2
key_e = np.unique(key_e)

src_u = (key_e // np.int64(N)).astype(np.int64)
dst_u = (key_e %  np.int64(N)).astype(np.int64)

# -----------------------------
# 4) Build CSR (indptr, indices) for fast access + GPU-ready
# -----------------------------
order_e = np.argsort(src_u)
src_u = src_u[order_e]
dst_u = dst_u[order_e]

deg = np.bincount(src_u, minlength=N).astype(np.int64)
indptr = np.zeros(N + 1, dtype=np.int64)
indptr[1:] = np.cumsum(deg)

indices = dst_u  # already grouped by src

# statistics
deg_stats = np.array(deg)
print("deg mean/p50/p95/max:",
      float(deg_stats.mean()),
      float(np.percentile(deg_stats, 50)),
      float(np.percentile(deg_stats, 95)),
      int(deg_stats.max()))

# -----------------------------
# 5) Save
# -----------------------------
os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.save(
    {
        "indptr": torch.from_numpy(indptr),
        "indices": torch.from_numpy(indices),
        "meta": {
            "N": N,
            "width": width,
            "height": height,
            "radius": radius,
            "cell": cell,
            "symmetrize": "union",
        },
    },
    out_path,
)
print("Saved CSR neighbors to:", out_path)

# -----------------------------
# 6) How to use on GPU later
# -----------------------------
# load = torch.load(out_path, map_location="cpu")
# indptr = load["indptr"].to("cuda")
# indices = load["indices"].to("cuda")
# i = 123
# nbr_i = indices[indptr[i].item():indptr[i+1].item()]   # neighbors of i

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
