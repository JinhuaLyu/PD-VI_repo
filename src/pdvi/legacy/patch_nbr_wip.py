import os
import torch

nx, ny = 5, 5
B = nx * ny  # size of patches
# ------------- load -------------
data = torch.load("../data/sample_data/tensor_pca_xy_true_initial_E165_E1S3.pt", map_location=device)
coords = data["spatial"].to(device)
data_pca = data["pca_data"].to(device) # (n, d)
true_labels = data["true_labels"].to(device)
initial_labels = data["initial_labels"].to(device)
# set model parameters
n = data_pca.shape[0]
d = data_pca.shape[1]
K = torch.unique(true_labels).numel()


# -------------------- cut the data into patches --------------------
x_edges = torch.linspace(coords[:, 0].min(), coords[:, 0].max(), steps=nx+1, device=coords.device)
y_edges = torch.linspace(coords[:, 1].min(), coords[:, 1].max(), steps=ny+1, device=coords.device)
all_idx = torch.arange(n, device=device)

patch_indices = [None] * B   # patch_indices[b] is a 1D LongTensor of indices

for ix in range(nx):
    x_lo, x_hi = x_edges[ix], x_edges[ix + 1]
    in_x = (coords[:, 0] >= x_lo) & (coords[:, 0] < x_hi if ix < nx - 1 else coords[:, 0] <= x_hi)
    for iy in range(ny):
        y_lo, y_hi = y_edges[iy], y_edges[iy + 1]
        in_y = (coords[:, 1] >= y_lo) & (coords[:, 1] < y_hi if iy < ny - 1 else coords[:, 1] <= y_hi)

        mask = in_x & in_y
        b = ix * ny + iy          
        patch_indices[b] = all_idx[mask]

# -------------------- load neighbors --------------------
graph_cpu = torch.load(
f"../data/sample_data/intermediate_data/weights_{neighbors_num}_lambda_{lambda_similarity}_E165_E1S3.pt",
map_location="cpu"
)
indptr  = graph_cpu["indptr"].to(torch.int64)     # (N+1,)
indices = graph_cpu["indices"].to(torch.int64)    # (E,)
weights = graph_cpu["weights"].to(torch.float32)  # (E,)
meta = graph_cpu["meta"]
N = int(meta["N"])

out_dir = f"../data/sample_data/intermediate_data/patch_graphs_k{neighbors_num}_lam{lambda_similarity}_E165_E1S3.pt"
os.makedirs(out_dir, exist_ok=True)

# 2) 逐 patch 抽诱导子图并保存
B = len(patch_indices)

for b in range(B):
idx = patch_indices[b].detach().cpu().to(torch.int64)  # (n_patch,)
n_patch = idx.numel()

if n_patch == 0:
    # 空 patch 直接跳过或存一个空文件
    torch.save(
        {
            "indptr": torch.zeros(1, dtype=torch.int64),
            "indices": torch.zeros(0, dtype=torch.int64),
            "weights": torch.zeros(0, dtype=torch.float32),
            "global_idx": idx,
            "meta": {"N": 0, "format": "CSR (patch induced)", "patch_id": int(b)},
        },
        os.path.join(out_dir, f"patch_{b:04d}.pt"),
    )
    continue

# old2new 映射：全局 -> patch 内局部编号
old2new = torch.full((N,), -1, dtype=torch.int64)  # (N,)
old2new[idx] = torch.arange(n_patch, dtype=torch.int64)

# 准备构建 CSR
indptr_sub = torch.zeros(n_patch + 1, dtype=torch.int64)
sub_indices_list = []
sub_weights_list = []

nnz = 0
for t in range(n_patch):
    u_global = int(idx[t].item())

    start = int(indptr[u_global].item())
    end   = int(indptr[u_global + 1].item())

    nbr_global = indices[start:end]          # (deg,)
    w = weights[start:end]                  # (deg,)

    # 只保留邻居也在 patch 内的边
    nbr_local = old2new[nbr_global]         # (deg,) in {-1,0..n_patch-1}
    mask = nbr_local >= 0

    keep_nbr_local = nbr_local[mask]        # (deg_in_patch,)
    keep_w = w[mask]                        # (deg_in_patch,)

    sub_indices_list.append(keep_nbr_local)
    sub_weights_list.append(keep_w)

    nnz += keep_nbr_local.numel()
    indptr_sub[t + 1] = nnz

indices_sub = torch.cat(sub_indices_list, dim=0).to(torch.int64)
weights_sub = torch.cat(sub_weights_list, dim=0).to(torch.float32)

# 保存 patch 图（关键：带 global_idx，后续切数据直接用它）
torch.save(
    {
        "indptr": indptr_sub,
        "indices": indices_sub,
        "weights": weights_sub,
        "global_idx": idx,  # 这个非常重要：告诉你 patch 对应全局哪些点
        "meta": {
            "N": int(n_patch),
            "format": "CSR (patch induced)",
            "patch_id": int(b),
            "neighbors_num": int(neighbors_num),
            "lambda_similarity": float(lambda_similarity),
        },
    },
    os.path.join(out_dir, f"patch_{b:04d}.pt"),
)

print("Done. Saved patch graphs to:", out_dir)