import scanpy as sc
import scipy.sparse as sp
import anndata as ad
import numpy as np
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import adjusted_rand_score
import time
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

seed = args.seed

# =========================================================
# Random seed (ONLY ADD)
# =========================================================

np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
# =========================================================

# print the starting time
time_start = time.time()

# input the preprocessed data
ROOT = Path(__file__).resolve().parents[3]
processed_dir = ROOT / "experiments" / "real_data" / "data" / "processed"
figures_dir = ROOT / "experiments" / "real_data" / "figures"
processed_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

in_path = processed_dir / "Mouse_embryo_E165_E1S3_prep.h5ad"
adata = sc.read_h5ad(in_path)
xy = adata.obsm["spatial"]
x = xy[:, 0]
y = xy[:, 1]
true_labels = adata.obs["annotation"].cat.codes.to_numpy()  # (N,)

# perform PCA
sc.tl.pca(adata, n_comps=50)
data_pca = adata.obsm["X_pca"]

# perform KMeans clustering
km = KMeans(n_clusters=23, random_state=seed).fit(data_pca)
initial_labels = km.predict(data_pca)

# calculate the ARI
ari = adjusted_rand_score(true_labels, initial_labels)
print(f"ARI: {ari}")

# plot the initial labels
c1 = plt.get_cmap("tab20").colors
c2 = plt.get_cmap("tab20b").colors
colors = list(c1) + list(c2)   # 40 colors
cmap23 = ListedColormap(colors[:23])

plt.figure()
plt.scatter(x, y, s=5, c=initial_labels, cmap=cmap23)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x")
plt.ylabel("y")
plt.title("KMeans labels")

plt.savefig(figures_dir / f"initial_kmeans_labels_seed_{seed}.png")

# save the tensor
pca_t = torch.from_numpy(data_pca.copy()).float()
x_t = torch.from_numpy(x.copy()).float().view(-1, 1)
y_t = torch.from_numpy(y.copy()).float().view(-1, 1)
true_t = torch.from_numpy(true_labels.copy()).long().view(-1, 1)
initial_t = torch.from_numpy(initial_labels.copy()).long().view(-1, 1)

# save the tensor
save_obj = {
    "pca_data": pca_t,                              # (N, 50)
    "spatial": torch.cat([x_t, y_t], dim=1),        # (N, 2)
    "true_labels": true_t.squeeze(1),               # (N,)
    "initial_labels": initial_t.squeeze(1),         # (N,)
    "seed": seed,                                   # (ONLY ADD, optional but useful)
}

out_path = processed_dir / f"tensor_pca_xy_true_initial_E165_E1S3_seed_{seed}.pt"
torch.save(save_obj, out_path)
print("Saved to:", out_path)

# print the ending time
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
