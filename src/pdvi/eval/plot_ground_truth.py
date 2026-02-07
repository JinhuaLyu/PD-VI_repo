import scanpy as sc
import scipy.sparse as sp
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
processed_dir = ROOT / "experiments" / "real_data" / "data" / "processed"
figures_dir = ROOT / "experiments" / "real_data" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

in_path = processed_dir / "Mouse_embryo_E165_E1S3_prep.h5ad"
adata = sc.read_h5ad(in_path)

xy = adata.obsm["spatial"]
x = xy[:, 0]
y = xy[:, 1]

true_labels = adata.obs["annotation"].cat.codes.to_numpy()

c1 = plt.get_cmap("tab20").colors
c2 = plt.get_cmap("tab20b").colors
colors = list(c1) + list(c2)   # 40 colors
cmap23 = ListedColormap(colors[:23])

# # plot
# plt.figure()
# plt.scatter(x, y, s=5, c=true_labels, cmap=cmap23)
# plt.gca().set_aspect("equal", adjustable="box")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("True labels")
# plt.savefig(figures_dir / "ground_truth_all.png")

nx, ny = 5, 5
x_edges = np.linspace(x.min(), x.max(), nx + 1)
y_edges = np.linspace(y.min(), y.max(), ny + 1)


ix, iy = 2, 2

x_lo, x_hi = x_edges[ix], x_edges[ix + 1]
y_lo, y_hi = y_edges[iy], y_edges[iy + 1]

# 注意边界：为了不重复，把右/上边界设为开区间，最后一格包含最大值
in_x = (x >= x_lo) & (x < x_hi if ix < nx - 1 else x <= x_hi)
in_y = (y >= y_lo) & (y < y_hi if iy < ny - 1 else y <= y_hi)
mask = in_x & in_y

idx = np.where(mask)[0]
print(f"Tile (ix={ix}, iy={iy}) has {idx.size} points.")




num_labels = np.unique(true_labels[mask]).size
print(f"Number of labels: {num_labels}")

plt.figure()
plt.scatter(x[mask], y[mask], s=5, c=true_labels[mask], cmap=cmap23)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Points in tile (ix={ix}, iy={iy})")
plt.savefig(figures_dir / f"ground_truth_{ix}_{iy}.png")
