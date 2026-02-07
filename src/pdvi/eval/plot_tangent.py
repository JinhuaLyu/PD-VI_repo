import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
intermediate_dir = ROOT / "experiments" / "real_data" / "data" / "intermediate"
processed_dir = ROOT / "experiments" / "real_data" / "data" / "processed"
figures_dir = ROOT / "experiments" / "real_data" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

tangents_path = intermediate_dir / "tangents_E165_E1S3.pt"
tangents = torch.load(tangents_path, map_location="cpu")  # (N,2)

in_path = processed_dir / "tensor_pca_xy_true_initial_E165_E1S3.pt"
data = torch.load(in_path, map_location="cpu")
coords = data["spatial"]          # (N,2)
true_labels = data["true_labels"] # (N,)
initial_labels = data["initial_labels"] # (N,)
# use x,y
x = coords[:, 0].numpy()
y = coords[:, 1].numpy()

U = tangents[:, 0].numpy()
V = tangents[:, 1].numpy()

# colormap for 23 classes
c1 = plt.get_cmap("tab20").colors
c2 = plt.get_cmap("tab20b").colors
colors = list(c1) + list(c2)
cmap23 = ListedColormap(colors[:23])

plt.figure(figsize=(7, 7))
plt.scatter(x, y, s=5, c=true_labels.numpy(), cmap=cmap23)

# 叠加 tangent arrows
plt.quiver(
    x, y, U, V,
    angles="xy",
    scale_units="xy",
    scale=0.8,      # 数值越大箭头越短；你可以调 0.2/0.8
    width=0.001,
    color="k",
)

plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x")
plt.ylabel("y")
plt.title("True labels + tangents")
plt.tight_layout()
plt.savefig(figures_dir / "tangents_all.png", dpi=300)


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

plt.figure(figsize=(7, 7))
plt.scatter(x[mask], y[mask], s=5, c=initial_labels[mask], cmap=cmap23)

plt.quiver(
    x[mask], y[mask], U[mask], V[mask],
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.002,
    color="k",
)

plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Tile (ix={ix}, iy={iy}) + tangents")
plt.tight_layout()
plt.savefig(figures_dir / f"tangents_tile_{ix}_{iy}_initial.png", dpi=300)
