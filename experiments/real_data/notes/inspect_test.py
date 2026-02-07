import scanpy as sc
from pathlib import Path

# input the preprocessed data
ROOT = Path(__file__).resolve().parents[3]
in_path = ROOT / "experiments" / "real_data" / "data" / "processed" / "Mouse_embryo_E165_E1S3_prep.h5ad"
adata = sc.read_h5ad(in_path)
xy = adata.obsm["spatial"]
x = xy[:, 0] # x coordinates
y = xy[:, 1] # y coordinates
true_labels = adata.obs["annotation"].cat.codes.to_numpy()  # (N,)
raw_data = adata.X # (N, 5000)
