import scanpy as sc
import scipy.sparse as sp
import anndata as ad
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
raw_dir = ROOT / "experiments" / "real_data" / "data" / "raw"
processed_dir = ROOT / "experiments" / "real_data" / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

in_path = raw_dir / "Mouse_embryo_E165_E1S3.h5ad"
out_path = processed_dir / "Mouse_embryo_E165_E1S3_prep.h5ad"

adata = sc.read_h5ad(in_path)

# 1) clean the data
adata_clean = ad.AnnData(
    X=adata.layers["count"].copy(),
    obs=adata.obs[["annotation"]].copy(),
)
adata_clean.obsm["spatial"] = adata.obsm["spatial"].copy()

#2) preprocess the data
sc.pp.highly_variable_genes(adata_clean, flavor="seurat_v3", n_top_genes=5000)
adata_clean = adata_clean[:, adata_clean.var["highly_variable"]].copy()
sc.pp.normalize_total(adata_clean, target_sum=1e4)
sc.pp.log1p(adata_clean)
sc.pp.scale(adata_clean, zero_center=False, max_value=15)
# print the processed data
print(adata_clean)

#3) save the data
adata_clean.write_h5ad(out_path)
print("Saved to:", out_path, "shape:", adata_clean.shape)

# # data distribution:
# # percentiles (50, 90, 95, 99, 99.5, 99.9, 99.99):
# # [  2.61281991   5.13458958   6.76237874  14.93098844  23.60913941 53.97897444 139.41092212]
