import torch
import argparse
from sklearn.metrics import adjusted_rand_score as sk_ari
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

seed = args.seed

ROOT = Path(__file__).resolve().parents[3]
processed_dir = ROOT / "experiments" / "real_data" / "data" / "processed"
data_path = processed_dir / f"tensor_pca_xy_true_initial_E165_E1S3_seed_{seed}.pt"
data = torch.load(data_path, map_location="cpu")
coords = data["spatial"]
data_pca = data["pca_data"]
true_labels = data["true_labels"]
initial_labels = data["initial_labels"]

ari = sk_ari(true_labels, initial_labels)
print(f"ARI: {ari}")
