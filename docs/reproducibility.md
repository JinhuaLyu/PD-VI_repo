# Reproducibility Guide

## Scope

This repository separates core logic from experiment orchestration.
The goal is to rerun experiments without editing algorithm source files.

## Canonical Real-Data Pipeline

1. Data ingestion
- Copy raw dataset to `experiments/real_data/data/raw/`.

2. Data preprocessing
- Run `python src/pdvi/data/preprocess_anndata.py`
- Output: `experiments/real_data/data/processed/Mouse_embryo_E165_E1S3_prep.h5ad`

3. Tensor dataset creation
- Run `python src/pdvi/data/build_tensor_dataset.py --seed <seed>`
- Output: `experiments/real_data/data/processed/tensor_pca_xy_true_initial_E165_E1S3_seed_<seed>.pt`

4. Graph artifact creation
- `python src/pdvi/graph/compute_tangent.py --seed <seed>`
- `python src/pdvi/graph/compute_neighbors.py`
- `python src/pdvi/graph/compute_weight.py --seed <seed>`

5. Training
- Main model: `python src/pdvi/models/train_graph.py --config experiments/real_data/configs/train_graph.yaml --seed <seed>`
- Baselines in `src/pdvi/models/`

## Batch Utilities

- `bash experiments/real_data/scripts/run_tangents_1to5.sh`
- `bash experiments/real_data/scripts/run_seeds.sh [path/to/train_graph.yaml]`
- `bash experiments/real_data/scripts/sweep_eta.sh [path/to/sweep_eta.yaml] [path/to/train_graph.yaml]`

## Extending Experiments

When adding a new experiment:
- Add or update a YAML file in `experiments/real_data/configs/`.
- Keep reusable math in `src/pdvi/*`.
- Keep run orchestration in `experiments/real_data/scripts/`.
- Write outputs only under `experiments/real_data/{results,figures,logs}`.
- Prefer changing hyperparameters in YAML over editing model scripts.

## Notes

- `src/pdvi/legacy/patch_nbr_wip.py` is intentionally archived as WIP.
- Core algorithmic updates should be done via new files or clearly versioned changes.
