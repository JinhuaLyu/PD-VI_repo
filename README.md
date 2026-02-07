# PD-VI Repository

Research/systems codebase for patch-distributed variational inference (PD-VI) experiments on real spatial transcriptomics data.

## Repository Layout

- `src/pdvi/`: core code (data preparation, graph construction, model/training logic, evaluation helpers)
- `experiments/real_data/`: experiment-facing assets (configs, scripts, results, figures, logs, data folders)
- `docs/`: runbooks and reproducibility notes

### Core Code

- `src/pdvi/data/`
  - `preprocess_anndata.py`: preprocess raw `.h5ad` into a cleaned dataset
  - `build_tensor_dataset.py`: PCA + KMeans initialization + tensor export
- `src/pdvi/graph/`
  - `compute_tangent.py`: local smooth direction computation
  - `compute_neighbors.py`: geometry-aware CSR neighborhood graph
  - `compute_weight.py`: edge weights from geometry + feature similarity
- `src/pdvi/models/`
  - `train_graph.py`: graph + patch training pipeline
  - `train_without_graph.py`: non-graph baseline
  - `train_highdim_mfvi.py`: high-dimensional MFVI baseline
  - `train_patch_highdim.py`: patch MFVI variant
  - `train_patch_highdim_multi.py`: multi-patch MFVI variant
- `src/pdvi/eval/`
  - plotting and quick-check scripts
- `src/pdvi/legacy/`
  - archived WIP scripts kept for reference

### Experiment Assets

- `experiments/real_data/configs/`: YAML configs/templates
- `experiments/real_data/scripts/`: reproducible run scripts
- `experiments/real_data/results/`: scalar logs and result artifacts
- `experiments/real_data/figures/`: generated plots
- `experiments/real_data/logs/`: run logs
- `experiments/real_data/data/`
  - `raw/`: raw input data
  - `processed/`: preprocessed tensors/h5ad
  - `intermediate/`: tangents/graphs/weights

## Quick Start

1. Put raw data in `experiments/real_data/data/raw/`.
2. Prepare data:
   - `python src/pdvi/data/preprocess_anndata.py`
   - `python src/pdvi/data/build_tensor_dataset.py --seed 1`
3. Build graph artifacts:
   - `python src/pdvi/graph/compute_tangent.py --seed 1`
   - `python src/pdvi/graph/compute_neighbors.py`
   - `python src/pdvi/graph/compute_weight.py --seed 1`
4. Train:
   - `python src/pdvi/models/train_graph.py --config experiments/real_data/configs/train_graph.yaml --seed 1`
5. End-to-end shortcut:
   - `bash experiments/real_data/scripts/run_pipeline.sh 1`

## Config-Driven Runs

- Main training script now loads YAML config:
  - `experiments/real_data/configs/train_graph.yaml`
- Example:
  - `python src/pdvi/models/train_graph.py --config experiments/real_data/configs/train_graph.yaml`
- `--seed` overrides the seed in YAML for that run.

See `docs/reproducibility.md` for full experiment protocol.
