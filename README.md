# NEXUS

> **N**eural network for **EX**ploring the **U**nified **S**tructure–functional connectome 
> Time-series dynamic graph learning for brain connectivity analysis.

---

## 📦 Structure

```
NEXUS/
├── SC-RNN.py                 # Main training script
├── analyse_notebook/         # Analysis & visualization notebooks
├── SC/                       # Structural connectivity matrices
│   ├── SEU_net_w_corticalLayers.xlsx
|   └── SEU_net_wo_corticalLayers.xlsx
├── requirements.txt
└── README.md
```

---

## ⚙️ Install

```bash
# Conda (recommended)
conda create -n nexus python=3.10
conda activate nexus
pip install -r requirements.txt
```

---

## 🗄️ Data

| File | Format | Description |
|------|--------|-------------|
| `--sc_file` | CSV | Structural connectivity matrix `[N, N]` |
| `--fc_file` | PKL/NPY | Functional connectivity time-series `[T, N, N]` or `[T, N]` |

> Place data in `SC/` or specify paths via arguments.

---

## 🚀 Usage

```bash
# Default run
python SC-RNN.py

# Custom experiment
python SC-RNN.py \
  --sc_file SC.csv \
  --fc_file traces.pkl \
  --save_dir results/exp1 \
  --seq_length 100 --pred_length 20 \
  --hidden_dim 64 --embed_dim 16 \
  --struct_constraint_weight 0.8 \
  --directed --seed 2024 --plot

# Visualize only (no training)
python SC-RNN.py --save_dir results/w_layer/stage3 --plot_only
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sc_file` | `structural_connectivity.csv` | SC matrix path |
| `--fc_file` | `traces.pkl` | FC data path |
| `--save_dir` | `./results/default` | Output directory |
| `--seq_length` / `--pred_length` | `50` / `10` | Input/prediction horizon |
| `--struct_constraint_weight` | `0.5` | Constraint strength (0.0 = off) |
| `--directed` | `False` | Use asymmetric adjacency |
| `--plot` / `--plot_only` | `False` | Generate/only generate plots |

*Run `python SC-RNN.py --help` for full options.*

---

## 📤 Outputs

```
results/exp/
├── model_best.pt           # Best model weights
├── train_losses.npy        # Training loss
├── val_losses.npy          # Validating loss
├── dynamic_full_graphs.npy # Predicted dynamic FC [T, N, N]
├── epoch_avg_graphs.npy    # Epoch-averaged graphs [E, N, N]
└── figures/                # Plots (if --plot enabled)
```

Use `analyse_notebook/*.ipynb` to load `.npy` files and generate publication-ready figures.

---

## 🔬 Reproducibility

- Fix random seed: `--seed <int>`
- Save config: auto-generated `config.yaml` in `--save_dir`
- Lock environment: use provided `requirements.txt`

