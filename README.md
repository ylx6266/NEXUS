# Learning Single-Cell Connectomes by Bridging Anatomy and Dynamics

>  **NEXUS** (**N**etwork **E**stimation e**X**ploiting a **U**nified **S**tructure)

> a structural constraint framework designed to reconstruct high-resolution functional connectomes from neural activity recorded at the single-cell resolution within a shared and targeted anatomical scaffold also derived from single-resolution neuronal connectivity.

---






## 📦 Structure

```
NEXUS/
├── SC-RNN.py                 # Main training script
├── analyse_notebook/         # Analysis & visualization notebooks
├── example/                       # examples
│   ├── SC.csv
|   └── trace.pkl
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
| `--fc_file` | PKL/NPY | Functional connectivity time-series `[N, T]`  |

> Place data in `SC/` or specify paths via arguments.

---

## 🚀 Usage

```bash
python SC-RNN.py --sc_file example/SC.csv --fc_file trace.pkl 

# Custom experiment
python SC-RNN.py \
  --sc_file SC.csv \
  --fc_file traces.pkl \
  --save_dir results/exp1 \
  --seq_length 100 --pred_length 20 \
  --hidden_dim 64 --embed_dim 16 \
  --struct_constraint_weight 0.8 \
  --directed --seed 42 --plot

# Visualize only (no training)
python SC-RNN.py --save_dir results/default --plot_only
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sc_file` | `SC.csv` | SC matrix path |
| `--fc_file` | `traces.pkl` | FC data path |
| `--save_dir` | `./results/default` | Output directory |
| `--seq_length` / `--pred_length` | `50` / `10` | Input/prediction horizon |
| `--struct_constraint_weight` | `0.7` | Constraint strength (0.0 = off) |
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

## 🔬 Tips

> For reconstructing functional connectomes from a specific dataset, we strongly recommend performing a systematic hyperparameter search, as demonstrated in our manuscript. The configuration achieving the lowest validation loss among the tested combinations can be selected for subsequent FC reconstruction. Users are also encouraged to examine convergence behavior and assess the stability of reconstructed networks across different random seeds.

