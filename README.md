# Learning Single-Cell Connectomes by Bridging Anatomy and Dynamics

>  **NEXUS** (**N**etwork **E**stimation e**X**ploiting a **U**nified **S**tructure)

> a structural constraint framework designed to reconstruct high-resolution functional connectomes from neural activity recorded at the single-cell resolution within a shared and targeted anatomical scaffold also derived from single-resolution neuronal connectivity.

---

## Data Availability

The datasets and reconstructed functional connectomes used in this study are available from the following sources:

- **Neuropixels recordings:** The International Brain Laboratory (IBL) Brain-Wide Map dataset is publicly available through the [IBL data portal](https://www.internationalbrainlab.com).

- **MERFISH data:** The MERFISH dataset is publicly available through the Allen Institute's [ABC Atlas data portal](https://alleninstitute.github.io/abc_atlas_access/descriptions/MERFISH-C57BL6J-638850.html).

- **Processed datasets and NEXUS outputs:** The reconstructed functional connectivity matrices and processed Neuropixels datasets, including pre-aggregation spike data and time-normalized regional activity traces, are temporarily available through [Google Drive](https://drive.google.com/drive/folders/18jGqBYK-bTRpADzvFMqXyyHomkGdnBk5?usp=sharing) for peer review.

- **Calcium-imaging data:** The calcium-imaging data and corresponding model outputs are available from the corresponding authors upon reasonable request.

Permanent repository links for the publicly shared processed datasets and model outputs will be provided upon publication.



## Structure

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

## Install

```bash
# Conda (recommended)
conda create -n nexus python=3.10
conda activate nexus
pip install -r requirements.txt
```

---

## Data

| File | Format | Description |
|------|--------|-------------|
| `--sc_file` | CSV | Structural connectivity matrix `[N, N]` |
| `--fc_file` | PKL/NPY | Functional connectivity time-series `[N, T]`  |

> Place data in `SC/` or specify paths via arguments.

---

## Usage

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
  --directed --plot

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

## Outputs

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

## Recommendations for Model Training and Selection

NEXUS relies on stochastic optimization, and the reconstructed functional networks may vary with random initialization and computational environment. Consequently, identical reconstruction results are not always guaranteed across different random seeds, hardware platforms, or software configurations.

For reliable application to new datasets, we strongly recommend evaluating multiple hyperparameter configurations rather than relying on a single training run. Following the procedure described in our manuscript, candidate configurations should be assessed using two complementary criteria:

1. **Optimization convergence:** Evaluate the convergence of validation loss during training.
2. **Network consistency:** Evaluate the similarity of the inferred functional connectome to those obtained from other hyperparameter configurations using signed weighted Jaccard similarity.

We recommend selecting a representative configuration based on the combined rankings of these two criteria, independently of the downstream biological findings.

Users should also record the random seed, software environment, and selected model configuration to facilitate reproducibility.

