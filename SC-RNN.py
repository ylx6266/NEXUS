#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time Series Dynamic Graph Learning Model - Command Line Version
Supports custom file paths and hyperparameters
"""

import argparse
import inspect
import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


# ==================== Utility Functions ====================
def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ==================== Model Definitions ====================
class SimpleGraphLearn(nn.Module):
    """Graph learning module"""
    def __init__(self, node_dim, nodes, dropout=0.5, struct_connectome=None, is_undirected=False):
        super(SimpleGraphLearn, self).__init__()
        self.nodes = nodes
        self.dropout = dropout
        self.is_undirected = is_undirected
        
        self.gate_net = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid()
        )
        
        self.attention = nn.Linear(node_dim, 1, bias=False)
        self.norm = nn.LayerNorm(nodes)
        
        self.w = nn.Parameter(torch.zeros(size=(nodes, node_dim)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        
        if struct_connectome is not None:
            self.struct_connectome = nn.Parameter(
                torch.FloatTensor(struct_connectome), 
                requires_grad=False
            )
        else:
            self.struct_connectome = None
            
        self.graph_weight = nn.Parameter(torch.ones(1))
        self.struct_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, node_features, temporal_features=None, batch_size=1):
        batch_size, nodes, node_dim = node_features.shape
        
        if temporal_features is not None:
            concat_features = torch.cat([node_features, temporal_features], dim=-1)
            gate = self.gate_net(concat_features)
            node_features = gate * node_features + (1 - gate) * temporal_features
        
        node_res = torch.einsum('bnd, nl -> bnl', node_features, self.w) + node_features
        skip_atten = torch.einsum('bnd,bdm->bnm', node_res, node_res.transpose(-1, -2))
        skip_atten = self.norm(skip_atten)
        
        attention_scores = torch.matmul(node_features, node_features.transpose(-2, -1))
        attention_scores = attention_scores / (node_dim ** 0.5)
        attention_scores = F.dropout(attention_scores, self.dropout, training=self.training)
        
        adj_raw = attention_scores + skip_atten
        
        if self.struct_connectome is not None:
            struct_adj = self.struct_connectome.unsqueeze(0).expand(batch_size, -1, -1)
            adj_dynamic = self.graph_weight * adj_raw + self.struct_weight * struct_adj
        else:
            adj_dynamic = adj_raw
            
        adj_dynamic = torch.tanh(adj_dynamic)
        
        # Symmetrize only if the structural connectome is undirected
        if self.is_undirected:
            adj_dynamic = 0.5 * (adj_dynamic + adj_dynamic.transpose(-1, -2))
        
        if self.struct_connectome is not None:
            struct_upper_bound = self.struct_connectome.unsqueeze(0).expand(batch_size, -1, -1)
            adj_dynamic = torch.clamp(adj_dynamic, -struct_upper_bound, struct_upper_bound)
        
        return adj_dynamic


class TimeSeriesDynamicGraphModel(nn.Module):
    """Time series dynamic graph learning model"""
    def __init__(self, device, num_nodes, seq_length=12, in_dim=1, out_dim=1, 
                 hidden_dim=32, dropout=0.3, embed_dim=10, heads=4, head_dim=8, 
                 struct_connectome=None, is_undirected=False):
        super(TimeSeriesDynamicGraphModel, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.node_embedding = nn.Embedding(num_nodes, embed_dim)
        
        self.temporal_fusion = nn.Linear(hidden_dim, embed_dim)
        
        self.graph_learn = SimpleGraphLearn(
            node_dim=embed_dim, 
            nodes=num_nodes,
            dropout=dropout,
            struct_connectome=struct_connectome,
            is_undirected=is_undirected
        )

        self.temporal_encoder = nn.LSTM(
            input_size=in_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout
        )

        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, input, pred_time_embed=None):
        batch_size, seq_length, num_nodes, in_dim = input.shape
        
        idx = torch.arange(num_nodes).to(self.device)
        node_embeddings = self.node_embedding(idx)
        node_embeddings = node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        x_reshaped = input.permute(0, 2, 1, 3).contiguous().view(batch_size * num_nodes, seq_length, in_dim)
        temporal_out, _ = self.temporal_encoder(x_reshaped)
        
        temporal_features = temporal_out.view(batch_size, num_nodes, seq_length, -1)
        
        dynamic_graphs = []
        temporal_hiddens = []
        
        for t in range(seq_length):
            temporal_features_t = temporal_features[:, :, t, :]
            
            temporal_features_proj_t = self.temporal_fusion(temporal_features_t)
            
            adj_dynamic_t = self.graph_learn(
                node_embeddings, 
                temporal_features_proj_t, 
                batch_size
            )
            
            dynamic_graphs.append(adj_dynamic_t)
            temporal_hiddens.append(temporal_features_t)
        
        adj_dynamic_all = torch.stack(dynamic_graphs, dim=1)
        
        gcn_outputs_all = []
        
        for t in range(seq_length):
            h_t = temporal_hiddens[t]
            adj_t = dynamic_graphs[t]
            
            current_h = h_t
            for gcn_layer in self.gcn_layers:
                h_gcn = torch.matmul(adj_t.transpose(-1,-2), current_h)
                h_gcn = gcn_layer(h_gcn)
                h_gcn = F.relu(h_gcn)
                h_gcn = F.dropout(h_gcn, self.dropout, training=self.training)
                current_h = h_gcn
            
            gcn_outputs_all.append(current_h)
        
        gcn_outputs_all = torch.stack(gcn_outputs_all, dim=1)
        
        final_hidden = gcn_outputs_all[:, -1, :, :]
        output = self.output_layer(final_hidden)
        
        return output, None, adj_dynamic_all


class Trainer:
    """Trainer class"""
    def __init__(self, num_nodes, seq_length, in_dim, out_dim, hidden_dim, dropout, lrate, wdecay, device, 
                 embed_dim, heads=4, head_dim=8, struct_connectome=None, struct_constraint_weight=0.2,
                 is_undirected=False):
        
        self.model = TimeSeriesDynamicGraphModel(
            device, num_nodes, seq_length, in_dim, out_dim,
            hidden_dim=hidden_dim, dropout=dropout, embed_dim=embed_dim,
            heads=heads, head_dim=head_dim, struct_connectome=struct_connectome,
            is_undirected=is_undirected
        )
        
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # Compatible with different PyTorch versions, some versions don't support verbose parameter
        scheduler_kwargs = {
            'mode': 'min',
            'factor': 0.5,
            'patience': 10,
            'eps': 0.00001,
            'cooldown': 20
        }
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_kwargs)
        
        self.loss = nn.MSELoss()
        self.clip = 0.5
        self.struct_constraint_weight = struct_constraint_weight

    def train(self, input, real_val, iter=0):
        self.model.train()
        output, gl_loss, adj_d = self.model(input)
        
        loss = self.loss(output, real_val)
        
        if hasattr(self.model.graph_learn, 'struct_connectome') and \
           self.model.graph_learn.struct_connectome is not None:
            
            if len(adj_d.shape) == 4:
                avg_adj = torch.mean(adj_d, dim=(0, 1))
            elif len(adj_d.shape) == 3:
                avg_adj = torch.mean(adj_d, dim=0)
            else:
                avg_adj = adj_d
            
            struct_adj = self.model.graph_learn.struct_connectome
            
            avg_adj_abs = torch.abs(avg_adj)
            struct_loss = F.mse_loss(avg_adj_abs, struct_adj)
            
            loss = loss + self.struct_constraint_weight * struct_loss
        
        loss.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()

    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output, gl_loss, adj_d = self.model(input)
            loss = self.loss(output, real_val)
        return loss.item(), output, adj_d


# ==================== Training Functions ====================
def train_timeseries_dynamic_graph(
    cell_trace, struct_connectome, seq_length, pred_length,
    hidden_dim, lr, epochs, batch_size, dropout,
    device=None, seed=42, val_ratio=0.2, patience=20,
    heads=4, head_dim=8, embed_dim=10, struct_constraint_weight=0.5,
    save_dir="./results/wo_layer/stage1", is_undirected=False
):
    """Train time series dynamic graph model"""
    
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_regions, total_time = cell_trace.shape
    if seq_length is None:
        seq_length = total_time - pred_length

    if not np.isfinite(cell_trace).all():
        raise ValueError("Activity data contains NaN or infinite values")
    if struct_connectome is not None and not np.isfinite(struct_connectome).all():
        raise ValueError("Structural connectivity contains NaN or infinite values")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    # Data preprocessing
    data, targets = [], []
    cell_trace = (cell_trace - np.mean(cell_trace)) / (np.std(cell_trace) + 1e-8)

    for t in range(total_time - seq_length - pred_length + 1):
        x = cell_trace[:, t:t+seq_length].T.reshape(seq_length, num_regions, 1)
        y = cell_trace[:, t+seq_length:t+seq_length+pred_length].T.reshape(num_regions, pred_length)
        data.append(x)
        targets.append(y)

    data, targets = np.array(data), np.array(targets)
    split = int(len(data) * (1 - val_ratio))

    if split < 1 or split >= len(data):
        raise ValueError("Time series and validation ratio must provide nonempty training and validation sets")

    train_data = torch.FloatTensor(data[:split]).to(device)
    train_targets = torch.FloatTensor(targets[:split]).to(device)
    val_data = torch.FloatTensor(data[split:]).to(device)
    val_targets = torch.FloatTensor(targets[split:]).to(device)

    # Model initialization
    engine = Trainer(
        num_nodes=num_regions, seq_length=seq_length, in_dim=1, out_dim=pred_length,
        hidden_dim=hidden_dim, dropout=dropout, lrate=lr, wdecay=0.0, device=device,
        embed_dim=embed_dim, heads=heads, head_dim=head_dim,
        struct_connectome=struct_connectome, struct_constraint_weight=struct_constraint_weight,
        is_undirected=is_undirected
    )

    saved_this_run = False
    best_val = float('inf')
    patience_count = 0
    train_losses, val_losses = [], []
    epoch_avg_graphs = []

    print(f"Training started | seq={seq_length}, T={total_time}")

    # Training loop
    for epoch in range(epochs):
        engine.model.train()
        train_loss = 0
        for i in range(0, len(train_data), batch_size):
            train_loss += engine.train(train_data[i:i+batch_size], train_targets[i:i+batch_size])
        train_loss /= (len(train_data)//batch_size + 1)
        train_losses.append(train_loss)

        # Validation
        engine.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                val_loss += engine.eval(val_data[i:i+batch_size], val_targets[i:i+batch_size])[0]
        val_loss /= (len(val_data)//batch_size + 1)
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}: train={train_loss}, val={val_loss}")
        val_losses.append(val_loss)
        engine.scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save(engine.model.state_dict(), f"{save_dir}/best_model.pth")
            saved_this_run = True
        else:
            patience_count += 1
        if patience_count >= patience:
            print(f"Early stopping @ epoch {epoch+1}")
            break

        # Collect average graph for each epoch
        with torch.no_grad():
            x = train_data[0:1]
            _, _, adj = engine.model(x)
            epoch_avg_graphs.append(np.mean(adj.squeeze(0).cpu().numpy(), axis=0))

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train {train_loss:.5f} | Val {val_loss:.5f}")

    print("Training completed, loading best model")
    if not saved_this_run:
        raise RuntimeError("No best model was saved during this training run")
    engine.model.load_state_dict(torch.load(f"{save_dir}/best_model.pth", map_location=device, weights_only=True))
    engine.model.eval()

    np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{save_dir}/val_losses.npy", np.array(val_losses))
    np.save(f"{save_dir}/epoch_avg_graphs.npy", np.array(epoch_avg_graphs))
    print(f"Saved epoch_avg_graphs.npy to {save_dir}")

    # Full time inference
    print("Reconstructing full dynamic graphs...")

    full_dynamic_graphs = np.zeros((total_time, num_regions, num_regions))
    frame_count = np.zeros(total_time)

    with torch.no_grad():
        for t in tqdm(range(total_time - seq_length + 1)):
            x = cell_trace[:, t:t+seq_length].T.reshape(1, seq_length, num_regions, 1)
            x = torch.FloatTensor(x).to(device)

            _, _, adj_seq = engine.model(x)
            adj_seq = adj_seq.squeeze(0).cpu().numpy()

            for k in range(seq_length):
                idx = t + k
                full_dynamic_graphs[idx] += adj_seq[k]
                frame_count[idx] += 1

    for i in range(total_time):
        if frame_count[i] > 0:
            full_dynamic_graphs[i] /= frame_count[i]

    np.save(f"{save_dir}/dynamic_full_graphs.npy", full_dynamic_graphs)
    print(f"Saved dynamic_full_graphs.npy to {save_dir}, shape={full_dynamic_graphs.shape}")

    return engine.model, train_losses, val_losses, full_dynamic_graphs


# ==================== Visualization Functions ====================
def load_plot_sc(save_dir, sc_file, num_regions):
    """Use the aligned, normalized training SC and its matching labels."""
    aligned_path = os.path.join(save_dir, "aligned_sc.csv")
    if os.path.exists(aligned_path):
        sc = pd.read_csv(aligned_path, index_col=0)
        values = sc.to_numpy()
    elif sc_file and os.path.exists(sc_file):
        sc = pd.read_csv(sc_file, index_col=0)
        values = MinMaxScaler().fit_transform(sc)
    else:
        return None, None
    if values.shape != (num_regions, num_regions):
        raise ValueError("SC and learned graph dimensions differ; regenerate plots with matching --fc_file and slice arguments")
    return values, sc.index.tolist()


def plot_dynamic_graph_results(save_dir, struct_connectome_array=None, struct_connectome_file=None):
    """Plot dynamic graph analysis results"""
    
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Save directory does not exist: {save_dir}")

    # Load files
    dyn_path = os.path.join(save_dir, "dynamic_full_graphs.npy")
    epoch_graphs_path = os.path.join(save_dir, "epoch_avg_graphs.npy")

    dyn = np.load(dyn_path)
    T, N, _ = dyn.shape

    struct, labels = load_plot_sc(save_dir, struct_connectome_file, N)
    if struct_connectome_array is not None:
        struct = struct_connectome_array

    epoch_graphs = np.load(epoch_graphs_path) if os.path.exists(epoch_graphs_path) else None

    # Calculate metrics
    degree = np.sum(np.abs(dyn), axis=2)
    mean_degree = np.mean(degree, axis=1)
    avg_strength = np.mean(np.abs(dyn), axis=(1,2))
    pos_ratio = np.sum(dyn > 0, axis=(1,2)) / (N*N)
    neg_ratio = np.sum(dyn < 0, axis=(1,2)) / (N*N)
    avg_graph = np.mean(dyn, axis=0)
    eps = 1e-4
    nonzero_ratio = np.sum(np.abs(dyn) > eps, axis=(1,2)) / (N*N)
    sparsity = 1 - nonzero_ratio

    # Plotting
    fig, axs = plt.subplots(2,3, figsize=(22,12))

    axs[0,0].plot(mean_degree, linewidth=1)
    axs[0,0].set_title("Degree Centrality Over Time")
    axs[0,0].set_xlabel("Time")
    axs[0,0].set_ylabel("Avg Degree")

    axs[0,1].plot(avg_strength, linewidth=1)
    axs[0,1].set_title("Non-zero Avg Connection Strength")
    axs[0,1].set_xlabel("Time")
    axs[0,1].set_ylabel("|Edge| Mean")

    axs[0,2].plot(pos_ratio, label="Positive", color="green")
    axs[0,2].plot(neg_ratio, label="Negative", color="red")
    axs[0,2].set_title("Positive / Negative Edge Ratio")
    axs[0,2].set_xlabel("Time")
    axs[0,2].legend()

    im1 = axs[1,0].imshow(avg_graph, cmap="seismic", vmin=-1, vmax=1)
    axs[1,0].set_title("Average Learned Dynamic Graph")
    plt.colorbar(im1, ax=axs[1,0], shrink=0.6)
    
    # Add labels to average graph if available
    if labels is not None:
        axs[1,0].set_xticks(np.arange(len(labels)))
        axs[1,0].set_yticks(np.arange(len(labels)))
        axs[1,0].set_xticklabels(labels, rotation=90, fontsize=4)
        axs[1,0].set_yticklabels(labels, fontsize=4)

    if struct is not None:
        im2 = axs[1,1].imshow(struct, cmap="Reds", vmin=0, vmax=1)
        axs[1,1].set_title("Structural Connectome")
        plt.colorbar(im2, ax=axs[1,1], shrink=0.6)
        
        # Add labels to structural connectome if available
        if labels is not None:
            axs[1,1].set_xticks(np.arange(len(labels)))
            axs[1,1].set_yticks(np.arange(len(labels)))
            axs[1,1].set_xticklabels(labels, rotation=90, fontsize=4)
            axs[1,1].set_yticklabels(labels, fontsize=4)
    else:
        axs[1,1].text(0.5,0.5,"No Structural Matrix",ha='center',va='center')
        axs[1,1].set_title("Structural Connectome (missing)")

    axs[1,2].plot(sparsity, linewidth=1)
    axs[1,2].set_title("Graph Sparsity Over Time")
    axs[1,2].set_xlabel("Time")
    axs[1,2].set_ylabel("Sparsity (1 - nonzero density)")
    axs[1,2].set_ylim(0, 1)

    plt.tight_layout()

    fig_path = os.path.join(save_dir, "dynamic_graph_analysis.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {fig_path}")
    return fig_path


def plot_dynamic_graph_single(save_dir, struct_connectome_file=None):
    """Plot single dynamic graph comparison"""
    
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Save directory does not exist: {save_dir}")

    dyn_path = os.path.join(save_dir, "dynamic_full_graphs.npy")
    dyn = np.load(dyn_path)
    T, N, _ = dyn.shape

    struct, labels = load_plot_sc(save_dir, struct_connectome_file, N)

    avg_graph = np.mean(dyn, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(14,7))

    im1 = axs[0].imshow(avg_graph, cmap="seismic", vmin=-1, vmax=1)
    axs[0].set_title("Structure-constrained functional connectome", fontsize=16)
    plt.colorbar(im1, ax=axs[0], shrink=0.6)

    if labels is not None:
        axs[0].set_xticks(np.arange(len(labels)))
        axs[0].set_yticks(np.arange(len(labels)))
        axs[0].set_xticklabels(labels, rotation=90, fontsize=4)
        axs[0].set_yticklabels(labels, fontsize=4)

    if struct is not None:
        cmap = plt.cm.get_cmap('seismic')
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            'white_red',
            cmap(np.linspace(0.5, 1, 256))
        )
        im2 = axs[1].imshow(struct, cmap=new_cmap, vmin=0, vmax=1)
        axs[1].set_title("Structural Connectome", fontsize=16)
        plt.colorbar(im2, ax=axs[1], shrink=0.6)

        if labels is not None:
            axs[1].set_xticks(np.arange(len(labels)))
            axs[1].set_yticks(np.arange(len(labels)))
            axs[1].set_xticklabels(labels, rotation=90, fontsize=4)
            axs[1].set_yticklabels(labels, fontsize=4)
    else:
        axs[1].text(0.5, 0.5, "No Structural Matrix", ha='center', va='center')
        axs[1].set_title("Structural Connectome (missing)")

    plt.tight_layout()

    fig_path = os.path.join(save_dir, "dynamic_graph_analysis_single.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {fig_path}")
    return fig_path


def plot_training_epoch_metrics(save_dir="./results"):
    """Plot training epoch metrics including loss curves and graph metrics"""
    
    # Check & load
    train_path = os.path.join(save_dir, "train_losses.npy")
    val_path = os.path.join(save_dir, "val_losses.npy")
    epoch_graphs_path = os.path.join(save_dir, "epoch_avg_graphs.npy")

    if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(epoch_graphs_path):
        raise FileNotFoundError(f"Required files missing in {save_dir}:\n"
                                f"{train_path}\n{val_path}\n{epoch_graphs_path}")

    train_losses = np.load(train_path)
    val_losses = np.load(val_path)
    epoch_graphs = np.load(epoch_graphs_path)  # shape (E, N, N)

    E, N, _ = epoch_graphs.shape

    # Epoch metrics
    # Mean |A|
    epoch_mean_conn = np.mean(np.abs(epoch_graphs), axis=(1,2))

    # Degree = % edges > eps
    eps = 1e-4
    degree_epoch = np.mean((np.abs(epoch_graphs) > eps), axis=(1,2))

    # Pos / neg ratio
    pos_epoch = np.sum(epoch_graphs > 0, axis=(1,2)) / (N*N)
    neg_epoch = np.sum(epoch_graphs < 0, axis=(1,2)) / (N*N)

    # Plot
    plt.figure(figsize=(16, 10))

    # 1. Loss curves
    plt.subplot(2,2,1)
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. Mean |A|
    plt.subplot(2,2,2)
    plt.plot(epoch_mean_conn, color="purple", linewidth=2)
    plt.title("Mean |Connection| per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean |A|")
    plt.grid(alpha=0.3)

    # 3. Degree centrality
    plt.subplot(2,2,3)
    plt.plot(degree_epoch, color="green", linewidth=2)
    plt.title("Mean Degree Centrality per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Degree")
    plt.grid(alpha=0.3)

    # 4. Pos vs Neg
    plt.subplot(2,2,4)
    plt.plot(pos_epoch, label="Positive", color="red", linewidth=2)
    plt.plot(neg_epoch, label="Negative", color="blue", linewidth=2)
    plt.title("Positive vs Negative Connections per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Save
    fig_path = os.path.join(save_dir, "training_epoch_metric_plot.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {fig_path}")

    return fig_path


# ==================== Data Loading Functions ====================
def parse_slice(slice_str):
    """Parse slice string, supports 'start:end' or 'start:end:step' format"""
    if slice_str is None or slice_str == '':
        return None
    
    parts = slice_str.split(':')
    if len(parts) == 1:
        # Single number, represents end position
        return slice(None, int(parts[0]))
    elif len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
        return slice(start, end)
    elif len(parts) == 3:
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
        step = int(parts[2]) if parts[2] else None
        return slice(start, end, step)
    else:
        raise ValueError(f"Invalid slice format: {slice_str}, please use 'start:end' or 'start:end:step' format")


def load_data(sc_file, fc_file, fc_time_slice=None, fc_region_slice=None, is_undirected=False, save_dir=None):
    """
    Load structural connectivity matrix and functional connectivity data
    
    Parameters:
        sc_file: Path to structural connectivity matrix file
        fc_file: Path to functional connectivity data file
        fc_time_slice: Time dimension slice for functional connectivity data, format: 'start:end' or 'start:end:step'
        fc_region_slice: Region dimension slice for functional connectivity data, format: 'start:end' or 'start:end:step'
        is_undirected: Boolean indicating if graph is undirected (symmetric). Default is False.
    
    Returns:
        SC_rescaled_array: Structural connectivity matrix (numpy array)
        FC_array: Functional connectivity data (numpy array)
    """
    # Load structural connectivity matrix
    if sc_file and os.path.exists(sc_file):
        SC = pd.read_csv(sc_file, index_col=0)
        if not np.isfinite(SC.to_numpy(dtype=float)).all():
            raise ValueError("SC contains NaN or infinite values")
        # Align and slice raw SC before normalization.
        SC_rescaled_array = SC.to_numpy(dtype=float)
        sc_labels = SC.index.tolist()
        
        graph_type = "undirected" if is_undirected else "directed"
        print(f"Loaded structural connectivity matrix: {sc_file}, shape={SC_rescaled_array.shape}, type={graph_type}")
    else:
        SC_rescaled_array = None
        graph_type = "undirected" if is_undirected else "directed"
        print("Warning: Structural connectivity matrix file not provided, structural constraints will not be used")
        print(f"Using {graph_type} graph (default: directed)")
    
    # Load functional connectivity data
    if fc_file and os.path.exists(fc_file):
        if fc_file.endswith('.pkl'):
            with open(fc_file, 'rb') as f:
                FC_array = pickle.load(f)
        elif fc_file.endswith('.npy'):
            FC_array = np.load(fc_file)
        else:
            raise ValueError(f"Unsupported file format: {fc_file}, please use .pkl or .npy format")
        
        # Regional trace dictionaries must follow the same region order as SC.
        if isinstance(FC_array, dict):
            traces = FC_array.get('binned_traces')
            if not isinstance(traces, dict) or not traces:
                raise ValueError("PKL dictionary must contain a nonempty 'binned_traces' region dictionary")
            if SC_rescaled_array is not None:
                if list(SC.index) != list(SC.columns):
                    raise ValueError("SC row and column region labels must match in the same order")
                regions = [region for region in SC.index if region in traces]
                if not regions:
                    raise ValueError("No common region labels between SC and binned_traces")
                missing = [region for region in SC.index if region not in traces]
                extra = [region for region in traces if region not in SC.index]
                if missing:
                    print(f"Excluding SC regions without activity: {missing}")
                if extra:
                    print(f"Excluding activity regions without SC: {extra}")
                sc_labels = regions
                indices = SC.index.get_indexer(regions)
                SC_rescaled_array = SC_rescaled_array[np.ix_(indices, indices)]
            else:
                regions = list(traces)
            values = [np.asarray(traces[region]) for region in regions]
            if any(value.ndim != 1 for value in values):
                raise ValueError("Each binned_traces entry must be a one-dimensional time series")
            if len({len(value) for value in values}) != 1:
                raise ValueError("All regional time series must have the same length")
            FC_array = np.stack(values, axis=0)
            print(f"Converted binned_traces to array in region order: {len(regions)} regions")

        original_shape = FC_array.shape
        print(f"Loaded functional connectivity data: {fc_file}, original shape={original_shape}")
        
        # Apply slicing
        if FC_array.ndim == 2:
            # Assume shape is (num_regions, time_points)
            region_slice = parse_slice(fc_region_slice) if fc_region_slice else slice(None)
            time_slice = parse_slice(fc_time_slice) if fc_time_slice else slice(None)
            
            FC_array = FC_array[region_slice, time_slice]
            print(f"Applied slicing - regions: {fc_region_slice if fc_region_slice else 'all'}, "
                  f"time: {fc_time_slice if fc_time_slice else 'all'}")
            print(f"Shape after slicing={FC_array.shape}")
        else:
            print(f"Warning: Data dimension is {FC_array.ndim}, automatic slicing not supported, please process data manually")
            
    else:
        raise FileNotFoundError(f"Functional connectivity data file does not exist: {fc_file}")
    
    if SC_rescaled_array is not None:
        if fc_region_slice:
            region_slice = parse_slice(fc_region_slice)
            SC_rescaled_array = SC_rescaled_array[region_slice, :][:, region_slice]
            sc_labels = sc_labels[region_slice]
        if FC_array.ndim == 2 and SC_rescaled_array.shape != (FC_array.shape[0], FC_array.shape[0]):
            raise ValueError(f"SC shape {SC_rescaled_array.shape} does not match activity region count {FC_array.shape[0]}")
        # Match the original notebook: scale each column after region alignment.
        SC_rescaled_array = MinMaxScaler().fit_transform(SC_rescaled_array)
        print(f"Final aligned SC shape={SC_rescaled_array.shape}")
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            pd.DataFrame(SC_rescaled_array, index=sc_labels, columns=sc_labels).to_csv(
                os.path.join(save_dir, "aligned_sc.csv"))

    return SC_rescaled_array, FC_array


# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(
        description='Time Series Dynamic Graph Learning Model Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # File path parameters
    parser.add_argument('--sc_file', type=str, default='DK68_structural_connectivity.csv',
                        help='Path to structural connectivity matrix file (CSV format)')
    parser.add_argument('--fc_file', type=str, default='N3_eeg_normalized.pkl',
                        help='Path to functional connectivity data file (PKL or NPY format)')
    parser.add_argument('--save_dir', type=str, default='./results/default',
                        help='Directory to save results')
    
    # Data slicing parameters
    parser.add_argument('--fc_time_slice', type=str, default=None,
                        help='Time dimension slice for functional connectivity data, format: "start:end" or "start:end:step", e.g., "0:1000" for first 1000 time points')
    parser.add_argument('--fc_region_slice', type=str, default=None,
                        help='Region dimension slice for functional connectivity data, format: "start:end" or "start:end:step", e.g., "0:68" for first 68 regions')
    
    # Model hyperparameters
    parser.add_argument('--seq_length', type=int, default=50,
                        help='Input sequence length')
    parser.add_argument('--pred_length', type=int, default=10,
                        help='Prediction length')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden layer dimension')
    parser.add_argument('--embed_dim', type=int, default=10,
                        help='Node embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--head_dim', type=int, default=8,
                        help='Dimension of each attention head')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation set ratio')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--struct_constraint_weight', type=float, default=1,
                        help='Structural constraint weight')
    
    # Graph type parameter
    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument('--undirected', action='store_true',
                             help='Enable adjacency symmetrization before SC clipping; default is directed')
    graph_group.add_argument('--directed', dest='undirected', action='store_false',
                             help='Use directed graph (default; accepted for compatibility)')
    parser.set_defaults(undirected=False)
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu), default auto-select')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to generate visualization plots')
    parser.add_argument('--plot_only', action='store_true',
                        help='Only generate visualization plots (no training)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("Time Series Dynamic Graph Learning Model")
    print("=" * 60)
    
    # Detailed device information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"Current device: {device}")
    else:
        print("⚠️  Warning: No GPU detected, will use CPU for training (slower)")
        print("   Possible reasons:")
        print("   1. PyTorch is installed with CPU-only version")
        print("   2. NVIDIA GPU driver is not installed")
        print("   3. CUDA version does not match PyTorch")
        print("   4. System does not have NVIDIA GPU")
        print(f"   Current device: {device}")
    
    print(f"Random seed: {args.seed}")
    print(f"Save directory: {args.save_dir}")
    print("-" * 60)
    
    # If only plotting, directly load results
    if args.plot_only:
        if not os.path.exists(args.save_dir):
            raise FileNotFoundError(f"Save directory does not exist: {args.save_dir}")
        
        load_data(args.sc_file, args.fc_file, args.fc_time_slice,
                  args.fc_region_slice, is_undirected=args.undirected,
                  save_dir=args.save_dir)
        print("Generating visualization plots only...")
        plot_dynamic_graph_results(args.save_dir, struct_connectome_array=None, struct_connectome_file=args.sc_file)
        plot_dynamic_graph_single(args.save_dir, args.sc_file)
        plot_training_epoch_metrics(args.save_dir)
        print("Done!")
        return
    
    # Load data
    print("Loading data...")
    # Directed by default; symmetrize only when --undirected is specified.
    is_undirected = args.undirected
    struct_connectome, cell_trace = load_data(
        args.sc_file, 
        args.fc_file,
        fc_time_slice=args.fc_time_slice,
        fc_region_slice=args.fc_region_slice,
        is_undirected=is_undirected,
        save_dir=args.save_dir
    )
    
    # Train model
    print("\nStarting training...")
    model, train_losses, val_losses, full_dynamic_graphs = train_timeseries_dynamic_graph(
        cell_trace=cell_trace,
        struct_connectome=struct_connectome,
        seq_length=args.seq_length,
        pred_length=args.pred_length,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
        device=device,
        seed=args.seed,
        val_ratio=args.val_ratio,
        patience=args.patience,
        heads=args.heads,
        head_dim=args.head_dim,
        embed_dim=args.embed_dim,
        struct_constraint_weight=args.struct_constraint_weight,
        save_dir=args.save_dir,
        is_undirected=is_undirected
    )
    
    # Generate visualization
    if args.plot:
        print("\nGenerating visualization plots...")
        plot_dynamic_graph_results(args.save_dir, struct_connectome_array=struct_connectome, struct_connectome_file=args.sc_file)
        plot_dynamic_graph_single(args.save_dir, args.sc_file)
        plot_training_epoch_metrics(args.save_dir)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
