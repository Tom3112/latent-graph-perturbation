"""Training loop for the Latent Graph Perturbation Engine."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from src.data import (
    compute_mean_profiles,
    get_gene_index,
    get_perturbation_genes,
    load_norman,
    split_control_perturbed,
)
from src.graph import apply_perturbation_mask, download_string_files, load_string_graph
from src.loss import PerturbationLoss
from src.model import LatentGraphPerturbationEngine


# ---------------------------------------------------------------------------
# Geneformer tokenization helpers
# ---------------------------------------------------------------------------

def _load_geneformer_dicts() -> tuple[dict, dict, dict]:
    """Download and return (token_dict, gene_name_id, gene_median) from HF Hub."""
    token_path = hf_hub_download("ctheodoris/Geneformer", "geneformer/token_dictionary_gc104M.pkl")
    name_path = hf_hub_download("ctheodoris/Geneformer", "geneformer/gene_name_id_dict_gc104M.pkl")
    median_path = hf_hub_download("ctheodoris/Geneformer", "geneformer/gene_median_dictionary_gc104M.pkl")
    with open(token_path, "rb") as f:
        token_dict = pickle.load(f)
    with open(name_path, "rb") as f:
        gene_name_id = pickle.load(f)
    with open(median_path, "rb") as f:
        gene_median = pickle.load(f)
    return token_dict, gene_name_id, gene_median


def _build_gene_token_ids(
    gene_names: list[str],
    token_dict: dict,
    gene_name_id: dict,
) -> list[int | None]:
    """Map var_names gene symbols → Geneformer token IDs (None if unmapped)."""
    ids = []
    for g in gene_names:
        ensembl = gene_name_id.get(g)
        if ensembl is None:
            ids.append(None)
        else:
            tok = token_dict.get(ensembl)
            ids.append(None if tok is None else int(tok))
    return ids


def _tokenize_expression(
    expr: np.ndarray,
    gene_names: list[str],
    gene_token_ids: list[int | None],
    gene_name_id: dict,
    gene_median: dict,
    max_length: int = 2048,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank genes by expr/median, return (input_ids, attention_mask) of length max_length."""
    normed = []
    for i, (g, tok) in enumerate(zip(gene_names, gene_token_ids)):
        if tok is None or expr[i] == 0:
            continue
        ensembl = gene_name_id.get(g)
        median = float(gene_median.get(ensembl, 1.0)) if ensembl else 1.0
        normed.append((expr[i] / max(median, 1e-8), tok))

    normed.sort(key=lambda x: x[0], reverse=True)
    token_ids = [tok for _, tok in normed[:max_length]]

    pad = max_length - len(token_ids)
    input_ids = torch.tensor(token_ids + [pad_token_id] * pad, dtype=torch.long)
    attention_mask = torch.tensor([1] * len(token_ids) + [0] * pad, dtype=torch.long)
    return input_ids, attention_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PerturbationDataset(Dataset):
    """One item per perturbation condition: (input_ids, attn_mask, node_features, target)."""

    def __init__(
        self,
        perturbations: list[str],
        mean_profiles: dict[str, np.ndarray],
        ctrl_profile: np.ndarray,
        gene_names: list[str],
        gene_index: dict[str, int],
        gene_token_ids: list[int | None],
        gene_name_id: dict,
        gene_median: dict,
        max_length: int = 2048,
    ):
        self.perturbations = perturbations
        self.mean_profiles = mean_profiles
        self.gene_index = gene_index
        self.ctrl_node_features = torch.tensor(ctrl_profile, dtype=torch.float32).unsqueeze(1)

        # Tokenize mean control cell once; reused for all conditions
        self.ctrl_input_ids, self.ctrl_attn_mask = _tokenize_expression(
            ctrl_profile, gene_names, gene_token_ids, gene_name_id, gene_median, max_length
        )

    def __len__(self) -> int:
        return len(self.perturbations)

    def __getitem__(self, idx: int):
        pert = self.perturbations[idx]
        node_features = apply_perturbation_mask(
            self.ctrl_node_features, self.gene_index, get_perturbation_genes(pert)
        )
        target = torch.tensor(self.mean_profiles[pert], dtype=torch.float32)
        return self.ctrl_input_ids, self.ctrl_attn_mask, node_features, target


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    adata_ctrl,
    adata_pert,
    batch_size: int,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    token_dict, gene_name_id, gene_median = _load_geneformer_dicts()

    gene_names = list(adata_ctrl.var_names)
    gene_index = get_gene_index(adata_ctrl)
    gene_token_ids = _build_gene_token_ids(gene_names, token_dict, gene_name_id)

    all_profiles = compute_mean_profiles(adata_pert)
    ctrl_profile = compute_mean_profiles(adata_ctrl)["control"]

    perts = [p for p in all_profiles if p != "control"]
    rng = np.random.default_rng(seed)
    rng.shuffle(perts)
    n_val = max(1, int(len(perts) * val_fraction))
    val_perts, train_perts = perts[:n_val], perts[n_val:]

    shared_kwargs = dict(
        mean_profiles=all_profiles,
        ctrl_profile=ctrl_profile,
        gene_names=gene_names,
        gene_index=gene_index,
        gene_token_ids=gene_token_ids,
        gene_name_id=gene_name_id,
        gene_median=gene_median,
    )

    train_loader = DataLoader(
        PerturbationDataset(train_perts, **shared_kwargs),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        PerturbationDataset(val_perts, **shared_kwargs),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    adata = load_norman(args.data_path)
    adata_ctrl, adata_pert = split_control_perturbed(adata)
    gene_index = get_gene_index(adata)
    num_genes = len(gene_index)
    print(
        f"Genes: {num_genes}  |  "
        f"Control cells: {len(adata_ctrl)}  |  "
        f"Perturbation conditions: {adata_pert.obs['perturbation'].nunique()}"
    )

    print("Building dataloaders (tokenizing cells)...")
    train_loader, val_loader = build_dataloaders(adata_ctrl, adata_pert, args.batch_size)
    print(f"Train conditions: {len(train_loader.dataset)}  |  Val conditions: {len(val_loader.dataset)}")

    print("Building STRING graph (downloading if needed)...")
    links_path, info_path = download_string_files(args.data_path.parent)
    graph = load_string_graph(
        links_path, gene_index, min_score=args.string_min_score, protein_info_path=info_path
    )
    graph = graph.to(device)
    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

    model = LatentGraphPerturbationEngine(num_genes=num_genes).to(device)
    if device.type == "mps":
        # MPS has limited unified memory — pin large non-trainable modules to CPU
        model.encoder.backbone.cpu()
        model.gat.cpu()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = PerturbationLoss()

    best_val_loss = float("inf")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for input_ids, attn_mask, node_feats, target in train_loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            node_feats = node_feats.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(input_ids, attn_mask, node_feats, graph)
            loss = criterion(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, attn_mask, node_feats, target in val_loader:
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                node_feats = node_feats.to(device)
                target = target.to(device)
                pred = model(input_ids, attn_mask, node_feats, graph)
                val_loss += criterion(pred, target).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = args.checkpoint_dir / "best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss}, ckpt)
            print(f"  → Saved checkpoint ({ckpt})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True,
                        help="Path to NormanWeissman2019_filtered.h5ad")
    parser.add_argument("--string_min_score", type=int, default=700)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    train(parser.parse_args())
