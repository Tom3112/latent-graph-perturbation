"""Training loop for the Latent Graph Perturbation Engine."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data

from src.data import get_gene_index, load_replogle, split_control_perturbed
from src.graph import apply_perturbation_mask, load_string_graph
from src.loss import PerturbationLoss
from src.model import LatentGraphPerturbationEngine


def build_dataloaders(adata_ctrl, adata_pert, tokenizer, batch_size: int):
    """Tokenize cells and build DataLoaders. Placeholder — adapt to Geneformer's tokenizer."""
    # Geneformer ranks genes by expression and tokenizes the top-k per cell.
    # Full tokenization pipeline to be implemented once Geneformer weights are confirmed.
    raise NotImplementedError("Geneformer tokenization pipeline — see notebooks/01_tokenize.ipynb")


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    adata = load_replogle(args.data_path)
    gene_index = get_gene_index(adata)
    adata_ctrl, adata_pert = split_control_perturbed(adata)

    graph = load_string_graph(args.string_path, gene_index, min_score=args.string_min_score)
    graph = graph.to(device)

    model = LatentGraphPerturbationEngine(num_genes=len(gene_index)).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    criterion = PerturbationLoss()

    # Full training loop added once tokenization pipeline is ready
    print("Model initialized. Run notebooks/01_tokenize.ipynb to prepare tokenized data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--string_path", type=Path, required=True)
    parser.add_argument("--string_min_score", type=int, default=700)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    train(parser.parse_args())
