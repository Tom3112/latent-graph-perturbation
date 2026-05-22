"""Build PyTorch Geometric gene-gene interaction graph from STRING-db data."""

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data


def load_string_graph(
    string_path: str | Path,
    gene_index: dict[str, int],
    min_score: int = 700,
) -> Data:
    """
    Build a PyG Data object from a STRING-db protein links file.

    Expects the tab-separated file from:
    https://string-db.org/cgi/download?species_text=Homo+sapiens
    (file: 9606.protein.links.v12.0.txt.gz)

    min_score: combined STRING confidence score threshold (0-1000).
    """
    df = pd.read_csv(string_path, sep=" ")
    df = df[df["combined_score"] >= min_score]

    # STRING uses Ensembl protein IDs — strip the "9606." prefix
    df["protein1"] = df["protein1"].str.replace("9606.", "", regex=False)
    df["protein2"] = df["protein2"].str.replace("9606.", "", regex=False)

    # Keep only edges where both proteins map to genes in our expression matrix
    df = df[df["protein1"].isin(gene_index) & df["protein2"].isin(gene_index)]

    src = torch.tensor([gene_index[g] for g in df["protein1"]], dtype=torch.long)
    dst = torch.tensor([gene_index[g] for g in df["protein2"]], dtype=torch.long)
    edge_index = torch.stack([
        torch.cat([src, dst]),  # undirected: add both directions
        torch.cat([dst, src]),
    ])

    scores = torch.tensor(df["combined_score"].values, dtype=torch.float)
    edge_attr = torch.cat([scores, scores]) / 1000.0  # normalize to [0, 1]

    num_nodes = len(gene_index)
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def apply_perturbation_mask(
    x: torch.Tensor,
    gene_index: dict[str, int],
    perturbed_genes: list[str],
) -> torch.Tensor:
    """Zero-out the expression values of knocked-out genes (in-silico CRISPR)."""
    x = x.clone()
    for gene in perturbed_genes:
        if gene in gene_index:
            x[:, gene_index[gene]] = 0.0
    return x
