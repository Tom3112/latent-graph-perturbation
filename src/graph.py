"""Build PyTorch Geometric gene-gene interaction graph from STRING-db data."""

import urllib.request
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data


def download_string_files(data_dir: str | Path) -> tuple[Path, Path]:
    """Download STRING v12 human protein links and info files if not present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    base = "https://stringdb-downloads.org/download"
    specs = {
        "links": ("protein.links.v12.0", "9606.protein.links.v12.0.txt.gz", "~1.5 GB"),
        "info": ("protein.info.v12.0", "9606.protein.info.v12.0.txt.gz", "~50 MB"),
    }
    paths: dict[str, Path] = {}
    for key, (folder, fname, size) in specs.items():
        dest = data_dir / fname
        if not dest.exists():
            url = f"{base}/{folder}/{fname}"
            print(f"Downloading {fname} from STRING-db ({size})...")
            urllib.request.urlretrieve(url, dest)
            print(f"  Saved to {dest}")
        paths[key] = dest
    return paths["links"], paths["info"]


def load_string_protein_info(info_path: str | Path) -> dict[str, str]:
    """Return {ensembl_protein_id: gene_symbol} from STRING protein info file."""
    df = pd.read_csv(info_path, sep="\t")
    df["protein_id"] = df["#string_protein_id"].str.replace("9606.", "", regex=False)
    return dict(zip(df["protein_id"], df["preferred_name"]))


def load_string_graph(
    string_path: str | Path,
    gene_index: dict[str, int],
    min_score: int = 700,
    protein_info_path: str | Path | None = None,
) -> Data:
    """
    Build a PyG Data object from a STRING-db protein links file.

    Expects the tab-separated file from:
    https://string-db.org/cgi/download?species_text=Homo+sapiens
    (file: 9606.protein.links.v12.0.txt.gz)

    min_score: combined STRING confidence score threshold (0-1000).
    protein_info_path: path to 9606.protein.info.v12.0.txt.gz for gene-name mapping.
    """
    df = pd.read_csv(string_path, sep=" ")
    df = df[df["combined_score"] >= min_score]

    df["protein1"] = df["protein1"].str.replace("9606.", "", regex=False)
    df["protein2"] = df["protein2"].str.replace("9606.", "", regex=False)

    if protein_info_path is not None:
        id_to_gene = load_string_protein_info(protein_info_path)
        df["protein1"] = df["protein1"].map(id_to_gene)
        df["protein2"] = df["protein2"].map(id_to_gene)
        df = df.dropna(subset=["protein1", "protein2"])

    df = df[df["protein1"].isin(gene_index) & df["protein2"].isin(gene_index)]

    src = torch.tensor([gene_index[g] for g in df["protein1"]], dtype=torch.long)
    dst = torch.tensor([gene_index[g] for g in df["protein2"]], dtype=torch.long)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    scores = torch.tensor(df["combined_score"].values, dtype=torch.float)
    edge_attr = torch.cat([scores, scores]) / 1000.0

    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(gene_index))


def apply_perturbation_mask(
    x: torch.Tensor,
    gene_index: dict[str, int],
    perturbed_genes: list[str],
) -> torch.Tensor:
    """Zero-out node features for perturbed genes. x shape: [num_genes, feature_dim]."""
    x = x.clone()
    for gene in perturbed_genes:
        if gene in gene_index:
            x[gene_index[gene]] = 0.0
    return x
