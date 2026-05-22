"""Single-cell data loading and preprocessing for perturbation modeling."""

from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc


def load_replogle(path: str | Path) -> ad.AnnData:
    """Load Replogle 2022 Perturb-seq .h5ad and apply standard QC."""
    adata = sc.read_h5ad(path)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Normalize to 10k counts per cell, log1p transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def split_control_perturbed(adata: ad.AnnData, perturbation_col: str = "perturbation"):
    """Split AnnData into control and perturbed cell populations."""
    mask_ctrl = adata.obs[perturbation_col] == "control"
    return adata[mask_ctrl].copy(), adata[~mask_ctrl].copy()


def get_gene_index(adata: ad.AnnData) -> dict[str, int]:
    """Return {gene_name: column_index} mapping from var_names."""
    return {gene: idx for idx, gene in enumerate(adata.var_names)}
