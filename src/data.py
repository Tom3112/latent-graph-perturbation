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
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def load_norman(path: str | Path) -> ad.AnnData:
    """Load Norman 2019 scPerturb h5ad (already QC'd and normalized)."""
    return sc.read_h5ad(path)


def split_control_perturbed(adata: ad.AnnData, perturbation_col: str = "perturbation"):
    """Split AnnData into control and perturbed cell populations."""
    mask_ctrl = adata.obs[perturbation_col] == "control"
    return adata[mask_ctrl].copy(), adata[~mask_ctrl].copy()


def get_gene_index(adata: ad.AnnData) -> dict[str, int]:
    """Return {gene_name: column_index} mapping from var_names."""
    return {gene: idx for idx, gene in enumerate(adata.var_names)}


def get_perturbation_genes(perturbation: str) -> list[str]:
    """Parse 'GENE1_GENE2' → ['GENE1', 'GENE2'], 'control' → []."""
    if perturbation == "control":
        return []
    return perturbation.split("_")


def compute_mean_profiles(
    adata: ad.AnnData, perturbation_col: str = "perturbation"
) -> dict[str, np.ndarray]:
    """Return {perturbation: mean_expression_vector} for every condition."""
    profiles: dict[str, np.ndarray] = {}
    for pert in adata.obs[perturbation_col].unique():
        mask = adata.obs[perturbation_col] == pert
        X = adata[mask].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        profiles[pert] = np.mean(X, axis=0).astype(np.float32)
    return profiles
