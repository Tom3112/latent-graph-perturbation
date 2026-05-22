# Latent Graph Perturbation Engine

Predict post-perturbation single-cell gene expression by combining a frozen [Geneformer](https://huggingface.co/ctheodoris/Geneformer) cell-state encoder with a Graph Attention Network (GAT) over the STRING protein-interaction graph.

## Architecture

```
Input cells (Perturb-seq)
        │
        ▼
┌─────────────────────┐
│  Geneformer Encoder │  ← frozen, extracts [CLS] cell-state embeddings
│  + Linear proj      │
└────────┬────────────┘
         │ cell_emb [B, 256]
         │
         │        STRING gene-gene graph
         │               │
         │               ▼
         │    ┌──────────────────┐
         │    │  GATv2 (2 layers)│  ← propagates perturbation cascades
         │    │  edge_attr = conf│
         │    └────────┬─────────┘
         │             │ gene_emb → mean-pool → graph_summary [128]
         │             │
         └──────┬──────┘
                ▼
       ┌─────────────────┐
       │  MLP Decoder    │  → predicted expression [B, num_genes]
       └─────────────────┘
```

**Key design choices:**
- Geneformer weights are fully frozen; only the projection layer, GAT, and decoder are trained.
- STRING combined score ≥ 700 (high confidence) is the default edge filter.
- Loss = 0.5 × MSE + 0.5 × (1 − Pearson correlation), operating on the full expression profile.
- In-silico CRISPR KO is modeled by zeroing the node features of perturbed genes before the GAT pass.

## Data

| Resource | Description | Download |
|---|---|---|
| Replogle 2022 | Perturb-seq .h5ad (~2M cells, essential genes screen) | [Figshare](https://figshare.com/articles/dataset/Replogle_et_al_2022_essential_gene_perturbation/20029387) |
| STRING v12 | Human protein links | [string-db.org](https://string-db.org/cgi/download?species_text=Homo+sapiens) → `9606.protein.links.v12.0.txt.gz` |

Place files anywhere; pass paths via CLI flags.

## Setup

```bash
# requires Python 3.12 and uv
uv sync
```

## Usage

```bash
python -m src.train \
  --data_path /path/to/replogle_essential.h5ad \
  --string_path /path/to/9606.protein.links.v12.0.txt.gz \
  --string_min_score 700 \
  --batch_size 32 \
  --lr 1e-4 \
  --epochs 20
```

> **Note:** The Geneformer tokenization pipeline (`build_dataloaders` in `src/train.py`) is a stub. See `notebooks/01_tokenize.ipynb` for the full tokenization walkthrough once Geneformer weights are confirmed.

## Project Structure

```
src/
  data.py    — Replogle h5ad loading, QC, control/perturbed split
  graph.py   — STRING graph construction + perturbation masking
  loss.py    — MSE + Pearson composite loss
  model.py   — GeneformerEncoder, GeneGAT, PerturbationDecoder, engine
  train.py   — CLI training entry point
notebooks/
  01_tokenize.ipynb  — Geneformer tokenization walkthrough (planned)
```

## Citation

If you use this code, please also cite:

- Theodoris et al. (2023) *Transfer learning enables predictions in network biology*. Nature.
- Replogle et al. (2022) *Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq*. Cell.
- Szklarczyk et al. (2023) *The STRING database in 2023*. Nucleic Acids Research.
