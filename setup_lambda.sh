#!/usr/bin/env bash
# Run once on a fresh Lambda Labs instance to set up the environment.
# Tested on Lambda Ubuntu 22.04 with CUDA 12.4.
set -euo pipefail

# -- Python deps (torch with CUDA 12.4, then everything else) -----------------
pip install --upgrade pip

pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

pip install torch_geometric

# PyG sparse-ops extensions (match torch + CUDA version)
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA="cu124"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html" || true

pip install \
    accelerate \
    anndata \
    datasets \
    huggingface_hub \
    matplotlib \
    numpy \
    pandas \
    scanpy \
    scipy \
    seaborn \
    "transformers<5.0.0"  # Geneformer (ctheodoris/Geneformer) requires transformers 4.x

# -- Data: copy NormanWeissman2019_filtered.h5ad from your Mac ----------------
# Run this from your LOCAL machine (replace <LAMBDA_IP>):
#
#   scp data/NormanWeissman2019_filtered.h5ad ubuntu@<LAMBDA_IP>:~/latent-graph-perturbation/data/
#
# STRING files (~1.5 GB) are downloaded automatically on first run.

echo ""
echo "Setup complete. Start training with:"
echo "  python -m src.train --data_path data/NormanWeissman2019_filtered.h5ad --batch_size 32 --epochs 20"
