"""Latent Graph Perturbation Engine: Geneformer (frozen) + GAT + MLP decoder."""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from transformers import AutoModel, AutoTokenizer


class GeneformerEncoder(nn.Module):
    """Frozen Geneformer encoder — extracts [CLS] cell-state embeddings."""

    MODEL_ID = "ctheodoris/Geneformer"

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.backbone = AutoModel.from_pretrained(self.MODEL_ID)

        # Freeze all Geneformer weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        geneformer_dim = self.backbone.config.hidden_size
        self.proj = nn.Linear(geneformer_dim, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [B, hidden_size]
        return self.proj(cls)  # [B, hidden_dim]


class GeneGAT(nn.Module):
    """Multi-head GAT over gene-gene interaction graph to propagate perturbation cascades."""

    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=1, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=1, concat=False)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x, edge_index, edge_attr.unsqueeze(-1)))
        x = self.act(self.conv2(x, edge_index, edge_attr.unsqueeze(-1)))
        return x  # [num_genes, hidden_channels]


class PerturbationDecoder(nn.Module):
    """MLP that maps fused cell + graph embeddings to post-perturbation expression profile."""

    def __init__(self, cell_dim: int, gene_dim: int, num_genes: int):
        super().__init__()
        in_dim = cell_dim + gene_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_genes),
        )

    def forward(self, cell_emb: torch.Tensor, graph_summary: torch.Tensor) -> torch.Tensor:
        """
        cell_emb:      [B, cell_dim]
        graph_summary: [gene_dim]  — mean-pooled GAT output, broadcast over batch
        """
        graph_summary = graph_summary.unsqueeze(0).expand(cell_emb.size(0), -1)
        return self.mlp(torch.cat([cell_emb, graph_summary], dim=-1))


class LatentGraphPerturbationEngine(nn.Module):
    def __init__(self, num_genes: int, cell_hidden: int = 256, gene_hidden: int = 128, gat_heads: int = 4):
        super().__init__()
        self.encoder = GeneformerEncoder(hidden_dim=cell_hidden)
        self.gat = GeneGAT(in_channels=1, hidden_channels=gene_hidden, heads=gat_heads)
        self.decoder = PerturbationDecoder(cell_dim=cell_hidden, gene_dim=gene_hidden, num_genes=num_genes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_features: torch.Tensor,  # [num_genes, 1] — masked expression values
        graph: Data,
    ) -> torch.Tensor:
        cell_emb = self.encoder(input_ids, attention_mask)
        gene_emb = self.gat(node_features, graph.edge_index, graph.edge_attr)
        graph_summary = gene_emb.mean(dim=0)
        return self.decoder(cell_emb, graph_summary)
