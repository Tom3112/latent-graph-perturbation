"""Latent Graph Perturbation Engine: Geneformer (frozen) + GAT + MLP decoder."""

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv
from transformers import AutoModel


class GeneformerEncoder(nn.Module):
    """Frozen Geneformer encoder — extracts [CLS] cell-state embeddings."""

    MODEL_ID = "ctheodoris/Geneformer"

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(self.MODEL_ID)

        for param in self.backbone.parameters():
            param.requires_grad = False

        geneformer_dim = self.backbone.config.hidden_size
        self.proj = nn.Linear(geneformer_dim, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Backbone is pinned to CPU (frozen, no grads) — move inputs, then bring CLS back
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids.cpu(), attention_mask=attention_mask.cpu())
        cls = out.last_hidden_state[:, 0, :].to(self.proj.weight.device)  # [B, hidden_size]
        return self.proj(cls)                                               # [B, hidden_dim]


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
        self.mlp = nn.Sequential(
            nn.Linear(cell_dim + gene_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_genes),
        )

    def forward(self, cell_emb: torch.Tensor, graph_summary: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([cell_emb, graph_summary], dim=-1))


class LatentGraphPerturbationEngine(nn.Module):
    def __init__(self, num_genes: int, cell_hidden: int = 256, gene_hidden: int = 128, gat_heads: int = 4):
        super().__init__()
        self.encoder = GeneformerEncoder(hidden_dim=cell_hidden)
        self.gat = GeneGAT(in_channels=1, hidden_channels=gene_hidden, heads=gat_heads)
        self.decoder = PerturbationDecoder(cell_dim=cell_hidden, gene_dim=gene_hidden, num_genes=num_genes)
        self._num_genes = num_genes

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, seq_len]
        attention_mask: torch.Tensor,  # [B, seq_len]
        node_features: torch.Tensor,   # [B, num_genes, 1] or [num_genes, 1]
        graph: Data,
    ) -> torch.Tensor:
        cell_emb = self.encoder(input_ids, attention_mask)  # [B, cell_hidden]

        # GAT may be pinned to CPU to avoid MPS OOM on large sparse graphs
        gat_dev = next(self.gat.parameters()).device
        ei = graph.edge_index.to(gat_dev)
        ea = graph.edge_attr.to(gat_dev)

        if node_features.dim() == 2:
            gene_emb = self.gat(node_features.to(gat_dev), ei, ea)
            graph_summary = gene_emb.mean(dim=0).unsqueeze(0).expand(cell_emb.size(0), -1)
        else:
            # [B, num_genes, 1] — run GAT per item; activations stay on CPU, no MPS pressure
            B = node_features.size(0)
            graph_summaries = []
            for i in range(B):
                gene_emb = self.gat(node_features[i].to(gat_dev), ei, ea)
                graph_summaries.append(gene_emb.mean(dim=0))
            graph_summary = torch.stack(graph_summaries)  # [B, gene_hidden] on gat_dev

        return self.decoder(cell_emb, graph_summary.to(cell_emb.device))  # [B, num_genes]
