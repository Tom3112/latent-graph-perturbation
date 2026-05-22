"""Composite MSE + Pearson correlation loss for perturbation prediction."""

import torch
import torch.nn as nn


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - mean Pearson correlation across the batch. Differentiable."""
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)
    num = (pred_c * target_c).sum(dim=1)
    denom = pred_c.norm(dim=1) * target_c.norm(dim=1) + eps
    return 1.0 - (num / denom).mean()


class PerturbationLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.5, pearson_weight: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_w = mse_weight
        self.pearson_w = pearson_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse_w * self.mse(pred, target) + self.pearson_w * pearson_loss(pred, target)
