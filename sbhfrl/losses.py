from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseProtoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.ce(logits, labels)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def supervised_contrastive(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if features.size(0) < 2:
        return torch.tensor(0.0, device=features.device)
    feats = F.normalize(features, dim=1)
    logits = torch.div(feats @ feats.t(), temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
    logits = logits * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1.0)
    return -mean_log_prob_pos.mean()


class HDIBLoss(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_ib = config.get("lambda_ib", 1.0)
        self.lambda_contrast = config.get("lambda_contrast", 0.2)
        self.lambda_consistency = config.get("lambda_consistency", 0.5)
        self.lambda_distill = config.get("lambda_distill", 0.1)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        fingerprints: List[torch.Tensor],
        latent_stats: List[Tuple[torch.Tensor, torch.Tensor]],
        fused_repr: torch.Tensor,
        teacher_proto: torch.Tensor = None,
    ) -> torch.Tensor:
        loss = self.ce(logits, labels)
        if latent_stats:
            ib_loss = sum(kl_divergence(mu, logvar) for mu, logvar in latent_stats)
            loss = loss + self.lambda_ib * ib_loss
        if fused_repr is not None:
            loss = loss + self.lambda_contrast * supervised_contrastive(fused_repr, labels)
        if len(fingerprints) > 1:
            deepest = fingerprints[-1].detach()
            consistency = torch.tensor(0.0, device=logits.device)
            for finger in fingerprints[:-1]:
                consistency = consistency + F.mse_loss(finger, deepest)
            loss = loss + self.lambda_consistency * consistency
        if teacher_proto is not None:
            target = teacher_proto[labels]
            loss = loss + self.lambda_distill * F.mse_loss(fused_repr, target)
        return loss
