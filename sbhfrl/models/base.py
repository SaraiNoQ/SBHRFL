from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoBackbone(nn.Module):
    def __init__(self, num_classes: int = 10, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._last_aux = {}
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed = nn.Linear(256, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x).flatten(1)
        embeddings = F.normalize(self.embed(feats), dim=1)
        logits = self.classifier(embeddings)
        self._last_aux = {}
        return logits, embeddings

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, embeddings = self.forward(x)
        return embeddings

    @property
    def aux(self):
        return self._last_aux
