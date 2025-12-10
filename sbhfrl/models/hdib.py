from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .resnet import BasicBlock


class SpectrumWiseFeaturePurifier(nn.Module):
    """Lightweight spatial purifier using depthwise separable convs."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x + residual)


class CrossLayerAttention(nn.Module):
    """Adaptive spectral attention over latent vectors."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # feats: [B, K, D]
        scores = self.attn_fc(feats)  # [B, K, 1]
        weights = torch.softmax(scores, dim=1)
        fused = (weights * feats).sum(dim=1)
        return fused, weights


class VariationalProjector(nn.Module):
    """Variational bottleneck that outputs mu/logvar and projects to latent_dim."""

    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(in_dim, latent_dim)
        self.logvar_head = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc(x)
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(min=-5.0, max=5.0)
        return mu, logvar


class HDIBNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 128,
        backbone: str = "custom",
        backbone_pretrained: bool = True,
    ):
        super().__init__()
        self.embedding_dim = latent_dim
        self._last_aux = {}
        self.backbone_type = backbone
        self.backbone_pretrained = backbone_pretrained
        self.res_inplanes = 64
        self.stem, self.blocks, self.purifiers, self.channels = self._build_backbone(backbone, backbone_pretrained)
        self.projectors = nn.ModuleList([VariationalProjector(c, latent_dim) for c in self.channels])
        self.cross_attention = CrossLayerAttention(latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar).clamp(min=1e-3)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _build_backbone(self, backbone: str, pretrained: bool):
        if backbone == "resnet10":
            return self._build_resnet10_backbone()
        if backbone == "resnet18":
            return self._build_resnet18_backbone(pretrained)
        return self._build_custom_backbone()

    def _build_custom_backbone(self):
        stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        channels = [64, 128, 256]
        blocks = nn.ModuleList()
        purifiers = nn.ModuleList()
        in_c = 64
        for out_c in channels:
            block = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
            blocks.append(block)
            purifiers.append(SpectrumWiseFeaturePurifier(out_c))
            in_c = out_c
        return stem, blocks, purifiers, channels

    def _build_resnet10_backbone(self):
        stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        channels = [64, 128, 256]
        blocks = nn.ModuleList()
        purifiers = nn.ModuleList()
        self.res_inplanes = 64
        strides = [1, 2, 2]
        for planes, stride in zip(channels, strides):
            block = self._make_res_layer(planes, blocks=1, stride=stride)
            blocks.append(block)
            purifiers.append(SpectrumWiseFeaturePurifier(planes))
        return stem, blocks, purifiers, channels

    def _build_resnet18_backbone(self, pretrained: bool = True):
        try:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
        except Exception:
            backbone = resnet18(weights=None)
        stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        channels = [64, 128, 256, 512]
        blocks = nn.ModuleList([backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4])
        purifiers = nn.ModuleList([SpectrumWiseFeaturePurifier(c) for c in channels])
        return stem, blocks, purifiers, channels

    def _make_res_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.res_inplanes, planes, stride=stride)]
        self.res_inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.res_inplanes, planes))
        return nn.Sequential(*layers)

    def _collect(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = self.stem(x)
        for block, purifier in zip(self.blocks, self.purifiers):
            feat = purifier(block(x))
            feats.append(feat)
            x = feat
        return feats

    def forward(self, x: torch.Tensor):
        feats = self._collect(x)
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        sampled = []
        mus = []
        logvars = []
        for feat, projector in zip(pooled, self.projectors):
            mu, logvar = projector(feat)
            z = self._reparameterize(mu, logvar)
            mus.append(mu)
            logvars.append(logvar)
            sampled.append(z)
        stacked = torch.stack(sampled, dim=1)
        fused, weights = self.cross_attention(stacked)
        embeddings = F.normalize(fused, dim=1)
        logits = self.classifier(embeddings)
        self._last_aux = {
            "mus": mus,
            "logvars": logvars,
            "sampled_feats": sampled,
            "weights": weights,
            "embeddings": embeddings,
        }
        return logits, embeddings

    @property
    def aux(self):
        return self._last_aux
