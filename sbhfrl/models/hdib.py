from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .resnet import BasicBlock

class SpectrumWiseFeaturePurifier(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x + residual)
# class SpectrumWiseFeaturePurifier(nn.Module):
#     """Conv residual filter to suppress localized noise before IB purification."""

#     def __init__(self, channels: int):
#         super().__init__()
#         self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
#         self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
#         self.bn = nn.BatchNorm2d(channels)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         residual = x
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return F.relu(x + residual)

class CrossLayerAttention(nn.Module):
    def __init__(self, levels: int, dim: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(levels)])
        self.score = nn.Linear(dim, 1)

    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        stacked = []
        for proj, feat in zip(self.proj, feats):
            stacked.append(torch.tanh(proj(feat)))
        stack = torch.stack(stacked, dim=1)
        logits = self.score(stack)
        weights = torch.softmax(logits, dim=1)
        fused = (weights * stack).sum(dim=1)
        return fused, weights
# class AdaptiveSpectralAttention(nn.Module):
#     """Adaptive spectral attention over purified features with temperature."""

#     def __init__(self, levels: int, dim: int, temperature: float = 0.7):
#         super().__init__()
#         self.levels = levels
#         self.temperature = temperature
#         self.mlp = nn.Sequential(
#             nn.Linear(levels * dim, dim),
#             nn.LayerNorm(dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(dim, levels),
#         )

#     def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
#         concat = torch.cat(feats, dim=1)
#         logits = self.mlp(concat) / max(self.temperature, 1e-3)
#         weights = torch.softmax(logits, dim=1).unsqueeze(-1)
#         stacked = torch.stack(feats, dim=1)
#         fused = (weights * stacked).sum(dim=1)
#         return fused, weights

class IBHead(nn.Module):
    def __init__(self, dim: int, latent_dim: int):
        super().__init__()
        self.mu = nn.Linear(dim, latent_dim)
        self.logvar = nn.Linear(dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mu(x), self.logvar(x)
# class IBHead(nn.Module):
#     def __init__(self, dim: int, latent_dim: int):
#         super().__init__()
#         self.mu = nn.Linear(dim, latent_dim)
#         self.logvar = nn.Linear(dim, latent_dim)

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         mu = self.mu(x)
#         logvar = self.logvar(x)
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         sample = mu + eps * std
#         return mu, logvar, sample


class AlignmentHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


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
        self.stem, self.blocks, self.purifiers, self.channels = self._build_backbone(backbone)
        self.ib_heads = nn.ModuleList([IBHead(c, latent_dim) for c in self.channels])
        self.align_heads = nn.ModuleList([AlignmentHead(c, latent_dim) for c in self.channels])
        self.cross_attention = CrossLayerAttention(len(self.channels), latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def _build_backbone(self, backbone: str):
        if backbone == "resnet10":
            return self._build_resnet10_backbone()
        if backbone == "resnet18":
            return self._build_resnet18_backbone(pretrained=self.backbone_pretrained)
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
        channels = [64, 128, 256]
        blocks = nn.ModuleList([backbone.layer1, backbone.layer2, backbone.layer3])
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
            x = purifier(block(x))
            feats.append(x)
        return feats

    def forward(self, x: torch.Tensor):
        feats = self._collect(x)
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        aligned = []
        latent_stats = []
        for feat, ib_head, align_head in zip(pooled, self.ib_heads, self.align_heads):
            mu, logvar = ib_head(feat)
            latent_stats.append((mu, logvar))
            aligned.append(align_head(feat))
        fused, weights = self.cross_attention(aligned)
        embeddings = F.normalize(fused, dim=1)
        logits = self.classifier(embeddings)
        self._last_aux = {"fingerprints": aligned, "latent_stats": latent_stats, "weights": weights, "embeddings": embeddings}
        return logits, embeddings

    @property
    def aux(self):
        return self._last_aux
