from typing import Dict

from .base import ProtoBackbone
from .hdib import HDIBNet
from .resnet import ResNet10, ResNet18


def build_model(config: Dict):
    model_name = config.get("model", "base")
    num_classes = config.get("num_classes", 10)
    embedding_dim = config.get("embedding_dim", 128)
    if model_name == "hdib":
        return HDIBNet(
            num_classes=num_classes,
            latent_dim=embedding_dim,
            backbone=config.get("hdib_backbone", "custom"),
            backbone_pretrained=config.get("backbone_pretrained", True),
        )
    if model_name == "resnet18":
        return ResNet18(num_classes=num_classes, embedding_dim=embedding_dim, pretrained=config.get("backbone_pretrained", True))
    if model_name == "resnet10":
        return ResNet10(num_classes=num_classes, embedding_dim=embedding_dim)
    return ProtoBackbone(num_classes=num_classes, embedding_dim=embedding_dim)
