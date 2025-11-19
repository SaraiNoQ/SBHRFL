import argparse
import copy
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SBHFRLConfig:
    num_shards: int = 2
    clients_per_shard: int = 5
    local_epochs: int = 1
    rounds: int = 3
    batch_size: int = 64
    proto_batch_size: int = 128
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    alpha_dirichlet: float = 0.5
    history: int = 4
    ema_decay: float = 0.7
    max_round_clients: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lambda_ib: float = 1.0
    lambda_contrast: float = 0.2
    lambda_consistency: float = 0.5
    lambda_distill: float = 0.1
    wasserstein_threshold: float = 0.8
    alpha_quality: float = 0.6
    beta_quality: float = 0.2
    gamma_quality: float = 0.2
    teacher_temperature: float = 0.07
    init_reputation: float = 0.8
    max_grad_norm: float = 5.0


class SpectrumWiseFeaturePurifier(nn.Module):
    """Spectrum-wise feature purification using depthwise + pointwise attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x + residual)


class CrossLayerAttention(nn.Module):
    def __init__(self, num_levels: int, feat_dim: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(feat_dim, feat_dim) for _ in range(num_levels)])
        self.score = nn.Linear(feat_dim, 1)

    def forward(self, features: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        stacked = []
        for proj, feat in zip(self.proj, features):
            stacked.append(torch.tanh(proj(feat)))
        stacked_feats = torch.stack(stacked, dim=1)
        logits = self.score(stacked_feats)
        weights = torch.softmax(logits, dim=1)
        fused = (weights * stacked_feats).sum(dim=1)
        return fused, weights.squeeze(-1)


class IBHead(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mu(x), self.logvar(x)


class AlignmentHead(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HDIBNet(nn.Module):
    def __init__(self, num_classes: int = 10, latent_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        channels = [64, 128, 256]
        self.features = nn.ModuleList()
        self.purifiers = nn.ModuleList()
        in_c = 64
        for out_c in channels:
            block = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
            self.features.append(block)
            self.purifiers.append(SpectrumWiseFeaturePurifier(out_c))
            in_c = out_c

        pooled_dims = [c for c in channels]
        self.ib_heads = nn.ModuleList([IBHead(dim, latent_dim) for dim in pooled_dims])
        self.align_heads = nn.ModuleList([AlignmentHead(dim, latent_dim) for dim in pooled_dims])
        self.cross_attn = CrossLayerAttention(len(channels), latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def _collect_level_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = self.stem(x)
        for block, purifier in zip(self.features, self.purifiers):
            x = block(x)
            x = purifier(x)
            feats.append(x)
        return feats

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        level_feats = self._collect_level_features(x)
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in level_feats]
        latent_stats = []
        aligned = []
        for pooled_feat, ib_head, align_head in zip(pooled, self.ib_heads, self.align_heads):
            mu, logvar = ib_head(pooled_feat)
            latent_stats.append((mu, logvar))
            aligned.append(align_head(pooled_feat))
        fused, _ = self.cross_attn(aligned)
        logits = self.classifier(fused)
        return logits, aligned, latent_stats, fused

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, fused = self.forward(x)
        return fused


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if len(features) < 2:
        return torch.tensor(0.0, device=features.device)
    features = F.normalize(features, dim=1)
    logits = torch.div(features @ features.t(), temperature)
    logits = logits - torch.max(logits, dim=1, keepdim=True).values
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
    logits = logits * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1.0)
    loss = -mean_log_prob_pos.mean()
    return loss


class HDIBLoss(nn.Module):
    def __init__(self, config: SBHFRLConfig):
        super().__init__()
        self.config = config
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        fingerprints: List[torch.Tensor],
        latent_stats: List[Tuple[torch.Tensor, torch.Tensor]],
        fused_repr: torch.Tensor,
        teacher_proto: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ce_loss = self.ce(logits, labels)
        ib_loss = sum(kl_divergence(mu, logvar) for mu, logvar in latent_stats) * self.config.lambda_ib
        contrast_loss = supervised_contrastive_loss(fused_repr, labels, temperature=0.1) * self.config.lambda_contrast
        deepest = fingerprints[-1].detach()
        consistency = torch.tensor(0.0, device=fused_repr.device)
        for fp in fingerprints[:-1]:
            consistency = consistency + F.mse_loss(fp, deepest)
        consistency = consistency * self.config.lambda_consistency
        distill = torch.tensor(0.0, device=fused_repr.device)
        if teacher_proto is not None:
            batch_teacher = teacher_proto[labels]
            distill = F.mse_loss(fused_repr, batch_teacher) * self.config.lambda_distill
        return ce_loss + ib_loss + contrast_loss + consistency + distill


def partition_dataset(dataset: Dataset, num_clients: int, alpha: float) -> List[Subset]:
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    idx_by_class = [np.where(labels == y)[0] for y in range(num_classes)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for class_indices in idx_by_class:
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)
        start = 0
        for client_id, end in enumerate(proportions):
            client_indices[client_id].extend(class_indices[start:end].tolist())
            start = end
        client_indices[-1].extend(class_indices[start:].tolist())
    subsets = [Subset(dataset, indices) for indices in client_indices]
    return subsets


def build_loaders(subsets: List[Subset], batch_size: int) -> List[DataLoader]:
    loaders = []
    for subset in subsets:
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True))
    return loaders


def prototype_from_loader(model: HDIBNet, loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    model.eval()
    proto = torch.zeros(num_classes, model.classifier.in_features, device=device)
    counts = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model.encode(images)
            proto.index_add_(0, labels, embeddings)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    counts = counts.clamp(min=1.0).unsqueeze(1)
    proto = proto / counts
    return F.normalize(proto, dim=1)


def tensorized_state_dict(state_dict: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in state_dict.items()}


def weighted_average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]], weights: torch.Tensor) -> Dict[str, torch.Tensor]:
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("Total weight must be positive for state dict averaging.")
    avg_state: Dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key] * w for sd, w in zip(state_dicts, weights)], dim=0)
        avg_state[key] = stacked.sum(dim=0) / total_weight
    return avg_state


def weighted_average_prototypes(prototypes: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    stacked = torch.stack(prototypes)
    weights = weights.view(-1, 1, 1)
    summed = (stacked * weights).sum(dim=0)
    return F.normalize(summed / weights.sum(), dim=1)


def estimate_density(proto: torch.Tensor) -> float:
    valid = proto[proto.abs().sum(dim=1) > 0]
    if valid.numel() == 0:
        return 0.0
    center = valid.mean(dim=0, keepdim=True)
    dist = torch.norm(valid - center, p=2, dim=1).mean().item()
    return 1.0 / (dist + 1e-6)


class ClientNode:
    def __init__(self, client_id: int, shard_id: int, loader: DataLoader, config: SBHFRLConfig):
        self.client_id = client_id
        self.shard_id = shard_id
        self.loader = loader
        self.config = config
        self.reputation = config.init_reputation + 0.1 * random.random()

    def run_round(
        self,
        global_state: Dict[str, torch.Tensor],
        teacher_proto: Optional[torch.Tensor],
        device: torch.device,
    ) -> Dict:
        model = HDIBNet().to(device)
        model.load_state_dict(global_state)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        criterion = HDIBLoss(self.config).to(device)
        model.train()
        for _ in range(self.config.local_epochs):
            for images, labels in self.loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits, fingerprints, latent_stats, fused = model(images)
                distill_proto = teacher_proto.to(device) if teacher_proto is not None else None
                loss = criterion(logits, labels, fingerprints, latent_stats, fused, distill_proto)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                break
        prototype_loader = DataLoader(self.loader.dataset, batch_size=self.config.proto_batch_size, shuffle=False)
        prototypes = prototype_from_loader(model, prototype_loader, num_classes=10, device=device)
        payload = {
            "state_dict": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
            "prototypes": prototypes.cpu(),
            "metrics": {
                "density": estimate_density(prototypes.to(device)),
                "channel": np.random.beta(5, 2),
                "reputation": self.reputation,
            },
        }
        return payload


class QualityAwarePrototypeFusion:
    def __init__(self, config: SBHFRLConfig):
        self.config = config

    def fuse(self, shard_id: int, payloads: List[Dict]) -> Dict:
        weights = []
        for payload in payloads:
            metrics = payload["metrics"]
            w = (
                self.config.alpha_quality * metrics["density"]
                + self.config.beta_quality * metrics["channel"]
                + self.config.gamma_quality * metrics["reputation"]
            )
            weights.append(max(w, 1e-6))
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        fused_state = weighted_average_state_dicts([{k: v.float() for k, v in payload["state_dict"].items()} for payload in payloads], weight_tensor)
        fused_proto = weighted_average_prototypes([payload["prototypes"].float() for payload in payloads], weight_tensor)
        shard_summary = {
            "shard_id": shard_id,
            "state_dict": fused_state,
            "prototypes": fused_proto,
            "weights": weight_tensor.sum().item(),
        }
        return shard_summary


class ReputationConsensus:
    def __init__(self, config: SBHFRLConfig):
        self.config = config
        self.shard_reputation: Dict[int, float] = defaultdict(lambda: config.init_reputation)
        self.blacklist: Dict[int, int] = defaultdict(int)

    def wasserstein_distance(self, protos: torch.Tensor) -> torch.Tensor:
        flat = protos.view(protos.size(0), -1)
        dist = torch.cdist(flat, flat, p=2)
        return dist

    def audit(self, shard_summaries: List[Dict]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        protos = torch.stack([summary["prototypes"].float() for summary in shard_summaries])
        distances = self.wasserstein_distance(protos)
        mean_distance = distances.mean(dim=1)
        valid_states = []
        valid_protos = []
        weights = []
        for idx, summary in enumerate(shard_summaries):
            shard_id = summary["shard_id"]
            rep = self.shard_reputation[shard_id]
            dist_score = math.exp(-mean_distance[idx].item())
            if dist_score < self.config.wasserstein_threshold:
                self.blacklist[shard_id] += 1
                continue
            new_rep = 0.6 * rep + 0.4 * dist_score
            self.shard_reputation[shard_id] = new_rep
            valid_states.append(summary["state_dict"])
            valid_protos.append(summary["prototypes"])
            weights.append(new_rep)
        if not valid_states:
            raise RuntimeError("All shards flagged as malicious; reduce strictness or check training signals.")
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        global_state = weighted_average_state_dicts(valid_states, weight_tensor)
        global_proto = weighted_average_prototypes(valid_protos, weight_tensor)
        return global_state, global_proto


class BlockchainMemory:
    def __init__(self, config: SBHFRLConfig):
        self.history: Deque[torch.Tensor] = deque(maxlen=config.history)
        self.config = config

    def update(self, prototypes: torch.Tensor) -> None:
        self.history.append(prototypes.cpu())

    def teacher(self) -> Optional[torch.Tensor]:
        if not self.history:
            return None
        weights = []
        proto_stack = []
        for idx, proto in enumerate(reversed(self.history)):
            decay = self.config.ema_decay ** idx
            weights.append(decay)
            proto_stack.append(proto)
        stacked = torch.stack(proto_stack)
        weight_tensor = torch.tensor(weights).view(-1, 1, 1)
        ema = (stacked * weight_tensor).sum(dim=0) / weight_tensor.sum()
        return F.normalize(ema, dim=1)


def evaluate(model: HDIBNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, _, _, _ = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def create_clients(loaders: List[DataLoader], config: SBHFRLConfig) -> List[ClientNode]:
    clients = []
    shard_size = config.clients_per_shard
    for client_idx, loader in enumerate(loaders):
        shard_id = client_idx // shard_size
        clients.append(ClientNode(client_idx, shard_id, loader, config))
    return clients


def run_training(config: SBHFRLConfig) -> None:
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    num_clients = config.num_shards * config.clients_per_shard
    client_subsets = partition_dataset(train_dataset, num_clients, config.alpha_dirichlet)
    client_loaders = build_loaders(client_subsets, config.batch_size)
    clients = create_clients(client_loaders, config)
    shard_groups: Dict[int, List[ClientNode]] = defaultdict(list)
    for client in clients:
        shard_groups[client.shard_id].append(client)

    global_model = HDIBNet().to(config.device)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    shard_fusion = QualityAwarePrototypeFusion(config)
    consensus = ReputationConsensus(config)
    blockchain = BlockchainMemory(config)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    for round_idx in range(config.rounds):
        shard_summaries = []
        teacher_proto = blockchain.teacher()
        for shard_id, shard_clients in shard_groups.items():
            participating_clients = random.sample(shard_clients, min(config.max_round_clients, len(shard_clients)))
            payloads = []
            for client in participating_clients:
                payload = client.run_round(global_state, teacher_proto, torch.device(config.device))
                payloads.append(payload)
            shard_summary = shard_fusion.fuse(shard_id, payloads)
            shard_summaries.append(shard_summary)
        global_state, global_proto = consensus.audit(shard_summaries)
        blockchain.update(global_proto)
        global_model.load_state_dict(global_state)
        acc = evaluate(global_model.to(config.device), test_loader, torch.device(config.device))
        print(f"[Round {round_idx + 1}] Test Accuracy: {acc * 100:.2f}% | Shards audited: {len(shard_summaries)}")


def parse_args() -> SBHFRLConfig:
    parser = argparse.ArgumentParser(description="SB-HFRL CIFAR10 Simulation")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--clients-per-shard", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-round-clients", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda-ib", type=float, default=1.0)
    parser.add_argument("--lambda-contrast", type=float, default=0.2)
    parser.add_argument("--lambda-consistency", type=float, default=0.5)
    parser.add_argument("--lambda-distill", type=float, default=0.1)
    parser.add_argument("--wasserstein-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = SBHFRLConfig(
        rounds=args.rounds,
        num_shards=args.num_shards,
        clients_per_shard=args.clients_per_shard,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        max_round_clients=args.max_round_clients,
        lr=args.lr,
        lambda_ib=args.lambda_ib,
        lambda_contrast=args.lambda_contrast,
        lambda_consistency=args.lambda_consistency,
        lambda_distill=args.lambda_distill,
        wasserstein_threshold=args.wasserstein_threshold,
    )
    set_seed(args.seed)
    return config


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
