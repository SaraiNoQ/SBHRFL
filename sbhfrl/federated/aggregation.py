from typing import Dict, List

import torch
import torch.nn.functional as F


def _avg_state_dicts(state_dicts: List[Dict[str, torch.Tensor]], weights: torch.Tensor) -> Dict[str, torch.Tensor]:
    total = weights.sum()
    if total <= 0:
        raise ValueError("Invalid aggregation weights.")
    averaged = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key] * w for sd, w in zip(state_dicts, weights)], dim=0)
        averaged[key] = stacked.sum(dim=0) / total
    return averaged


def _avg_prototypes(prototypes: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    stacked = torch.stack(prototypes)
    weights = weights.view(-1, 1, 1)
    fused = (stacked * weights).sum(dim=0) / weights.sum().clamp(min=1e-6)
    return torch.nn.functional.normalize(fused, dim=1)


def _clustered_prototypes(prototypes: List[torch.Tensor], weights: torch.Tensor, threshold: float) -> torch.Tensor:
    stacked = torch.stack(prototypes)
    num_classes = stacked.size(1)
    fused = []
    eps = 1e-6
    for class_idx in range(num_classes):
        vecs = stacked[:, class_idx, :]
        clusters: List[Dict[str, torch.Tensor]] = []
        for vec, weight in zip(vecs, weights):
            if torch.allclose(vec, torch.zeros_like(vec)):
                continue
            assigned = False
            for cluster in clusters:
                sim = F.cosine_similarity(vec.unsqueeze(0), cluster["center"].unsqueeze(0), dim=1).item()
                if sim >= threshold:
                    cluster["sum"].add_(vec * weight)
                    cluster["weight"] += float(weight.item())
                    cluster["center"] = cluster["sum"] / max(cluster["weight"], eps)
                    assigned = True
                    break
            if not assigned:
                clusters.append({
                    "sum": vec.clone() * weight,
                    "weight": float(weight.item()),
                    "center": vec.clone(),
                })
        if not clusters:
            fused.append(torch.zeros_like(vecs[0]))
        else:
            best = max(clusters, key=lambda item: item["weight"])
            fused.append(F.normalize(best["center"], dim=0))
    return torch.stack(fused, dim=0)


def estimate_density(proto: torch.Tensor) -> float:
    valid = proto[proto.abs().sum(dim=1) > 0]
    if valid.numel() == 0:
        return 0.0
    center = valid.mean(dim=0, keepdim=True)
    dist = torch.norm(valid - center, p=2, dim=1).mean().item()
    return 1.0 / (dist + 1e-6)


def _merge_prototypes(prototypes: List[torch.Tensor], weights: torch.Tensor, use_cluster: bool, threshold: float) -> torch.Tensor:
    if use_cluster:
        return _clustered_prototypes(prototypes, weights, threshold)
    return _avg_prototypes(prototypes, weights)


class SimpleAggregator:
    def __init__(self, use_cluster: bool = False, cluster_threshold: float = 0.8):
        self.use_cluster = use_cluster
        self.cluster_threshold = cluster_threshold

    def fuse(self, shard_id: int, payloads: List[Dict]) -> Dict:
        weights = torch.tensor([payload["num_samples"] for payload in payloads], dtype=torch.float32)
        state_dicts = [{k: v.float() for k, v in payload["state_dict"].items()} for payload in payloads]
        prototypes = [payload["prototypes"].float() for payload in payloads]
        summary = {
            "shard_id": shard_id,
            "state_dict": _avg_state_dicts(state_dicts, weights),
            "prototypes": _merge_prototypes(prototypes, weights, self.use_cluster, self.cluster_threshold),
            "weight": weights.sum().item(),
        }
        return summary


class QualityAwareAggregator:
    def __init__(self, alpha: float, beta: float, gamma: float, use_cluster: bool = False, cluster_threshold: float = 0.8):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_cluster = use_cluster
        self.cluster_threshold = cluster_threshold

    def fuse(self, shard_id: int, payloads: List[Dict]) -> Dict:
        weights = []
        for payload in payloads:
            metrics = payload["metrics"]
            w = self.alpha * metrics["density"] + self.beta * metrics["channel"] + self.gamma * metrics["reputation"]
            weights.append(max(w, 1e-6))
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        state_dicts = [{k: v.float() for k, v in payload["state_dict"].items()} for payload in payloads]
        prototypes = [payload["prototypes"].float() for payload in payloads]
        summary = {
            "shard_id": shard_id,
            "state_dict": _avg_state_dicts(state_dicts, weight_tensor),
            "prototypes": _merge_prototypes(prototypes, weight_tensor, self.use_cluster, self.cluster_threshold),
            "weight": weight_tensor.sum().item(),
        }
        return summary
