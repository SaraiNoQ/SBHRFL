from typing import Dict, List

import torch


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
    fused = (stacked * weights).sum(dim=0) / weights.sum()
    return torch.nn.functional.normalize(fused, dim=1)


def estimate_density(proto: torch.Tensor) -> float:
    valid = proto[proto.abs().sum(dim=1) > 0]
    if valid.numel() == 0:
        return 0.0
    center = valid.mean(dim=0, keepdim=True)
    dist = torch.norm(valid - center, p=2, dim=1).mean().item()
    return 1.0 / (dist + 1e-6)


class SimpleAggregator:
    def fuse(self, shard_id: int, payloads: List[Dict]) -> Dict:
        weights = torch.tensor([payload["num_samples"] for payload in payloads], dtype=torch.float32)
        state_dicts = [{k: v.float() for k, v in payload["state_dict"].items()} for payload in payloads]
        prototypes = [payload["prototypes"].float() for payload in payloads]
        summary = {
            "shard_id": shard_id,
            "state_dict": _avg_state_dicts(state_dicts, weights),
            "prototypes": _avg_prototypes(prototypes, weights),
            "weight": weights.sum().item(),
        }
        return summary


class QualityAwareAggregator:
    def __init__(self, alpha: float, beta: float, gamma: float):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

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
            "prototypes": _avg_prototypes(prototypes, weight_tensor),
            "weight": weight_tensor.sum().item(),
        }
        return summary
