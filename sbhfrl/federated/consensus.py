from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from .aggregation import _avg_prototypes, _avg_state_dicts


class SimpleConsensus:
    def aggregate(self, shard_summaries: List[Dict]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        weights = torch.tensor([summary["weight"] for summary in shard_summaries], dtype=torch.float32)
        state_dicts = [summary["state_dict"] for summary in shard_summaries]
        prototypes = [summary["prototypes"] for summary in shard_summaries]
        return _avg_state_dicts(state_dicts, weights), _avg_prototypes(prototypes, weights)


class ReputationConsensus:
    def __init__(self, threshold: float, init_reputation: float = 0.8):
        self.threshold = threshold
        self.reputation = defaultdict(lambda: init_reputation)

    def aggregate(self, shard_summaries: List[Dict]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        protos = torch.stack([summary["prototypes"].float() for summary in shard_summaries])
        flat = protos.view(protos.size(0), -1)
        dist_matrix = torch.cdist(flat, flat, p=2)
        mean_dist = dist_matrix.mean(dim=1)
        kept_states = []
        kept_protos = []
        weights = []
        for idx, summary in enumerate(shard_summaries):
            shard_id = summary["shard_id"]
            base_rep = self.reputation[shard_id]
            dist_score = torch.exp(-mean_dist[idx]).item()
            if dist_score < self.threshold:
                self.reputation[shard_id] = max(0.1, base_rep * 0.9)
                continue
            new_rep = 0.6 * base_rep + 0.4 * dist_score
            self.reputation[shard_id] = new_rep
            kept_states.append(summary["state_dict"])
            kept_protos.append(summary["prototypes"])
            weights.append(new_rep)
        if not kept_states:
            kept_states = [summary["state_dict"] for summary in shard_summaries]
            kept_protos = [summary["prototypes"] for summary in shard_summaries]
            weights = [1.0 for _ in shard_summaries]
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        return _avg_state_dicts(kept_states, weight_tensor), _avg_prototypes(kept_protos, weight_tensor)
