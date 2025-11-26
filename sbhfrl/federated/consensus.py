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
        # 先过滤显式标记的恶意摘要（模拟已知攻击者）；若全是恶意则退化为全部参与
        clean = [s for s in shard_summaries if not s.get("malicious", False)]
        summaries = clean if clean else shard_summaries

        protos = torch.stack([summary["prototypes"].float() for summary in summaries])
        flat = protos.view(protos.size(0), -1)
        dist_matrix = torch.cdist(flat, flat, p=2)
        # 忽略自身距离，避免 0 距离稀释异常分数
        mask = ~torch.eye(dist_matrix.size(0), dtype=torch.bool, device=dist_matrix.device)
        mean_dist = (dist_matrix * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        dist_score = torch.exp(-mean_dist).cpu()  # 距离越小分数越高
        kept_states = []
        kept_protos = []
        weights = []
        for idx, summary in enumerate(summaries):
            shard_id = summary["shard_id"]
            if summary.get("malicious", False):
                self.reputation[shard_id] = max(0.05, self.reputation[shard_id] * 0.5)
                continue
            base_rep = self.reputation[shard_id]
            score = dist_score[idx].item()
            if score < self.threshold:
                self.reputation[shard_id] = max(0.1, base_rep * 0.9)
                continue
            new_rep = 0.6 * base_rep + 0.4 * score
            self.reputation[shard_id] = new_rep
            kept_states.append(summary["state_dict"])
            kept_protos.append(summary["prototypes"])
            weights.append(new_rep)
        # 若全部被过滤，保留分数最高的一份，避免把所有异常又放回
        if not kept_states:
            best_idx = int(dist_score.argmax().item())
            best_summary = summaries[best_idx]
            kept_states = [best_summary["state_dict"]]
            kept_protos = [best_summary["prototypes"]]
            weights = [self.reputation[best_summary["shard_id"]]]
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        return _avg_state_dicts(kept_states, weight_tensor), _avg_prototypes(kept_protos, weight_tensor)
