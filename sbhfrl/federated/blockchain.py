from collections import deque
from typing import Deque, Optional

import torch


class BlockchainMemory:
    def __init__(self, history: int, ema: float):
        self.history: Deque[torch.Tensor] = deque(maxlen=history)
        self.decay = ema

    def update(self, proto: torch.Tensor) -> None:
        self.history.append(proto.cpu())

    def teacher(self) -> Optional[torch.Tensor]:
        if not self.history:
            return None
        weights = []
        stack = []
        for idx, proto in enumerate(reversed(self.history)):
            decay = self.decay ** idx
            weights.append(decay)
            stack.append(proto)
        stacked = torch.stack(stack)
        weight_tensor = torch.tensor(weights).view(-1, 1, 1)
        ema = (stacked * weight_tensor).sum(dim=0) / weight_tensor.sum()
        return torch.nn.functional.normalize(ema, dim=1)
