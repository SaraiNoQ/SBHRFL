import math
from typing import Iterable, Tuple

import torch
from torch.optim import Optimizer


class Muon(Optimizer):
    """A lightweight Muon-style optimizer with adaptive moment estimates."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError("Learning rate must be positive")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(param, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1
                update = exp_avg / denom

                # Layer-wise normalization to mimic Muon's scale invariance.
                if update.ndim > 1:
                    norm = update.norm().clamp_min(1e-6)
                    update = update / norm

                param.add_(update, alpha=-step_size)
