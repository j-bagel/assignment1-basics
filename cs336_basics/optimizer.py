from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        # lr is 'alpha', weight_decay is 'lambda'
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p.data))  # first moment
                v = state.get("v", torch.zeros_like(p.data))  # second moment

                # updates:
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                state["m"] = betas[0] * m + (1.0 - betas[0]) * grad
                state["v"] = betas[1] * v + (1.0 - betas[1]) * (grad ** 2)
                alpha_t = lr * math.sqrt(1 - betas[1] ** (t+1)) / (1 - betas[0] ** (t+1))  # adjusted alpha

                p.data -= alpha_t / (torch.sqrt(state["v"]) + eps) * state["m"]  # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1 # Increment iteration number.

        return loss


def learning_rate_scheduler(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    cosine_cycle_start: int | None = None,
) -> float:
    if cosine_cycle_start:
        # aggressive cosine scheduler, flat from warmup_iters to cosine_cycle_start
        cosine_cycle_start = max(cosine_cycle_start, warmup_iters)
        if it < warmup_iters:
            return it / warmup_iters * max_learning_rate
        elif it < cosine_cycle_start:
            return max_learning_rate
        elif it <= cosine_cycle_iters:
            return min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - cosine_cycle_start) / (cosine_cycle_iters - cosine_cycle_start))) * (max_learning_rate - min_learning_rate)
        else:
            return min_learning_rate
    else:
        # regular cosine scheduler
        if it < warmup_iters:
            return it / warmup_iters * max_learning_rate
        elif it <= cosine_cycle_iters:
            return min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
        else:
            return min_learning_rate
