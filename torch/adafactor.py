import torch
from torch.optim import Optimizer
import math


class AdafactorTorch(Optimizer):

    def __init__(
        self,
        params,
        lr=None,
        beta1=None,
        eps1=1e-30,
        eps2=1e-3,
        clip_threshold=1.0,
        beta2=-0.8,
        weight_decay=0.0,
        relative_step=True,
        scale_parameter=True
    ):

        defaults = dict(
            lr=lr,
            beta1=beta1,
            eps1=eps1,
            eps2=eps2,
            clip_threshold=clip_threshold,
            beta2=beta2,
            weight_decay=weight_decay,
            relative_step=relative_step,
            scale_parameter=scale_parameter
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            eps1 = group["eps1"]
            eps2 = group["eps2"]
            clip_threshold = group["clip_threshold"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:

                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0

                    if grad.ndim >= 2:
                        state["r"] = torch.zeros(
                            grad.shape[:-1], device=param.device
                        )
                        state["c"] = torch.zeros(
                            grad.shape[-1], device=param.device
                        )
                    else:
                        state["v"] = torch.zeros_like(param)

                state["step"] += 1
                t = state["step"]

                beta2_t = 1.0 - t ** group["beta2"]

                if grad.ndim >= 2:

                    r, c = state["r"], state["c"]

                    grad_sq = grad.pow(2).add(eps1)

                    r.mul_(beta2_t).add_(grad_sq.mean(dim=-1), alpha=1 - beta2_t)

                    c.mul_(beta2_t).add_(grad_sq.mean(dim=0), alpha=1 - beta2_t)

                    v_hat = torch.outer(r, c) / r.mean()

                else:

                    v = state["v"]

                    grad_sq = grad.pow(2).add(eps1)

                    v.mul_(beta2_t).add_(grad_sq,alpha=1 - beta2_t)

                    v_hat = v

                update = grad / (v_hat.sqrt().add(eps1))

                # clipping
                update_norm = torch.norm(update)
                clip_denom = torch.clamp(update_norm / clip_threshold,min=1.0)
                update = update / clip_denom

                # learning rate
                if group["relative_step"]:
                    lr_t = min(1e-2, 1.0 / math.sqrt(t))
                else:
                    lr_t = lr

                if group["scale_parameter"]:
                    param_scale = torch.clamp(
                        param.norm(),
                        min=eps2
                    )
                    lr_t = lr_t * param_scale

                if weight_decay != 0:
                    param.add_(param, alpha=-weight_decay)

                param.add_(update, alpha=-lr_t)

        return loss