import torch
from torch.optim import Optimizer
import math


class RAdamTorch(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps
        )

        super().__init__(params, defaults)

        self.rho_inf = 2.0 / (1.0 - beta2) - 1.0

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            eps = group["eps"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]

            for param in group["params"]:

                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                state["step"] += 1
                t = state["step"]

                m = state["m"]
                v = state["v"]

                # first moment
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # second moment
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                beta2_t = beta2 ** t
                rho_t = self.rho_inf - (2 * t * beta2_t) / (1 - beta2_t)

                if rho_t > 4:

                    r_t = math.sqrt(((rho_t - 4) * (rho_t - 2) * self.rho_inf) / ((self.rho_inf - 4) * (self.rho_inf - 2) * rho_t))

                    denom = v_hat.sqrt().add_(eps)
                    param.addcdiv_(m_hat, denom, value=-lr * r_t)

                else:
                    param.add_(m_hat, alpha=-lr)

        return loss