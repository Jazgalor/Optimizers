import torch
from torch.optim import Optimizer


class AdamTorch(Optimizer):

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

    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]

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
                step = state["step"]

                # update moments
                exp_avg.mul_(beta1)
                exp_avg.add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2)
                exp_avg_sq.addcmul_(grad, grad, value=1 - beta2)

                # bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                # update
                update = exp_avg_sq_hat.sqrt().add_(eps)

                param.addcdiv_(exp_avg_hat, update, value=-lr)

        return loss