import torch
from torch.optim import Optimizer


class AdamaxTorch(Optimizer):

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
                    state["exp_inf"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_inf = state["exp_inf"]

                state["step"] += 1
                step = state["step"]

                # first moment
                exp_avg.mul_(beta1)
                exp_avg.add_(grad, alpha=1 - beta1)

                # infinity norm
                exp_inf.mul_(beta2)
                torch.maximum(exp_inf, grad.abs(), out=exp_inf)

                # bias correction
                bias_correction = 1 - beta1 ** step

                exp_avg_hat = exp_avg / bias_correction

                # update
                param.addcdiv_(exp_avg_hat, exp_inf.add_(eps), value=-lr)

        return loss