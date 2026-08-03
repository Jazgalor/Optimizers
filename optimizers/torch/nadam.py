import torch
from torch.optim import Optimizer


class NadamTorch(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,
        beta1=0.99,
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
                    state["beta1_product"] = 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                beta1_t = beta1 * (1.0 - 0.5 * 0.96 ** (step / 250))

                beta1_next = beta1 * (1.0 - 0.5 * 0.96 ** ((step + 1) / 250))

                state["beta1_product"] *= beta1_t

                beta1_product = state["beta1_product"]

                # update moments
                exp_avg.mul_(beta1)
                exp_avg.add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2)
                exp_avg_sq.addcmul_(grad, grad, value=1 - beta2)

                # bias correction
                bias_correction1 = 1 - beta1_product * beta1_next
                bias_correction2 = 1 - beta2 ** step
                grad_correction = 1 - beta1_product

                exp_avg_hat = torch.div(exp_avg, bias_correction1)
                exp_avg_sq_hat = torch.div(exp_avg_sq, bias_correction2)

                # grad correction
                grad_hat = torch.div(grad, grad_correction)

                # Nesterov correction
                exp_avg_nesterov = grad_hat.mul(1 - beta1_t).add(exp_avg_hat, alpha=beta1_next)

                update = exp_avg_sq_hat.sqrt().add_(eps)

                param.addcdiv_(exp_avg_nesterov, update, value=-lr)

        return loss