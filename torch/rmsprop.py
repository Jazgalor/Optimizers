import torch
from torch.optim import Optimizer


class RMSPropTorch(Optimizer):

    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):

        defaults = dict(
            lr=lr,
            beta=beta,
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
            beta = group["beta"]
            eps = group["eps"]

            for param in group["params"]:

                if param.grad is None:
                    continue

                grad = param.grad

                state = self.state[param]

                if len(state) == 0:

                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(param)

                square_avg = state["square_avg"]

                state["step"] += 1

                # square_avg = beta * square_avg + (1 - beta) * grad^2
                square_avg.mul_(beta)
                square_avg.addcmul_(grad, grad, value=1 - beta)

                # param -= lr * grad / (sqrt(square_avg) + eps)
                update = square_avg.sqrt().add_(eps)

                param.addcdiv_(grad, update, value=-lr)

        return loss