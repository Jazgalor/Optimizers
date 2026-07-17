import torch
from torch.optim import Optimizer


class AdadeltaTorch(Optimizer):

    def __init__(self, params, beta=0.9, eps=1e-6):

        defaults = dict(
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
                    state["acc_delta"] = torch.zeros_like(param)

                square_avg = state["square_avg"]
                acc_delta = state["acc_delta"]

                state["step"] += 1

                # square_avg update
                square_avg.mul_(beta)
                square_avg.addcmul_(grad, grad, value=1 - beta)

                # compute update
                rms_delta = acc_delta.sqrt().add_(eps)
                rms_grad = square_avg.sqrt().add_(eps)

                update = rms_delta.div_(rms_grad).mul_(grad).neg_()

                # acc_delta update
                acc_delta.mul_(beta)
                acc_delta.addcmul_(update, update, value=1 - beta)

                # apply update
                param.add_(update)

        return loss