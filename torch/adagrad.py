import torch
from torch.optim import Optimizer
import torch.optim.optimizer


class TorchAdagrad(Optimizer):

    def __init__(self, params, lr=1e-2, eps=1e-10):

        defaults = dict(
            lr=lr,
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
            eps = group["eps"]

            for param in group["params"]:

                if param.grad is None:
                    continue

                grad = param.grad

                state = self.state[param]

                # init state
                if len(state) == 0:

                    state["step"] = 0

                    state["s"] = torch.zeros_like(param.data)

                s = state["s"]

                state["step"] += 1

                # accumulate squared gradients
                s.addcmul_(grad, grad)

                update = s.sqrt().add(eps)

                # compute update
                param.addcdiv_(grad, update, value=-lr)

        return loss