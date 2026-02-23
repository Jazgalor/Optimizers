import torch


class SGDTorch(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):

        defaults = dict(lr=lr)

        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]

            for param in group["params"]:

                if param.grad is None:
                    continue

                grad = param.grad

                state = self.state[param]

                if len(state) == 0:

                    state["step"] = 0

                state["step"] += 1

                update = grad

                param.add_(update, alpha=-lr)

        return loss