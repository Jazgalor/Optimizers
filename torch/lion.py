import torch


class LionTorch(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=1e-1):

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay
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
            weight_decay = group["weight_decay"]

            for param in group["params"]:

                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["m"] = torch.zeros_like(param)

                m = state["m"]

                # evolved sign direction
                blended = beta1 * m + (1 - beta1) * grad
                update = blended.sign()

                # weight decay
                if weight_decay != 0:
                    update = update + weight_decay * param

                # momentum update
                m.mul_(beta2).add_(grad, alpha=1 - beta2)

                # parameter update
                param.add_(update, alpha=-lr)

        return loss