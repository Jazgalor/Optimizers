import torch


class SGDNesterovTorch(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, beta=0.9):

        defaults = dict(lr=lr, beta=beta)

        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is None:
            raise RuntimeError("SGDNesterovTorch requires a closure.")
        
        for group in self.param_groups:

            beta = group["beta"]

            for param in group["params"]:

                state = self.state[param]

                if len(state) == 0:

                    state["step"] = 0
                    state["m"] = ( torch.zeros_like(param) )

                m = state["m"]
                param.add_(m, alpha=beta)

        with torch.enable_grad():

            loss = closure()        

        for group in self.param_groups:

            lr = group["lr"]
            beta = group["beta"]

            for param in group["params"]:

                if param.grad is None:
                    continue

                grad = param.grad

                state = self.state[param]

                state["step"] += 1

                m = state["m"]

                # m = beta * m + grad * -lr
                m.mul_(beta).add_(grad, alpha=-lr)

                # update = grad
                update = grad

                param.add_(update, alpha=-lr)

        return loss