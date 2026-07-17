import torch
from torch.optim import Optimizer


class AMSGradTorch(Optimizer):

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
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)
                    state["v_hat_max"] = torch.zeros_like(param)

                state["step"] += 1
                t = state["step"]

                m = state["m"]
                v = state["v"]
                v_hat_max = state["v_hat_max"]

                # first moment
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # second moment
                v.mul_(beta2).addcmul_(grad,grad,value=1 - beta2)

                # bias correction
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # AMSGrad max trick
                torch.maximum(v_hat_max, v_hat, out=v_hat_max)

                update = v_hat_max.sqrt().add_(eps)

                param.addcdiv_(m_hat, update, value=-lr)

        return loss