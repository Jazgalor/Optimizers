import torch
from torch.optim import Optimizer
import math


class RAdamTorch(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-2,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )

        super().__init__(params, defaults)

        self.rho_inf = (2.0 / (1.0 - beta2) - 1.0)


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


                state["step"] += 1

                t = state["step"]

                m = state["m"]

                v = state["v"]


                # First moment:
                # m_t = beta1 * m_(t-1) + (1 - beta1) * g_t

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1,)


                # Second moment:
                # v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2

                v.mul_(1 / beta2).addcmul_(grad,grad, value=1.0 - beta2,)


                # Bias correction of first moment:
                # m_hat_t = m_t / (1 - beta1^t)

                m_hat = m / (1.0 - beta1 ** t)


                # Compute rho_t

                beta2_t = beta2 ** t

                rho_t = (self.rho_inf - (2.0 * t * beta2_t / (1.0 - beta2_t)))


                if rho_t > 4:

                    # Variance rectification term

                    r_t = math.sqrt(
                        (
                            (rho_t - 4.0)
                            * (rho_t - 2.0)
                            * self.rho_inf
                        )
                        /
                        (
                            (self.rho_inf - 4.0)
                            * (self.rho_inf - 2.0)
                            * rho_t
                        )
                    )


                    # Adaptive learning rate: l_t = sqrt(1 - beta2^t) / sqrt(v_t)

                    adaptive_lr = (math.sqrt(1.0 - beta2_t / v.add(eps)))


                    # theta_t = theta_(t-1) - alpha_t * r_t * l_t * m_hat_t

                    update = (m_hat * adaptive_lr)

                    param.add_(update, alpha=-lr * r_t,)

                else:

                    # Un-adapted momentum:
                    # theta_t = theta_(t-1) - alpha_t * m_hat_t

                    param.add_(m_hat, alpha=-lr,)

        return loss