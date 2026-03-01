import torch
from torch.optim import Optimizer


class LAMB(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,
        total_steps=10000,
        beta1=0.9,
        beta2=0.999,
        eps=1e-6,
        weight_decay=0.01,
        power=1.0,  # polynomial power
        phi=lambda x: x  # scaling function
    ):

        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")

        defaults = dict(
            lr=lr,
            total_steps=total_steps,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            power=power
        )

        super().__init__(params, defaults)
        self.phi = phi
        self.global_step = 0

    def _get_lr(self, group):
        t = self.global_step
        T = group["total_steps"]
        base_lr = group["lr"]
        power = group["power"]

        if t >= T:
            return 0.0

        return base_lr * (1 - t / T) ** power

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        for group in self.param_groups:

            lr_t = self._get_lr(group)

            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

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

                # Adam moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Adam ratio
                r = m_hat / (v_hat.sqrt().add(eps))

                # Add weight decay inside trust ratio
                update = r + weight_decay * param

                w_norm = torch.norm(param)
                u_norm = torch.norm(update) 

                if w_norm > 0 and u_norm > 0:
                    trust_ratio = self.phi(w_norm) / u_norm
                else:
                    trust_ratio = 1.0

                param.add_(update, alpha=-lr_t * trust_ratio)

        return loss