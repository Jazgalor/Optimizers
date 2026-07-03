import torch


class SophiaTorch(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=1e-4,
        beta1=0.965,
        beta2=0.99,
        gamma=0.04,
        weight_decay=0.1,
        eps=1e-15,
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            gamma=gamma,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

        self.step_count = 0

    @torch.no_grad()
    def update_hessian(self):
        """
        EMA of squared gradients (as in official Sophia-G implementation)
        """
        for group in self.param_groups:
            beta2 = group["beta2"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["h"] = torch.zeros_like(p)

                state["h"].mul_(beta2).addcmul_(
                    p.grad, p.grad, value=1 - beta2
                )

    @torch.no_grad()
    def step(self, closure=None):

        if closure is not None:
            with torch.enable_grad():
                closure()

        self.step_count += 1

        for group in self.param_groups:

            lr = group["lr"]
            beta1 = group["beta1"]
            gamma = group["gamma"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["h"] = torch.zeros_like(p)

                m = state["m"]
                h = state["h"]

                # ---------------- momentum ----------------
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)

                # ---------------- denom ----------------
                denom = torch.maximum(
                    gamma * h,
                    torch.full_like(h, eps),
                )

                # ---------------- ratio ----------------
                ratio = torch.minimum(
                    m.abs() / denom,
                    torch.ones_like(m),
                )

                update = m.sign() * ratio

                # ---------------- weight decay ----------------
                p.mul_(1 - lr * weight_decay)

                # ---------------- parameter update ----------------
                p.add_(update, alpha=-lr)

        return None