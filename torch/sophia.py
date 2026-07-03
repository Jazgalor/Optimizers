import torch


class SophiaTorch(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=1e-4,
        beta1=0.965,
        beta2=0.99,
        rho=0.04,
        weight_decay=0.1,
        eps=1e-12,
        hessian_update_period=10,
    ):

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            rho=rho,
            weight_decay=weight_decay,
            eps=eps,
            hessian_update_period=hessian_update_period,
        )

        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if (
            closure is not None
            and self.state.setdefault("global_step", 0)
            % self.param_groups[0]["hessian_update_period"]
            == 0
        ):

            loss, hessian = closure()

        elif closure is not None:

            loss = closure()

            hessian = None

        self.state["global_step"] = self.state.get("global_step", 0) + 1

        hessian_idx = 0

        for group in self.param_groups:

            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            rho = group["rho"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for param in group["params"]:

                if param.grad is None:
                    continue

                grad = param.grad

                state = self.state[param]

                if len(state) == 0:

                    state["m"] = torch.zeros_like(param)
                    state["h"] = torch.zeros_like(param)

                if hessian is not None:

                    state["h"].mul_(beta2).add_(
                        hessian[hessian_idx],
                        alpha=(1.0 - beta2),
                    )

                m = state["m"]
                h = state["h"]

                m.mul_(beta1).add_(
                    grad,
                    alpha=(1.0 - beta1),
                )

                update = torch.clamp(
                    m / (rho * h + eps),
                    min=-1.0,
                    max=1.0,
                )

                update.add_(param, alpha=weight_decay)

                param.add_(update, alpha=-lr)

                hessian_idx += 1

        return loss