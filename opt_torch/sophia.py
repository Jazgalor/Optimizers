import torch


class SophiaTorch(torch.optim.Optimizer):

    requires_closure = True

    def __init__(
        self,
        params,
        lr=1e-4,
        beta1=0.965,
        beta2=0.99,
        gamma=0.04,
        weight_decay=0.1,
        eps=1e-15,
        k=10,
    ):

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            gamma=gamma,
            weight_decay=weight_decay,
            eps=eps,
            k=k,
        )

        super().__init__(
            params,
            defaults
        )

        self.step_count = 0


    @torch.no_grad()
    def update_hessian(self):
        """
        EMA Hessian estimator:
        h_t = beta2 * h_(t-1) + (1-beta2) * g_t^2
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


                h = state["h"]


                h.mul_(beta2).addcmul_(
                    p.grad,
                    p.grad,
                    value=1 - beta2
                )


    @torch.no_grad()
    def step(self, closure=None):

        loss = None


        # ---------------- closure ----------------

        if closure is not None:

            with torch.enable_grad():

                loss = closure()



        # ---------------- optimizer step ----------------

        self.step_count += 1


        # ---------------- Hessian update every k steps ----------------

        k = self.defaults["k"]

        if self.step_count % k == 1:

            self.update_hessian()



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



                # ---------------- first moment ----------------

                m.mul_(beta1).add_(
                    p.grad,
                    alpha=1 - beta1
                )



                # ---------------- clipped update ----------------

                denom = torch.maximum(
                    gamma * h,
                    torch.full_like(
                        h,
                        eps
                    )
                )


                ratio = torch.minimum(
                    m.abs() / denom,
                    torch.ones_like(m)
                )


                update = m.sign() * ratio



                # ---------------- weight decay ----------------

                p.mul_(
                    1 - lr * weight_decay
                )



                # ---------------- parameter update ----------------

                p.add_(
                    update,
                    alpha=-lr
                )



        return loss