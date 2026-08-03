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
        batch_size=128,
    ):

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            gamma=gamma,
            weight_decay=weight_decay,
            eps=eps,
            k=k,
            batch_size=batch_size,
        )

        super().__init__(
            params,
            defaults
        )

        self.step_count = 0


    @torch.no_grad()
    def update_hessian(self, closure):

        with torch.enable_grad():
            closure("gnb")

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

                batch_size = group["batch_size"]

                h.mul_(beta2).addcmul_(
                    p.grad,
                    p.grad,
                    value=(1 - beta2) * batch_size
                )


    @torch.no_grad()
    def step(self, closure=None):

        assert closure is not None, "Sophia requires closure"


        # ---------------- closure ----------------

        with torch.enable_grad():
            loss = closure()
            


        # ---------------- gradient ----------------
        saved_grad = {}

        for group in self.param_groups:
            for p in group["params"]:

                if p.grad is not None:
                    saved_grad[p] = p.grad.clone()



        # ---------------- optimizer step ----------------

        self.step_count += 1


        # ---------------- Hessian update every k steps ----------------

        k = group["k"]

        if self.step_count % k == 1:

            self.update_hessian(closure)

            for p, grad in saved_grad.items():
                p.grad.copy_(grad)



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

                m.mul_(beta1).add_(p.grad, alpha=1-beta1)



                # ---------------- clipped update ----------------

                denom = torch.maximum(
                    gamma * h,
                    torch.full_like(h, eps)
                )

                ratio = torch.minimum(
                    m.abs() / denom,
                    torch.ones_like(m)
                )

                update = m.sign() * ratio



                # ---------------- weight decay ----------------

                p.mul_(1 - lr * weight_decay)



                # ---------------- parameter update ----------------

                p.add_(update, alpha=-lr)



        return loss