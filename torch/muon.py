import torch


class MuonTorch(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=5,
        eps=1e-7,
    ):

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
        )

        super().__init__(params, defaults)

    @staticmethod
    def newton_schulz5(matrix, steps=5, eps=1e-7):

        if matrix.ndim != 2:
            raise ValueError("Newton-Schulz requires a 2D matrix.")

        a = 3.4445
        b = -4.7750
        c = 2.0315

        X = matrix.clone()

        transpose = False

        if X.size(0) > X.size(1):
            X = X.t()
            transpose = True

        X /= X.norm() + eps

        for _ in range(steps):

            A = X @ X.t()
            B = b * A + c * (A @ A)

            X = a * X + B @ X

        if transpose:
            X = X.t()

        return X

    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for param in group["params"]:

                if param.grad is None:
                    continue

                grad = param.grad

                state = self.state[param]

                if len(state) == 0:

                    state["m"] = torch.zeros_like(param)

                m = state["m"]

                # ---------------- Momentum ----------------

                m.mul_(momentum)
                m.add_(grad)

                # ---------------- Orthogonalization ----------------

                if m.ndim == 1:

                    update = m.clone()

                elif m.ndim == 2:

                    update = self.newton_schulz5(
                        m,
                        steps=ns_steps,
                        eps=eps,
                    )

                elif m.ndim == 4:

                    original_shape = m.shape

                    matrix = m.reshape(m.shape[0], -1)

                    update = self.newton_schulz5(
                        matrix,
                        steps=ns_steps,
                        eps=eps,
                    )

                    update = update.reshape(original_shape)

                else:

                    raise ValueError(
                        f"Unsupported parameter dimension ({m.ndim}) for Muon."
                    )

                # ---------------- Weight decay ----------------

                if weight_decay != 0:

                    param.mul_(1 - lr * weight_decay)

                # ---------------- Parameter update ----------------

                param.add_(update, alpha=-lr)

        return loss