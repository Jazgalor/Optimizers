import torch


class ASAMTorch(torch.optim.Optimizer):

    def __init__(self, base_optimizer, rho=0.05, eps=1e-12):

        self.optimizer = base_optimizer
        self.rho = rho
        self.eps = eps

        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state

    @torch.no_grad()
    def step(self, closure):

        assert closure is not None, "ASAM requires closure"

        # 1. first forward-backward
        with torch.enable_grad():
            loss = closure()

        # 2. compute ||T_w g||
        norm_sq = 0.0

        for group in self.param_groups:
            for param in group["params"]:

                if param.grad is None:
                    continue

                scaled = param.abs() * param.grad
                norm_sq += torch.sum(scaled ** 2)

        norm = torch.sqrt(norm_sq)
        norm = torch.clamp(norm, min=self.eps)

        # 3. perturb parameters

        for group in self.param_groups:
            for param in group["params"]:

                if param.grad is None:
                    continue

                perturbation = (param.abs() ** 2) * param.grad

                eps = self.rho * perturbation / norm

                self.state[param]["eps"] = eps

                param.add_(eps)

        # 4. second forward-backward
        with torch.enable_grad():
            loss = closure()

        # 5. restore parameters

        for group in self.param_groups:
            for param in group["params"]:

                if param.grad is None:
                    continue

                param.sub_(self.state[param]["eps"])

        # 6. update using base optimizer
        self.optimizer.step()

        return loss