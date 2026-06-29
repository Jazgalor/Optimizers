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
        eps_list = []

        for group in self.param_groups:
            for param in group["params"]:

                if param.grad is None:
                    eps_list.append(None)
                    continue

                perturbation = (param.abs() ** 2) * param.grad

                eps = self.rho * perturbation / norm

                param.add_(eps)
                eps_list.append(eps)

        # 4. second forward-backward
        closure()

        # 5. restore parameters
        idx = 0

        for group in self.param_groups:
            for param in group["params"]:

                eps = eps_list[idx]
                idx += 1

                if eps is not None:
                    param.sub_(eps)

        # 6. update using base optimizer
        self.optimizer.step()

        return loss