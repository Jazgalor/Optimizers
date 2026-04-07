import torch


class SAMTorch(torch.optim.Optimizer):

    def __init__(self, base_optimizer, rho=0.05):

        self.optimizer = base_optimizer
        self.rho = rho

        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state

    @torch.no_grad()
    def step(self, closure):

        assert closure is not None, "SAM requires closure"

        loss = closure()
        
        grad_norm = torch.norm(
            torch.stack([
                param.grad.norm()
                for group in self.param_groups
                for param in group["params"]
                if param.grad is not None
            ])
        )

        if grad_norm == 0:
            grad_norm = torch.tensor(1e-12, device=grad_norm.device)

        eps_list = []

        for group in self.param_groups:
            for param in group["params"]:

                if param.grad is None:
                    eps_list.append(None)
                    continue

                eps = self.rho * param.grad / grad_norm
                param.add_(eps)
                eps_list.append(eps)

        closure()

        idx = 0
        for group in self.param_groups:
            for param in group["params"]:

                eps = eps_list[idx]
                idx += 1

                if eps is not None:
                    param.sub_(eps)

        self.optimizer.step()

        return loss