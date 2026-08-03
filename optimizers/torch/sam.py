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

        with torch.enable_grad():
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

        # Compute perturbation ε(w)
        for group in self.param_groups:
            for param in group["params"]:

                if param.grad is None:
                    continue

                eps = self.rho * param.grad / grad_norm

                self.state[param]["eps"] = eps

                param.add_(eps)

        # Gradient at w + ε
        with torch.enable_grad():
            closure()

        # Restore original weights
        for group in self.param_groups:
            for param in group["params"]:

                if param.grad is None:
                    continue

                param.sub_(self.state[param]["eps"])

        # Update using gradient computed at w + ε
        self.optimizer.step()

        return loss