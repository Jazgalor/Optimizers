import torch
from torch.optim import Optimizer


class LookaheadTorch(Optimizer):

    def __init__(self, optimizer, lr=0.5, k=5):

        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

        self.lr = lr
        self.k = k
        self.step_counter = 0

        super().__init__(self.param_groups, dict(lr=lr, k=k))

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["slow_param"] = p.clone().detach()

    @torch.no_grad()
    def step(self, closure=None):

        loss = self.optimizer.step(closure)

        self.step_counter += 1

        if self.step_counter % self.k != 0:
            return loss

        for group in self.param_groups:

            lr = group["lr"]

            for param in group["params"]:

                if param.grad is None:
                    continue

                state = self.state[param]

                slow = state["slow_param"]

                slow.add_(param - slow, alpha=lr)

                param.copy_(slow)

        return loss

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)