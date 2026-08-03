import math

import torch
from torch.optim import Optimizer


# scheduler reset after 1% - 10% max epochs

# weight decay is equal to normalized weight_decay (0.025, 0.05) times sqrt(b/BT) where b - batch size, B - Total training points, T - max epochs

class AdamWTorch(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-2,
        T_i=5,
        T_mult=3,
    ):

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            T_i=T_i,
            T_mult=T_mult,
        )

        super().__init__(params, defaults)

        self.T_i = T_i
        self.T_mult = T_mult
        self.T_cur = 0
        

    def _lr_multiplier(self):
        return 0.5 * (1 + math.cos(math.pi * self.T_cur / self.T_i)
    )

    @torch.no_grad()
    def step(self, closure=None):

        # Scheduler update
        
        schedule = self._lr_multiplier()
        # schedule = 1.0

        self.T_cur += 1

        if self.T_cur >= self.T_i:

            self.T_cur = 0
            self.T_i *= self.T_mult

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:

                if param.grad is None:
                    continue

                grad = param.grad

                state = self.state[param]

                if len(state) == 0:

                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]
                

                # update moments
                exp_avg.mul_(beta1)
                exp_avg.add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2)
                exp_avg_sq.addcmul_(grad, grad, value=1 - beta2)

                # bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                exp_avg_hat = torch.div(exp_avg, bias_correction1)
                exp_avg_sq_hat = torch.div(exp_avg_sq, bias_correction2)

                dw = param.mul(weight_decay)

                update = torch.div(exp_avg_hat.mul(lr), exp_avg_sq_hat.sqrt().add_(eps))

                # Adam update
                param.add_(update, alpha=-schedule).add_(dw, alpha=-schedule)


        return loss