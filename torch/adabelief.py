import torch
from torch.optim import Optimizer


class AdaBeliefTorch(Optimizer):

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']

            for param in group['params']:

                if param.grad is None:
                    continue

                grad = param.grad

                state = self.state[param]

                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(param)
                    state['s'] = torch.zeros_like(param)

                m = state['m']
                s = state['s']

                state['t'] += 1
                t = state['t']

                # First moment
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Belief variance
                diff = grad - m
                s.mul_(beta2).addcmul_(diff, diff, value=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1 ** t)
                s_hat = s / (1 - beta2 ** t)

                update = s_hat.sqrt().add_(eps)

                # Update
                param.addcdiv_(m_hat, update, value=-lr)