import torch
from torch.optim import Optimizer

#In the original paper there was no defined bias correction so for this implementation it was omitted
class AdanTorch(Optimizer):

    def __init__(self,
                 params,
                 lr=1e-3,
                 beta1=0.02,
                 beta2=0.08,
                 beta3=0.01,
                 eps=1e-8,
                 weight_decay=0.02):

        defaults = dict(lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        beta3=beta3,
                        eps=eps,
                        weight_decay=weight_decay)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:

            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            beta3 = group['beta3']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for param in group['params']:

                if param.grad is None:
                    continue

                grad = param.grad

                state = self.state[param]

                if len(state) == 0:

                    state['m'] = torch.zeros_like(param)
                    state['v'] = torch.zeros_like(param)
                    state['n'] = torch.zeros_like(param)
                    state['prev_grad'] = torch.zeros_like(param)

                m = state['m']
                v = state['v']
                n = state['n']
                prev_grad = state['prev_grad']


                diff = grad - prev_grad

                # m_t
                m.mul_(1 - beta1).add_(grad, alpha=beta1)

                # v_t
                v.mul_(1 - beta2).add_(diff, alpha=beta2)

                # n_t
                combined = grad + (1 - beta2) * diff
                n.mul_(1 - beta3).addcmul_(combined, combined, value=beta3)

                # eta_t
                eta_t = lr / (n.sqrt().add(eps))

                update = eta_t * (m + (1 - beta2) * v)

                shrink = 1.0 / (1.0 + weight_decay * lr)

                param.mul_(shrink).add_(-update)

                prev_grad.copy_(grad)
        
        return loss