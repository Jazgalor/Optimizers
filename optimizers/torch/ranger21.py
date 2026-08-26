import torch
from torch.optim import Optimizer


class Ranger21Torch(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=1e-4,
        beta0=0.9,
        beta1=0.9,
        beta2=0.999,
        beta_lookahead=0.5,
        eps=1e-8,
        eps_clipping=1e-3,
        tau_clipping=1e-2,
        k_lookahead=5,
        t_max=70400,
        t_warmup=None,
        t_warmdown=None,
    ):

        if t_warmup is None:
            t_warmup = int(0.22 * t_max)

        if t_warmdown is None:
            t_warmdown = int(0.28 * t_max)

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta0=beta0,
            beta1=beta1,
            beta2=beta2,
            beta_lookahead=beta_lookahead,
            eps=eps,
            eps_clipping=eps_clipping,
            tau_clipping=tau_clipping,
            k_lookahead=k_lookahead,
            t_max=t_max,
            t_warmup=t_warmup,
            t_warmdown=t_warmdown,
        )

        super().__init__(params, defaults)

    @staticmethod
    def adaptive_gradient_clipping(
        grad,
        param,
        tau,
        eps,
    ):

        if grad.ndim <= 1:
            return grad

        if grad.ndim == 2:

            reduce_dims = 1

        elif grad.ndim == 4:

            reduce_dims = (1, 2, 3)

        else:

            return grad

        param_norm = torch.linalg.vector_norm(
            param,
            dim=reduce_dims,
            keepdim=True,
        )

        grad_norm = torch.linalg.vector_norm(
            grad,
            dim=reduce_dims,
            keepdim=True,
        )

        max_norm = torch.clamp(
            param_norm,
            min=eps,
        )

        scale = tau * max_norm / (grad_norm + 1e-8)

        grad = torch.where(
            grad_norm > tau * max_norm,
            grad * scale,
            grad,
        )

        return grad

    @staticmethod
    def gradient_centralization(grad):

        if grad.ndim == 2:

            grad = grad - grad.mean(
                dim=1,
                keepdim=True,
            )

        elif grad.ndim == 4:

            grad = grad - grad.mean(
                dim=(1, 2, 3),
                keepdim=True,
            )

        return grad

    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            wd = group["weight_decay"]
            beta0 = group["beta0"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            beta_l = group["beta_lookahead"]
            eps = group["eps"]
            eps_c = group["eps_clipping"]
            tau = group["tau_clipping"]
            k = group["k_lookahead"]
            t_max = group["t_max"]
            t_warmup = group["t_warmup"]
            t_warmdown = group["t_warmdown"]

            for param in group["params"]:

                if param.grad is None:
                    continue

                state = self.state[param]

                if len(state) == 0:

                    state["step"] = 0

                    state["m_prev"] = torch.zeros_like(param)
                    state["m_prev2"] = torch.zeros_like(param)

                    state["v"] = torch.zeros_like(param)
                    state["v_max"] = torch.zeros_like(param)

                    state["slow"] = param.clone().detach()

                state["step"] += 1
                t = state["step"]

                grad = param.grad

                grad = self.adaptive_gradient_clipping(grad, param, tau, eps_c, )

                grad = self.gradient_centralization(grad)

                m_prev = state["m_prev"]
                m_prev2 = state["m_prev2"]

                m = m_prev2.mul(beta1**2).add(grad, alpha=1 - beta1**2)
                bias_correction = 1 / (1 - beta1**t)
                m_hat = m.mul(1 + beta0).add(m_prev, alpha= -beta0).mul(bias_correction)

                state["m_prev2"].copy_(m_prev)
                state["m_prev"].copy_(m)

                v = state["v"]
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                v_max = torch.maximum(state["v_max"], v)

                v_hat = torch.div(v_max,  (1 - beta2**t))

                state["v"].copy_(v)
                state["v_max"].copy_(v_max)

                denom = (((1 + beta0)**2 + beta0**2) ** 0.5)

                u = torch.div(m_hat, (denom * (v_hat.sqrt() + eps)))

                warmup = max(((1-beta2)/2) * t, t / max(1, t_warmup))

                warmdown = (t_max - t) / max(1, t_warmdown)

                schedule = min(1.0, warmup, warmdown)

                lr_t = lr * schedule

                variance = torch.sqrt(v_hat.mean()) + eps

                correction = 1.0 - 1.0 / (torch.norm(param)+ eps)

                decay = torch.div(lr_t, variance).mul(wd * correction).mul(param)

                param.add_(u, alpha=-lr_t,)

                param.add_(decay, alpha=-lr_t,)

                if t % k == 0:
                    slow = state["slow"]
                    slow.mul_(beta_l)
                    slow.add_(param, alpha=1 - beta_l,)
                    param.copy_(slow)

        return loss