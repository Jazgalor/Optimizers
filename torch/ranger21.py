import torch


class Ranger21Torch(torch.optim.Optimizer):

    def __init__(self,
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
                 t_max=1000,
                 t_warmup=0,
                 t_warmdown=0):

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
            t_warmdown=t_warmdown
        )

        super().__init__(params, defaults)

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

            for p in group["params"]:

                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["m_prev"] = torch.zeros_like(p)
                    state["m_prev2"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["v_max"] = torch.zeros_like(p)
                    state["slow"] = p.clone().detach()

                state["step"] += 1
                t = state["step"]

                g = p.grad

                # ---------------- AGC ----------------
                p_norm = p.norm()
                g_norm = g.norm()

                if g_norm / (p_norm.clamp_min(eps_c)) > tau:
                    g = g * (tau * p_norm.clamp_min(eps_c) / (g_norm + 1e-12))

                # ---------------- GC ----------------
                g = g - g.mean()

                # ---------------- Momentum ----------------
                m_prev = state["m_prev"]
                m_prev2 = state["m_prev2"]

                m_new = (beta1**2) * m_prev2 + (1 - beta1**2) * g

                m_hat = ((1 + beta0) * m_new - beta0 * m_prev) / (1 - beta1**t)

                state["m_prev2"] = m_prev.clone()
                state["m_prev"] = state["m"].clone()
                state["m"] = m_new.clone()

                # ---------------- Variance ----------------
                v_new = beta2 * state["v"] + (1 - beta2) * (g ** 2)
                v_max = torch.maximum(state["v_max"], v_new)

                v_hat = v_max / (1 - beta2**t)

                state["v"] = v_new
                state["v_max"] = v_max

                denom_scale = torch.sqrt((1 + beta0)**2 + beta0**2)

                update = m_hat / (denom_scale * (torch.sqrt(v_hat) + eps))

                # ---------------- LR schedule ----------------
                warmup = t / max(1, group["t_warmup"])
                cooldown = (group["t_max"] - t) / max(1, group["t_warmdown"])
                explore = (1 - beta2) * t / 2

                scale = min(1.0, max(warmup, explore), cooldown)
                lr_t = scale * lr

                # ---------------- WD ----------------
                decay = wd * (1 - 1 / (p.norm() + eps))
                decay_term = (lr_t / (torch.mean(v_hat).sqrt() + eps)) * decay * p

                # ---------------- Update ----------------
                p.add_(update * lr_t + decay_term * lr_t)

                # ---------------- Lookahead ----------------
                if t % k == 0:
                    slow = state["slow"]
                    slow.mul_(beta_l).add_(p, alpha=(1 - beta_l))
                    p.copy_(slow)

                
        return loss