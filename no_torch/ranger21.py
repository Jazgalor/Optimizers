import numpy as np


class Ranger21:

    def __init__(self,
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

        self.lr = lr
        self.weight_decay = weight_decay

        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta_lookahead = beta_lookahead

        self.eps = eps
        self.eps_clipping = eps_clipping
        self.tau_clipping = tau_clipping

        self.k_lookahead = k_lookahead
        self.t_max = t_max
        self.t_warmup = max(1, t_warmup)
        self.t_warmdown = max(1, t_warmdown)

        self.step = 0
        self.state = {}


    def init_state(self, param):

        return {
            "m": np.zeros_like(param),
            "m_prev": np.zeros_like(param),
            "m_prev2": np.zeros_like(param),
            "v": np.zeros_like(param),
            "v_max": np.zeros_like(param),
            "slow": param.copy()
        }


    def compute_update(self, param, grad, state):

        self.step += 1
        t = self.step

        m = state["m"]
        m_prev = state["m_prev"]
        m_prev2 = state["m_prev2"]
        v = state["v"]
        v_max = state["v_max"]
        slow = state["slow"]

        g = grad.copy()

        # -------------------------
        # 1. Adaptive Gradient Clipping (row-wise simplified)
        # -------------------------
        param_norm = np.linalg.norm(param)
        grad_norm = np.linalg.norm(g)

        denom = max(param_norm, self.eps_clipping)

        if grad_norm / denom > self.tau_clipping:
            g *= (self.tau_clipping * denom / (grad_norm + 1e-12))

        # -------------------------
        # 2. Gradient centralization
        # -------------------------
        g = g - np.mean(g)

        # -------------------------
        # 3. Momentum (Ranger21-style 2-step)
        # -------------------------
        m_new = (self.beta1**2) * m_prev2 + (1 - self.beta1**2) * g

        m_hat = ((1 + self.beta0) * m_new - self.beta0 * m_prev) / (1 - self.beta1**t)

        # shift history
        state["m_prev2"] = m_prev.copy()
        state["m_prev"] = m.copy()
        state["m"] = m_new.copy()

        # -------------------------
        # 4. Variance (AMSGrad style)
        # -------------------------
        v_new = self.beta2 * v + (1 - self.beta2) * (g ** 2)
        v_max = np.maximum(v_max, v_new)

        v_hat = v_max / (1 - self.beta2**t)

        state["v"] = v_new
        state["v_max"] = v_max

        # -------------------------
        # 5. Update vector
        # -------------------------
        denom_scale = np.sqrt((1 + self.beta0)**2 + self.beta0**2)

        update = m_hat / (denom_scale * (np.sqrt(v_hat) + self.eps))

        # -------------------------
        # 6. LR schedule
        # -------------------------
        warmup = t / self.t_warmup if self.t_warmup > 0 else 1.0
        cooldown = (self.t_max - t) / self.t_warmdown if self.t_warmdown > 0 else 1.0
        explore = (1 - self.beta2) * t / 2

        scale = min(1.0, max(warmup, explore), cooldown)
        lr_t = scale * self.lr

        # -------------------------
        # 7. Weight decay (stable WD)
        # -------------------------
        decay = self.weight_decay * (1 - 1 / (np.linalg.norm(param) + self.eps))
        decay_term = (lr_t / (np.sqrt(np.mean(v_hat)) + self.eps)) * decay * param

        # -------------------------
        # 8. final update
        # -------------------------
        update = lr_t * update + lr_t * decay_term

        # -------------------------
        # 9. Lookahead
        # -------------------------
        if t % self.k_lookahead == 0:
            slow = self.beta_lookahead * slow + (1 - self.beta_lookahead) * param
            param[:] = slow

        state["slow"] = slow

        return update


    def apply_update(self, param, update):
        param -= update


    def step_param(self, param, grad):

        pid = id(param)

        if pid not in self.state:
            self.state[pid] = self.init_state(param)

        state = self.state[pid]

        update = self.compute_update(param, grad, state)

        self.apply_update(param, update)


    def step(self, params, grads):

        for p, g in zip(params, grads):
            self.step_param(p, g)