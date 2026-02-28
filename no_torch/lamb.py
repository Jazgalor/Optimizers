import numpy as np


class LAMB:

    def __init__(
        self,
        lr=1e-3,
        total_steps=10000,
        beta1=0.9,
        beta2=0.999,
        eps=1e-6,
        weight_decay=0.01,
        power=1.0,
        phi=lambda x: x
    ):

        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")

        self.lr0 = lr
        self.total_steps = total_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.power = power
        self.phi = phi

        self.global_step = 0
        self.state = {}

    def _get_lr(self):
        t = self.global_step
        T = self.total_steps

        if t >= T:
            return 0.0

        return self.lr0 * (1 - t / T) ** self.power

    def step(self, params, grads):

        self.global_step += 1
        lr_t = self._get_lr()

        for i, (x, g) in enumerate(zip(params, grads)):

            if i not in self.state:
                self.state[i] = {
                    "step": 0,
                    "m": np.zeros_like(x),
                    "v": np.zeros_like(x),
                }

            state = self.state[i]
            state["step"] += 1
            t = state["step"]

            m = state["m"]
            v = state["v"]

            # Adam moments
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            # Adam ratio
            r = m_hat / (np.sqrt(v_hat) + self.eps)

            # Add weight decay INSIDE trust ratio
            update = r + self.weight_decay * x

            w_norm = np.linalg.norm(x)
            u_norm = np.linalg.norm(update)

            if w_norm > 0 and u_norm > 0:
                trust_ratio = self.phi(w_norm) / u_norm
            else:
                trust_ratio = 1.0

            # Parameter update
            x -= lr_t * trust_ratio * update

            # Save state
            state["m"] = m
            state["v"] = v