import numpy as np
import math


class RAdam:

    def __init__(
        self,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.state = {}
        self.t = 0

        self.rho_inf = 2.0 / (1.0 - beta2) - 1.0

    def compute_update(self, param, grad, state):

        if "m" not in state:
            state["m"] = np.zeros_like(param)
            state["v"] = np.zeros_like(param)

        m = state["m"]
        v = state["v"]

        # moment updates
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

        # bias correction
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)

        # rho_t
        beta2_t = self.beta2 ** self.t
        rho_t = self.rho_inf - (2 * self.t * beta2_t) / (1 - beta2_t)

        if rho_t > 4:
            r_t = math.sqrt(
                ((rho_t - 4) * (rho_t - 2) * self.rho_inf) /
                ((self.rho_inf - 4) * (self.rho_inf - 2) * rho_t)
            )

            update = self.lr * r_t * m_hat / (np.sqrt(v_hat) + self.eps)
        else:
            update = self.lr * m_hat

        state["m"] = m
        state["v"] = v

        return update

    def step(self, params, grads):

        self.t += 1

        for i, (param, grad) in enumerate(zip(params, grads)):

            if i not in self.state:
                self.state[i] = {}

            update = self.compute_update(param, grad, self.state[i])
            param -= update