import numpy as np


class AMSGrad:

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

    def compute_update(self, param, grad, state):

        if "m" not in state:
            state["m"] = np.zeros_like(param)
            state["v"] = np.zeros_like(param)
            state["v_hat_max"] = np.zeros_like(param)

        m = state["m"]
        v = state["v"]
        v_hat_max = state["v_hat_max"]

        # first moment
        m = self.beta1 * m + (1 - self.beta1) * grad

        # second moment
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

        # bias correction
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)

        # AMSGrad max trick
        v_hat_max = np.maximum(v_hat_max, v_hat)

        update = self.lr * m_hat / (np.sqrt(v_hat_max) + self.eps)

        state["m"] = m
        state["v"] = v
        state["v_hat_max"] = v_hat_max

        return update

    def step(self, params, grads):

        self.t += 1

        for i, (param, grad) in enumerate(zip(params, grads)):

            if i not in self.state:
                self.state[i] = {}

            update = self.compute_update(param, grad, self.state[i])
            param -= update