import numpy as np


class AdaBelief:

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):

        if not 0 < beta1 < 1:
            raise ValueError("beta1 must be in (0,1)")
        if not 0 < beta2 < 1:
            raise ValueError("beta2 must be in (0,1)")

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.m = None
        self.s = None

    def step(self, params, grads):

        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.s = [np.zeros_like(p) for p in params]

        self.t += 1

        for i in range(len(params)):

            g = grads[i]

            # First moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

            # Belief term (variance of gradient)
            diff = g - self.m[i]
            self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * (diff ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            s_hat = self.s[i] / (1 - self.beta2 ** self.t)

            # Parameter update
            params[i] -= self.lr * m_hat / (np.sqrt(s_hat) + self.eps)