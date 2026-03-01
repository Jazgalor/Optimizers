import numpy as np


class Adan:

    def __init__(self,
                 lr=1e-3,
                 beta1=0.02,
                 beta2=0.08,
                 beta3=0.01,
                 eps=1e-8,
                 weight_decay=0.0):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = None
        self.v = None
        self.n = None
        self.prev_g = None

    def step(self, params, grads):

        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.n = [np.zeros_like(p) for p in params]
            self.prev_g = [np.zeros_like(p) for p in params]

        for i in range(len(params)):

            g = grads[i]
            g_prev = self.prev_g[i]
            diff = g - g_prev

            # m_t
            self.m[i] = (1 - self.beta1) * self.m[i] + self.beta1 * g

            # v_t
            self.v[i] = (1 - self.beta2) * self.v[i] + self.beta2 * diff

            # n_t
            combined = g + (1 - self.beta2) * diff
            self.n[i] = (1 - self.beta3) * self.n[i] + self.beta3 * (combined ** 2)

            # eta_t
            eta_t = self.lr / (np.sqrt(self.n[i]) + self.eps)

            # update core
            update = eta_t * (self.m[i] + (1 - self.beta2) * self.v[i])

            # decoupled shrinkage
            shrink = 1.0 / (1.0 + self.weight_decay * self.lr)

            params[i] = shrink * (params[i] - update)

            self.prev_g[i] = g.copy()