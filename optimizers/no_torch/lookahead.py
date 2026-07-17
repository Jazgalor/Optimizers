import numpy as np


class Lookahead:

    def __init__(self, optimizer, k=5, lr=0.5):

        self.optimizer = optimizer
        self.k = k
        self.lr = lr

        self.step_counter = 0
        self.slow_weights = None


    def init_slow(self, params):

        self.slow_weights = [p.copy() for p in params]


    def step(self, params, grads):

        if self.slow_weights is None:
            self.init_slow(params)

        # inner optimizer step
        self.optimizer.step(params, grads)

        self.step_counter += 1

        # synchronization
        if self.step_counter % self.k == 0:

            for i, param in enumerate(params):

                slow = self.slow_weights[i]

                slow += self.lr * (param - slow)

                param[:] = slow