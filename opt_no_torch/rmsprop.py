import numpy as np


class RMSProp:

    def __init__(self, lr=1e-3, beta=0.9, eps=1e-8):

        self.lr = lr
        self.beta = beta
        self.eps = eps

        self.state = {}

    def init_state(self, param):

        self.state[id(param)] = {
            "step": 0,
            "square_avg": np.zeros_like(param)
        }

    def compute_update(self, param, grad):

        param_id = id(param)

        if param_id not in self.state:
            self.init_state(param)

        state = self.state[param_id]

        square_avg = state["square_avg"]

        state["step"] += 1

        # EMA of squared gradients
        square_avg *= self.beta
        square_avg += (1 - self.beta) * (grad ** 2)

        update = - self.lr * grad / (np.sqrt(square_avg) + self.eps)

        return update

    def step(self, params, grads):

        for param, grad in zip(params, grads):

            update = self.compute_update(param, grad)

            param += update