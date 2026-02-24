import numpy as np


class Adagrad:

    def __init__(self, lr=1e-2, eps=1e-10):

        self.lr = lr
        self.eps = eps
        self.step = 0
        self.state = {}

    def init_state(self, param):

        return {
            "s": np.zeros_like(param)
        }

    def compute_update(self, param, grad, state):

        s = state["s"]

        # accumulate squared gradients
        s += grad * grad

        # compute adaptive step
        update = grad / (np.sqrt(s) + self.eps)

        return update

    def apply_update(self, param, update):

        param -= self.lr * update

    def step_param(self, param, grad):

        param_id = id(param)

        if param_id not in self.state:
            self.state[param_id] = self.init_state(param)

        state = self.state[param_id]

        update = self.compute_update(param, grad, state)

        self.apply_update(param, update)

    def step(self, params, grads):

        self.step += 1

        for param, grad in zip(params, grads):
            self.step_param(param, grad)