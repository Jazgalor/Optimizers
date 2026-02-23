import numpy as np


class SGD:

    def __init__(self, lr=1e-3):

        self.lr = lr
        self.step = 0
        self.state = {}


    def init_state(self, param):

        return {}


    def compute_update(self, param, grad, state):

        update = grad

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