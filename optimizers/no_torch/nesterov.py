import numpy as np


class SGDNesterov:

    def __init__(self, lr=1e-3, beta=0.9):

        self.lr = lr
        self.beta = beta
        self.step = 0
        self.state = {}


    def init_state(self, param):

        return {
            "m": np.zeros_like(param)
        }


    def compute_update(self, param, grad, state):

        m = state["m"]

        # save previous momentum
        m_prev = m.copy()

        # update momentum
        m = self.beta * m + grad

        state["m"] = m

        # Nesterov update
        update = grad + self.beta * m

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