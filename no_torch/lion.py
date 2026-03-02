import numpy as np


class Lion:

    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=1e-1):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

        self.step = 0
        self.state = {}

    def init_state(self, param):

        return {
            "m": np.zeros_like(param),
        }

    def compute_update(self, param, grad, state):

        m = state["m"]

        # evolved sign momentum direction
        blended = self.beta1 * m + (1 - self.beta1) * grad
        update = np.sign(blended)

        # weight decay
        if self.weight_decay != 0:
            update = update + self.weight_decay * param

        # momentum update (AFTER computing direction)
        m[:] = self.beta2 * m + (1 - self.beta2) * grad

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