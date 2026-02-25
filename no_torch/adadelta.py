import numpy as np


class Adadelta:

    def __init__(self, beta=0.9, eps=1e-6):

        self.beta = beta
        self.eps = eps

        self.state = {}

    def init_state(self, param):

        self.state[id(param)] = {
            "step": 0,
            "square_avg": np.zeros_like(param),
            "acc_delta": np.zeros_like(param)
        }

    def compute_update(self, param, grad):

        param_id = id(param)

        if param_id not in self.state:
            self.init_state(param)

        state = self.state[param_id]

        square_avg = state["square_avg"]
        acc_delta = state["acc_delta"]

        state["step"] += 1

        # EMA grad²
        square_avg *= self.beta
        square_avg += (1 - self.beta) * (grad ** 2)

        # compute update
        rms_delta = np.sqrt(acc_delta + self.eps)
        rms_grad = np.sqrt(square_avg + self.eps)

        update = - (rms_delta / rms_grad) * grad

        # EMA update²
        acc_delta *= self.beta
        acc_delta += (1 - self.beta) * (update ** 2)

        return update

    def step(self, params, grads):

        for param, grad in zip(params, grads):

            update = self.compute_update(param, grad)

            param += update