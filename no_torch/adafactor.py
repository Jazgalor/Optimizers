import numpy as np


class Adafactor:

    def __init__(
        self,
        lr=1e-2,
        beta2=0.999,
        eps=1e-30
    ):

        self.lr = lr
        self.beta2 = beta2
        self.eps = eps

        self.state = {}

    def init_state(self, param):

        shape = param.shape

        if len(shape) != 2:
            raise ValueError("Adafactor factorization requires 2D tensors")

        self.state[id(param)] = {
            "row_avg": np.zeros(shape[0]),
            "col_avg": np.zeros(shape[1]),
            "step": 0
        }

    def compute_update(self, param, grad):

        param_id = id(param)

        if param_id not in self.state:
            self.init_state(param)

        state = self.state[param_id]

        row_avg = state["row_avg"]
        col_avg = state["col_avg"]

        state["step"] += 1

        grad_sq = grad ** 2

        row_avg *= self.beta2
        row_avg += (1 - self.beta2) * grad_sq.mean(axis=1)

        col_avg *= self.beta2
        col_avg += (1 - self.beta2) * grad_sq.mean(axis=0)

        v_hat = np.outer(row_avg, col_avg)
        v_hat /= row_avg.mean()

        update = - self.lr * grad / (np.sqrt(v_hat) + self.eps)

        return update

    def step(self, params, grads):

        for param, grad in zip(params, grads):

            update = self.compute_update(param, grad)

            param += update