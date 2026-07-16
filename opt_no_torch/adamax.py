import numpy as np


class Adamax:

    def __init__(
        self,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.state = {}

    def init_state(self, param):

        self.state[id(param)] = {
            "step": 0,
            "exp_avg": np.zeros_like(param),
            "exp_inf": np.zeros_like(param)
        }

    def compute_update(self, param, grad):

        param_id = id(param)

        if param_id not in self.state:
            self.init_state(param)

        state = self.state[param_id]

        exp_avg = state["exp_avg"]
        exp_inf = state["exp_inf"]

        state["step"] += 1
        step = state["step"]

        # first moment
        exp_avg *= self.beta1
        exp_avg += (1 - self.beta1) * grad

        # infinity norm accumulator
        exp_inf *= self.beta2
        exp_inf = np.maximum(exp_inf, np.abs(grad))
        state["exp_inf"] = exp_inf

        # bias correction
        exp_avg_hat = exp_avg / (1 - self.beta1 ** step)

        # update
        update = - self.lr * exp_avg_hat / (exp_inf + self.eps)

        return update

    def step(self, params, grads):

        for param, grad in zip(params, grads):

            update = self.compute_update(param, grad)

            param += update