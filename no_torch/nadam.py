import numpy as np


class Nadam:

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
            "exp_avg_sq": np.zeros_like(param)
        }

    def compute_update(self, param, grad):

        param_id = id(param)

        if param_id not in self.state:
            self.init_state(param)

        state = self.state[param_id]

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        state["step"] += 1
        step = state["step"]

        # update moments
        exp_avg *= self.beta1
        exp_avg += (1 - self.beta1) * grad

        exp_avg_sq *= self.beta2
        exp_avg_sq += (1 - self.beta2) * (grad ** 2)

        # bias correction
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step

        exp_avg_hat = exp_avg / bias_correction1
        exp_avg_sq_hat = exp_avg_sq / bias_correction2

        # Nesterov correction
        exp_avg_nesterov = (
            self.beta1 * exp_avg_hat
            +
            ((1 - self.beta1) / bias_correction1) * grad
        )

        update = - self.lr * exp_avg_nesterov / (
            np.sqrt(exp_avg_sq_hat) + self.eps
        )

        return update

    def step(self, params, grads):

        for param, grad in zip(params, grads):

            update = self.compute_update(param, grad)

            param += update