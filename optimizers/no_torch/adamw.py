import numpy as np


class AdamW:

    def __init__(
        self,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-2
    ):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

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
        exp_avg_hat = exp_avg / (1 - self.beta1 ** step)
        exp_avg_sq_hat = exp_avg_sq / (1 - self.beta2 ** step)

        # Adam update
        update = - self.lr * exp_avg_hat / (
            np.sqrt(exp_avg_sq_hat) + self.eps
        )

        # weight decay (decoupled)
        update -= self.lr * self.weight_decay * param

        return update

    def step(self, params, grads):

        for param, grad in zip(params, grads):

            update = self.compute_update(param, grad)

            param += update