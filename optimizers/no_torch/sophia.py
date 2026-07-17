import numpy as np


class Sophia:

    def __init__(
        self,
        lr=1e-4,
        beta1=0.965,
        beta2=0.99,
        gamma=0.04,
        weight_decay=0.1,
        eps=1e-15,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.eps = eps

        self.step = 0
        self.state = {}

    def init_state(self, param):
        return {
            "m": np.zeros_like(param),
            "h": np.zeros_like(param),
        }

    def update_hessian(self, param, grad):
        param_id = id(param)

        if param_id not in self.state:
            self.state[param_id] = self.init_state(param)

        state = self.state[param_id]

        state["h"] = (
            self.beta2 * state["h"]
            + (1 - self.beta2) * (grad * grad)
        )

    def compute_update(self, param, grad, state):

        m = state["m"]
        h = state["h"]

        # first moment
        m[:] = self.beta1 * m + (1 - self.beta1) * grad

        denom = np.maximum(self.gamma * h, self.eps)

        ratio = np.minimum(np.abs(m) / denom, 1.0)

        update = np.sign(m) * ratio

        return update

    def apply_update(self, param, update):

        # decoupled weight decay
        param *= (1 - self.lr * self.weight_decay)

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