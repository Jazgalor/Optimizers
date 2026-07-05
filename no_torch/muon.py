import numpy as np


class Muon:

    def __init__(
        self,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=5,
        eps=1e-7,
    ):

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.eps = eps

        self.step = 0
        self.state = {}

    def init_state(self, param):

        return {
            "m": np.zeros_like(param),
        }

    def newton_schulz5(self, matrix):

        if matrix.ndim != 2:
            raise ValueError("Newton-Schulz requires a 2D matrix.")

        a = 3.4445
        b = -4.7750
        c = 2.0315

        X = matrix.copy()

        transpose = False

        if X.shape[0] > X.shape[1]:
            X = X.T
            transpose = True

        X /= np.linalg.norm(X) + self.eps

        for _ in range(self.ns_steps):

            A = X @ X.T
            B = b * A + c * (A @ A)

            X = a * X + B @ X

        if transpose:
            X = X.T

        return X

    def compute_update(self, param, grad, state):

        m = state["m"]

        m *= self.momentum
        m += grad

        if m.ndim == 1:

            update = m.copy()

        elif m.ndim == 2:

            update = self.newton_schulz5(m)

        elif m.ndim == 4:

            original_shape = m.shape

            matrix = m.reshape(m.shape[0], -1)

            update = self.newton_schulz5(matrix)

            update = update.reshape(original_shape)

        else:

            raise ValueError(
                f"Unsupported parameter dimension ({m.ndim}) for Muon."
            )

        return update

    def apply_update(self, param, update):

        if self.weight_decay != 0:

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