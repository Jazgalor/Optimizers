import numpy as np


class SAM:

    def __init__(self, optimizer, rho=0.05):

        self.optimizer = optimizer
        self.rho = rho

    def step(self, params, grads, closure):

        # closure: funkcja licząca nowe grads dla aktualnych params

        grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads))

        if grad_norm == 0:
            grad_norm = 1e-12

        eps_list = []

        for param, grad in zip(params, grads):

            eps = self.rho * grad / grad_norm
            param += eps
            eps_list.append(eps)

        _, new_grads = closure(params)

        for param, eps in zip(params, eps_list):
            param -= eps

        self.optimizer.step(params, new_grads)