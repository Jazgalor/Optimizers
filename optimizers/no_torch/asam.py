import numpy as np


class ASAM:

    def __init__(self, optimizer, rho=0.05, eps=1e-12):

        self.optimizer = optimizer
        self.rho = rho
        self.eps = eps

    def step(self, params, grads, closure):

        # 1. compute ||T_w g||
        scaled_norm = 0.0

        for param, grad in zip(params, grads):

            scaled = np.abs(param) * grad
            scaled_norm += np.sum(scaled ** 2)

        scaled_norm = np.sqrt(scaled_norm)
        scaled_norm = max(scaled_norm, self.eps)

        # 2. perturbation
        eps_list = []

        for param, grad in zip(params, grads):

            perturbation = (np.abs(param) ** 2) * grad

            eps = self.rho * perturbation / scaled_norm

            param += eps
            eps_list.append(eps)

        # 3. gradient at perturbed weights
        _, new_grads = closure(params)

        # 4. restore parameters
        for param, eps in zip(params, eps_list):
            param -= eps

        # 5. base optimizer step
        self.optimizer.step(params, new_grads)