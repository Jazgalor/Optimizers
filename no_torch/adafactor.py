import numpy as np

class Adafactor:

    def __init__(
        self,
        lr=None,
        beta1=None,
        eps1=1e-30,
        eps2=1e-3,
        clip_threshold=1.0,
        beta2=-0.8,
        weight_decay=0.0,
        relative_step=True,
        scale_parameter=True
    ):
        self.lr = lr
        self.beta1 = beta1
        self.eps1 = eps1
        self.eps2 = eps2
        self.clip_threshold = clip_threshold
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.relative_step = relative_step
        self.scale_parameter = scale_parameter

        self.state = {}
        self.t = 0

    def _beta2_t(self):
        return 1.0 - self.t ** self.beta2

    def _get_lr(self, param):
        if self.relative_step:
            lr_t = min(1e-2, 1.0 / np.sqrt(self.t))
        else:
            lr_t = self.lr

        if self.scale_parameter:
            param_scale = max(self.eps2, np.linalg.norm(param))
            lr_t *= param_scale

        return lr_t

    def compute_update(self, param, grad, state):

        beta2_t = self._beta2_t()

        if grad.ndim >= 2:
            if "r" not in state:
                state["r"] = np.zeros(grad.shape[:-1])
                state["c"] = np.zeros(grad.shape[-1])

            grad_sq = grad ** 2 + self.eps1

            r = state["r"]
            c = state["c"]

            r = beta2_t * r + (1 - beta2_t) * grad_sq.mean(axis=-1)
            c = beta2_t * c + (1 - beta2_t) * grad_sq.mean(axis=0)

            v_hat = np.outer(r, c) / r.mean()

            state["r"], state["c"] = r, c

        else:
            if "v" not in state:
                state["v"] = np.zeros_like(param)

            v = state["v"]
            grad_sq = grad ** 2 + self.eps1

            v = beta2_t * v + (1 - beta2_t) * grad_sq
            v_hat = v

            state["v"] = v

        update = grad / (np.sqrt(v_hat) + self.eps1)

        # clipping
        update_norm = np.linalg.norm(update)
        clip_denom = max(1.0, update_norm / self.clip_threshold)
        update = update / clip_denom

        lr_t = self._get_lr(param)

        return lr_t * update

    def step(self, params, grads):

        self.t += 1

        for i, (param, grad) in enumerate(zip(params, grads)):

            if i not in self.state:
                self.state[i] = {}

            update = self.compute_update(param, grad, self.state[i])

            if self.weight_decay != 0:
                param -= self.weight_decay * param

            param -= update