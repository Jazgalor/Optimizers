import torch
from torch.optim import Optimizer


class AdafactorTorch(Optimizer):

    def __init__(
        self,
        params,
        eps1=1e-30,
        eps2=1e-3,
        d=1.0,
        beta2_decay=-0.8,
    ):

        defaults = dict(
            eps1=eps1,
            eps2=eps2,
            d=d,
            beta2_decay=beta2_decay,
        )

        super().__init__(
            params,
            defaults,
        )


    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:

            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:

            eps1 = group["eps1"]
            eps2 = group["eps2"]
            d = group["d"]
            beta2_decay = group["beta2_decay"]


            for param in group["params"]:

                if param.grad is None:
                    continue


                grad = param.grad

                state = self.state[param]


                # ==========================================
                # INITIALIZATION
                # ==========================================

                if len(state) == 0:

                    state["step"] = 0


                    if grad.ndim > 1:

                        # Factorization over
                        # the last two dimensions

                        row_shape = list(grad.shape)

                        row_shape[-1] = 1


                        col_shape = list(grad.shape)

                        col_shape[-2] = 1


                        state["row_var"] = (torch.zeros( row_shape, device=param.device, dtype=param.dtype,))

                        state["col_var"] = (torch.zeros(col_shape, device=param.device, dtype=param.dtype,))


                    else:

                        state["variance"] = (torch.zeros_like(param))


                # ==========================================
                # STEP
                # ==========================================

                state["step"] += 1

                t = state["step"]


                # ==========================================
                # BETA2
                # ==========================================

                # beta2_t = 1 - t^(-0.8)

                one_minus_beta2_t = (t ** beta2_decay)


                beta2_t = (1.0 - one_minus_beta2_t)


                # ==========================================
                # RELATIVE STEP SIZE
                # ==========================================

                # rho_t = min(10^-2, 1/sqrt(t))

                rho_t = min(1e-2, 1.0 / t ** (0.5), )


                # ==========================================
                # PARAMETER SCALE
                # ==========================================

                # RMS(X)

                param_rms = torch.div(param.norm(), (param.numel() ** 0.5))


                # alpha_t =
                # max(eps2, RMS(X)) * rho_t

                alpha_t = torch.clamp(param_rms, min=eps2) * rho_t


                # ==========================================
                # MATRIX / MULTI-DIMENSIONAL PARAMETERS
                # ==========================================

                if grad.ndim > 1:

                    row_var = state["row_var"]

                    col_var = state["col_var"]


                    # ======================================
                    # G^2 + eps1
                    # ======================================

                    grad_squared = torch.mul(grad, grad).add(eps1)


                    # ======================================
                    # R_t
                    # ======================================

                    row_mean = (grad_squared.mean(dim=-1, keepdim=True, ))


                    row_var.mul_(beta2_t)

                    row_var.add_(row_mean, alpha=one_minus_beta2_t, )


                    # ======================================
                    # C_t
                    # ======================================

                    col_mean = (grad_squared.mean(dim=-2, keepdim=True,))


                    col_var.mul_(beta2_t)

                    col_var.add_(col_mean, alpha=one_minus_beta2_t,)


                    # ======================================
                    # V_hat_t
                    # ======================================

                    variance = torch.matmul(row_var, col_var)


                    row_mean_value = (row_var.mean(dim=-2, keepdim=True,))


                    variance.div_(row_mean_value)


                # ==========================================
                # VECTOR PARAMETERS
                # ==========================================

                else:

                    variance = state["variance"]


                    grad_squared = torch.mul(grad, grad).add(eps1)


                    # V_hat_t =
                    # beta2_t * V_hat_(t-1)
                    # +
                    # (1 - beta2_t) * G_t^2

                    variance.mul_(beta2_t)

                    variance.add_(grad_squared, alpha=one_minus_beta2_t,)


                # ==========================================
                # U_t
                # ==========================================

                update = torch.div(grad, variance.sqrt())


                # ==========================================
                # CLIPPING
                # ==========================================

                # RMS(U_t)

                update_rms = torch.div(update.norm(), (update.numel() ** 0.5))


                # U_hat_t =
                # U_t /
                # max(1, RMS(U_t) / d)

                clip_denom = torch.clamp(update_rms / d, min=1.0)


                update.div_(clip_denom)


                # ==========================================
                # PARAMETER UPDATE
                # ==========================================

                # X_t =
                # X_(t-1) - alpha_t * U_hat_t

                param.add_(update, alpha=-alpha_t, )


        return loss