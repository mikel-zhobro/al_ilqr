from __future__ import print_function, annotations
from abc import ABC, abstractmethod
import time

import torch

from .state import BaseState
from .utils import dfdx_vmap, col


# --------------------------------------------------------------------------------------
# Loss function definition
# --------------------------------------------------------------------------------------
class LossFunctionBase(ABC):
    def __init__(self, nx: int, nu: int) -> None:
        self.nx = nx
        self.nu = nu

        self.default_lx = True
        self.default_lu = True
        self.default_lxx = True
        self.default_luu = True
        self.default_lux = True

        self.update_default()

    def update_default(self):
        self.default_lx = type(self)._dldx == LossFunctionBase._dldx
        self.default_lu = type(self)._dldu == LossFunctionBase._dldu
        self.default_lxx = type(self)._dldx == LossFunctionBase._dldxx
        self.default_luu = type(self)._dldx == LossFunctionBase._dlduu
        self.default_lux = type(self)._dldx == LossFunctionBase._dldux

    def print_info(self):
        print(f"--- {col.HEADER}Loss function Info{col.ENDC} ---")
        print(f"\tGradients are computed {'via autograd'if self.default_lx or self.default_lu else 'anallytically'}")

    @abstractmethod
    def evaluate(self, x, u, t: int, terminal=False) -> torch.Tensor:
        pass

    def __call__(self, x: BaseState, u: torch.Tensor, t: int, terminal=False):
        return self.evaluate(x, u, t, terminal)

    def _get_grad_hess(self, x_t: BaseState, u_t: torch.Tensor, t: int, terminal=False):
        """This is the main function used to get gradients and hessians of the loss.

        Args:
            x_t (BaseState): current state
            u_t (torch.Tensor): current input
            t (int): time step

        Returns:
            tuple[torch.Tensor]: l, lx, lu, lxx, luu, lux
        """

        _x: BaseState = x_t.detach()
        _u = u_t.detach()
        if self.default_lx or self.default_lu:
            _x = _x.requires_grad_()
            _u = _u.requires_grad_()
            l = self.evaluate(_x, _u, t, terminal)
            l.backward(retain_graph=True, create_graph=True)

        lx = self._dldx(_x, _u, t, terminal)
        lu = self._dldu(_x, _u, t, terminal)

        lxx = self._dldxx(lx, _x, _u, t, terminal)
        luu = self._dlduu(lu, _x, _u, t, terminal)
        lux = self._dldux(lu, _x, _u, t, terminal)

        return lx.detach(), lu.detach(), lxx.detach(), luu.detach(), lux.detach()

    def evaluate_trajectory_cost(
        self, x_array: list[BaseState], u_array: list[torch.Tensor]
    ):
        """Expects x_array to be one longer than u_array

        Args:
            x_array (list[BaseState]): length T+1
            u_array (list[Union[torch.Tensor, BaseState]]): length T

        Returns:
            torch.Tensor: accumulated loss
        """
        assert len(x_array) - 1 == len(u_array), f"Input array must be one shorter. {len(x_array) - 1} == {len(u_array)}"
        T = len(x_array) - 1
        opt_cost = torch.sum(
            torch.stack(
                # [self.evaluate(x, u, t) for t, (x, u) in enumerate(zip(x_array, u_array))]
                [
                    self.evaluate(x, u, t, False)
                    for t, (x, u) in enumerate(zip(x_array[:-1], u_array))
                ]
            )
        ) + self.evaluate(x_array[-1], None, len(x_array) - 1, True)
        return opt_cost

    def comp_trajectory_derivs(
        self,
        x_t_array: list[BaseState],  # N+1 length
        u_t_array: list[torch.Tensor],  # N length
    ):
        _start2 = time.time()
        dldx, dldu, dldxx, dlduu, dldux = [], [], [], [], []
        # [u.retain_grad() for u in u_t_array]

        # ll =torch.sum(torch.stack([self.evaluate(x, u, t) for t, (x, u) in enumerate(zip(x_t_array[:-1], u_t_array))]))
        # ll.backward(retain_graph=True)
        # dx = [x.d_in().grad for x in x_t_array[:-1]]
        # du = [u.grad for u in u_t_array]
        for t, (x_t, u_t) in enumerate(zip(x_t_array[:-1], u_t_array)):
            dldx_t, dldu_t, dldxx_t, dlduu_t, dldux_t = self._get_grad_hess(x_t, u_t, t, False)

            # assert torch.allclose(dx[t], dldx_t)
            # assert torch.allclose(du[t], dldu_t)

            dldx.append(dldx_t.view(-1, 1))
            dldu.append(dldu_t.view(-1, 1))
            dldxx.append(dldxx_t)
            dlduu.append(dlduu_t)
            dldux.append(dldux_t)

        x_end = x_t_array[-1]
        u_end = torch.zeros_like(u_t_array[-1])

        dldx_t, _, dldxx_t, _, _ = self._get_grad_hess(x_end, u_end, len(u_t_array), True)
        dldx.append(dldx_t.view(-1, 1))
        dldxx.append(dldxx_t)

        _end2 = time.time() - _start2
        return dldx, dldu, dldxx, dlduu, dldux, _end2

    ## The following definitions can be overloaded
    # ---------- Gradients --------------
    def _dldx(self, x_t, u_t: torch.Tensor, t: int, terminal=False) -> torch.Tensor:
        return (
            x_t.d_in().grad
            if x_t.d_in().grad is not None
            else torch.zeros_like(x_t.d_in())
        )

    def _dldu(self, x_t, u_t: torch.Tensor, t: int, terminal=False):
        return u_t.grad if u_t.grad is not None else torch.zeros_like(u_t)

    # ---------- Hessians --------------
    def _dldxx(self, dldx_t: torch.Tensor, x_t, u_t: torch.Tensor, t: int, terminal=False):
        return dfdx_vmap(dldx_t, x_t)

    def _dlduu(self, dldu_t: torch.Tensor, x_t, u_t: torch.Tensor, t: int, terminal=False):
        return dfdx_vmap(dldu_t, u_t)

    def _dldux(self, dldu_t: torch.Tensor, x_t, u_t: torch.Tensor, t: int, terminal=False):
        return dfdx_vmap(dldu_t, x_t, allow_unused=True)
