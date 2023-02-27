from __future__ import print_function, annotations
from typing import Any, List, Sequence, Tuple, Union, Callable

import torch

from .state import BaseState
from .utils import dfdx_vmap


# --------------------------------------------------------------------------------------
# Constraint definition
# --------------------------------------------------------------------------------------
def print_ass(**args) -> int:
    assert (
        True
    ), "Not implemented. Either use the build() function or implement your own class"
    return 0


MyFunc = Union[
    Callable[[BaseState, Union[torch.Tensor, BaseState], Union[int, None]], Any], None
]


class Constraint:
    EQ = 0
    INEQ = 1
    LOSS = 2
    TYPE_STR = ["EQ", "INEQ", "LOSS"]

    def __init__(self, type: int, description="") -> None:
        self.type = type
        self._n_out: Union[Callable[[], int], None] = None
        self._evaluate: MyFunc = None
        self._jacobian: MyFunc = None  # returns (c_x, c_u)
        self._hessian: MyFunc = None  # returns ((c_xx, c_xu), (c_uu, c_ux))

        self.description = description

    def build(self, N_out, c, jac=None, hess=None):
        self._n_out = lambda: N_out
        self._evaluate = c
        self._jacobian = jac if jac is not None else self.default_jacobian
        self._hessian = hess if hess is not None else self.default_hessian

    def __call__(self, x: BaseState, u: torch.Tensor, t: int):
        return self.evaluate(x, u, t)

    # To bo implelemented by the user
    def n_out(self) -> int:
        if self._n_out is not None:
            return self._n_out()
        else:
            return 0

    def evaluate(
        self,
        x: BaseState,
        u: Union[torch.Tensor, BaseState],
        t: Union[int, None] = None,
    ) -> torch.Tensor:
        if self._evaluate is None:
            raise NotImplementedError
        return self._evaluate(x, u, t)

    def jacobian(
        self,
        x: BaseState,
        u: Union[torch.Tensor, BaseState],
        t: Union[int, None] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._jacobian is None:
            raise NotImplementedError
        return self._jacobian(x, u, t)

    def hessian(
        self,
        x: BaseState,
        u: torch.Tensor,
        t: Union[int, None] = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        if self._hessian is None:
            raise NotImplementedError
        return self._hessian(x, u, t)

    def default_jacobian(  # should work for both state and tensor
        self,
        x: BaseState,
        u: Union[torch.Tensor, BaseState],
        t: Union[int, None] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _x, _u = x.detach().requires_grad_(), u.detach().requires_grad_()
        c = self.evaluate(_x, _u, t)
        dx = dfdx_vmap(c, _x, allow_unused=True)
        du = dfdx_vmap(c, _u, allow_unused=True)
        return dx, du

    def default_hessian(  # TODO: does not work should work for both state and tensor
        self,
        x: BaseState,
        u,  #: Union[torch.Tensor, BaseState],
        t: Union[int, None] = None,
    ):
        return torch.autograd.functional.hessian(
            lambda x, u: self.evaluate(x, u, t),
            (x.d_in(), u if torch.is_tensor(u) else u.d_in()),
        )

    def get_log(self):
        return {"type": Constraint.TYPE_STR[self.type], "n_out": self.n_out(), "description": self.description}

class InEqConstraint(Constraint):
    def __init__(self, description="") -> None:
        super().__init__(Constraint.INEQ, description)


class EqConstraint(Constraint):
    def __init__(self, description="") -> None:
        super().__init__(Constraint.EQ, description)
