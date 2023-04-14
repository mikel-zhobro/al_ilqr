import torch
from typing import Any, Union
import numpy as np

from .state import BaseState
from .loss import LossFunctionBase
from .dynamic_system import (
    DynamicSystem,
    BaseILQRDynSys,
    ILQRDynSys,
    ILQRDynSysClosedLoop,
    Controller,
)
from .constraints import Constraint
from .al_ilqr import PyLQR_iLQRSolver


def createILQRDynSys(
    T, plant: DynamicSystem, controller: Union[Controller, None] = None
) -> BaseILQRDynSys:
    if controller is not None:
        return ILQRDynSysClosedLoop(T, plant, controller)
    return ILQRDynSys(T, plant)


# --------------------------------------------------------------------------------------
# ilQR optimization problem setup-class definition
# --------------------------------------------------------------------------------------
class Optim:
    def __init__(self):
        self.res_dict: dict[str, Any] = dict()
        self.dyn_sys: DynamicSystem

    def build_ilqr_problem(
        self,
        T,
        dyn_sys: DynamicSystem,
        loss_func: LossFunctionBase,
        controller: Union[Controller, None] = None,
        **kwargs,
    ):
        self.T = T
        self.dyn_sys = dyn_sys
        self.ilqr_dyn_sys = createILQRDynSys(T, dyn_sys, controller)
        self.my_loss = loss_func
        self.controller: Union[Controller, None] = controller

        self.res_dict.clear()

        self.ilqr = PyLQR_iLQRSolver(
            plant_dyn=self.ilqr_dyn_sys,
            cost=self.my_loss,
            limits=dyn_sys.limits,
            T=T,
            **kwargs,
        )

    def solve_ilqr_problem(
        self,
        x0: BaseState,
        x_des_traj: list[BaseState],
        u_init_traj: Union[list[torch.Tensor], list[BaseState]],
        constraints: Union[list[Constraint], None]=None,
        verbose=True,
        initializer_ctrl: Union[Controller, None]=None,
        initializer_u_init: Union[list[BaseState], None]=None
    ):
        self.res_dict["T"] = self.T
        self.res_dict["x_plant_array_des"] = np.stack([x.numpy() for x in x_des_traj])

        # solve
        initializer = createILQRDynSys(self.T, self.dyn_sys, initializer_ctrl) if initializer_ctrl is not None else None
        self.res_dict = dict(
            **self.ilqr.solve(x0, u_init_traj,
                            constraints = constraints,
                            initializer = initializer,
                            initializer_u_init = initializer_u_init,
                            verbose = verbose),
            **self.res_dict,
            **self.dyn_sys.get_log_dict()
        )
        return self.res_dict
