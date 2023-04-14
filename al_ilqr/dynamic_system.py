from __future__ import print_function, annotations
from typing import Callable, Tuple, Union, Any
from abc import ABC, abstractmethod
import time
from tqdm import tqdm

import torch

from .utils import dfdx_vmap, filter_me, col, print_dict
from .state import BaseState, StackedState

# --------------------------------------------------------------------------------------
# Dynamic model definition
# --------------------------------------------------------------------------------------
class DynamicSystem(ABC):
    def __init__(self, dt: float, nx: int, nu: int, limits=None) -> None:
        self.dt = dt
        self.nx = nx
        self.nu = nu
        self.limits = limits

    @abstractmethod
    def state(self):
        raise NotImplementedError

    @abstractmethod
    def step_abs(self, x, u, t):
        raise NotImplementedError

    def step(self, x: BaseState, u: torch.Tensor, t: int) -> BaseState:
        """Step function of the plant. Does not include feedback loops.

        Args:
            x (BaseState): current state
            u (torch.Tensor): input to the plant
            t (_type_): current timestep

        Returns:
            BaseState: next state
        """
        if self.limits is not None:
            u_t = torch.clamp(u, min=self.limits[0], max=self.limits[1])
        else:
            u_t = u
        x_new = self.step_abs(x, u_t, t)
        return x_new

    # ---------- Gradients (can be implemented by the user)--------------
    def get_derivatives(self, x_t_1, x_t, u_t, t):
        return dfdx_vmap(x_t_1, x_t).detach(), dfdx_vmap(x_t_1, u_t).detach()

    # ---------- Logger --------------
    def get_log_dict(self) -> dict[str, Any]:
        return {
            'dt': self.dt,
            'nx': self.nx,
            'nu': self.nu
        }
    @property
    def requires_grad(self):
        return (type(self).get_derivatives == DynamicSystem.get_derivatives)

class BaseILQRDynSys(ABC):
    def __init__(self, plant: DynamicSystem, T) -> None:
        self.plant = plant
        self.T = T
        self.loss_traj: list[torch.Tensor] = []
        self.x_rollout_traj_0: list[BaseState] = []
        self.x_rollout_traj_1: list[BaseState] = []

        self.controller: Controller
        self.step_cost: Union[Callable, None] = None

        self.backup = None

    def print_info(self):
        print(f"--- {col.HEADER}Rollout Dynamic System Info{col.ENDC} ---")
        print(f"\tT: {self.T}")
        print()
        print(f"\t{col.HEADER}Plant{col.ENDC}")
        print_dict(self.plant.get_log_dict(), 1)
        print()
        print(f"\t{col.HEADER}Controller{col.ENDC} \n\t\t")

    def set_step_cost(self, step_cost: Callable):
        self.step_cost = step_cost

    @property
    def requires_grad(self):
        return self.plant.requires_grad

    @property
    def feedback_controlled(self):
        return hasattr(self, "controller")

    @abstractmethod
    def get_plant_rollouts(self) -> tuple[list[BaseState], list[torch.Tensor]]:
        pass

    @abstractmethod
    def get_rollouts(
        self,
    ) -> tuple[list[BaseState], Union[list[torch.Tensor], list[BaseState]]]:
        pass

    @abstractmethod
    def clear_rollouts(self):
        pass

    @abstractmethod
    def backup_rollouts(self):
        pass
    def get_backup(self):
        return self.backup

    @abstractmethod
    def reset_rollouts(self):
        pass

    @abstractmethod
    def _step_rollout(self, x, u, t: int, req_grad: bool) -> Tuple[BaseState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step function of the plant including feedback loop. Used in the lqr rollout function.
           It also backes up the state and input trajectories used.
        Returns:
            Tuple[BaseState, torch.Tensor, torch.Tensor, torch.Tensor]: new_state, ilqr_step_cost, aug_step_cost, step_violations
        """
        pass

    def _rollout(self, x0: BaseState, u_array):
        """Rollout from initial state."""
        req_grad = self.requires_grad
        # Structures to collect data
        x = x0.detach().clone()

        viols: list[torch.Tensor] = []
        ilqr_cost = x0._base_tensor.new_zeros(1)
        aug_cost = x0._base_tensor.new_zeros(1)
        _start = time.time()
        for t, u_tmp in enumerate(tqdm(u_array, desc="Rollout")):  # disturbs the cluster stdout
            x, ilqr_l, aug_l, viol = self._step_rollout(x, u_tmp, t, req_grad)
            ilqr_cost += ilqr_l
            aug_cost += aug_l
            viols.extend(viol)

        self.x_rollout_traj_0.append(x.detach().requires_grad_())
        _end1 = time.time() - _start
        print(f"\t--- Initial Rollout ({_end1:.5f})")
        return ilqr_cost+aug_cost, ilqr_cost, aug_cost, torch.stack(viols)

    def compute_step_costs(self, t, x_t, xp_t, u_t=None, up_t: Union[torch.Tensor, None]=None):
        lqr_l, aug_c, viol = [torch.zeros(1)]*3
        if self.step_cost is not None:
            opt_c, ctrl_c, lqr_c, aug_c, viol = self.step_cost(t, x_t, u_t, xp_t, up_t)
            lqr_l = opt_c + ctrl_c + lqr_c

        return lqr_l, aug_c, viol

    def rollout(
        self,
        current_cost: float,
        x0,
        x_array,  #: Union[list[BaseState], list[StackedState]],
        u_array,  #: list[torch.Tensor],
        k_array: list[torch.Tensor],
        K_array: list[torch.Tensor],
        alpha=1.0,
    ):
        """LQR rollout for given state trajectory and input trajctory according to feedback K and feedforward k."""
        self.clear_rollouts()
        req_grad = self.requires_grad

        _x0 = x0 if not self.feedback_controlled else StackedState([x0, self.controller.init_state()])
        # Case without feedback
        if not k_array or not K_array or not x_array: # i.e. either of the lists is empty
            tmp =  self._rollout(_x0, u_array)
            try:
                self.plant.render2()
            except:
                pass
            return tmp

        # Case with feedback
        # Structures to collect data
        viols: list[torch.Tensor] = []
        ilqr_cost = k_array[0].new_zeros(1)
        aug_cost = k_array[0].new_zeros(1)

        _start = time.time()
        x = _x0
        for t, u_tmp in enumerate(tqdm(u_array, desc="Rollout")):
            # Compute u
            u = u_tmp + (
                alpha * k_array[t] + K_array[t] @ (x.diff(x_array[t]).view(-1, 1))
            ).view(-1)

            x, ilqr_l, aug_l, viol = self._step_rollout(x, u, t, req_grad)
            ilqr_cost += ilqr_l
            aug_cost += aug_l
            viols.extend(viol)
            running_cost = (ilqr_cost+aug_cost).item()
            if running_cost > current_cost:
                # print(f"\t--- Rollout exploded with cost:{running_cost}>{current_cost}, time:({time.time()-_start:.5f})")
                return ilqr_cost+aug_cost, ilqr_cost, aug_cost, torch.stack(viols)

        self.x_rollout_traj_0.append(x.detach().requires_grad_())

        try:
            self.plant.render2()
        except:
            pass
        _end1 = time.time() - _start
        print(f"\t--- Rollout ({_end1:.5f})")
        return ilqr_cost+aug_cost, ilqr_cost, aug_cost, torch.stack(viols)

    @abstractmethod
    def comp_derivs(
        self, alpha=1.0  # smoothout factor
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor], float]:
        pass

    def compute_controller_cost(
        self, x_c, t, terminal=False
    ) -> torch.Tensor:
        return self.controller.compute_cost(x_c, t, terminal)


class ILQRDynSys(BaseILQRDynSys):
    def __init__(self, T, plant: DynamicSystem) -> None:
        super().__init__(plant, T)
        self.u_rollout_traj: list[torch.Tensor] = []

    # ---------- Implement abstract methods --------------
    def get_plant_rollouts(self):
        return self.get_rollouts()

    def get_rollouts(self):
        x_traj = self.x_rollout_traj_0# + [self.x_rollout_traj_1[-1].detach().requires_grad_()]  # x0, .. xT
        u_traj = self.u_rollout_traj  # u0, .. uT-1
        return x_traj, u_traj

    def reset_rollouts(self):
        if self.backup is None:
            return
        self.u_rollout_traj, self.x_rollout_traj_0, self.x_rollout_traj_1 = self.backup

    def backup_rollouts(self):
        self.backup = (
            self.u_rollout_traj,
            self.x_rollout_traj_0,
            self.x_rollout_traj_1,
        )

    def clear_rollouts(self):
        self.u_rollout_traj = []
        self.x_rollout_traj_0 = []
        self.x_rollout_traj_1 = []

    def _step_rollout(self, x: BaseState, u: torch.Tensor, t: int, req_grad: bool):
        """Step function of the plant including feedback loop. Used in the lqr rollout function

        Args:
            x (BaseState): current state
            u (torch.Tensor): current input (could be type state)
            t (int): timestep
            req_grad (bool): whether we require grads

        Returns:
            Basestate: next state
        """
        x_t = x.detach().requires_grad_(req_grad)
        u_t = u.detach().view(-1).requires_grad_(req_grad)
        x_t_1 = self.plant.step(x_t, u_t, t)

        # x_t_1.requires_grad_()
        self.u_rollout_traj.append(u_t)
        self.x_rollout_traj_0.append(x_t)
        self.x_rollout_traj_1.append(x_t_1)

        ilqr_l, aug_l, viol = self.compute_step_costs(t, x_t, x_t, u_t, u_t)
        viol = [viol]

        if t == self.T-1:
            # print("TERMINALKA no feedback")
            ilqr_l_term, aug_l_term , viol_term = self.compute_step_costs(self.T, x_t, x_t)
            ilqr_l += ilqr_l_term
            aug_l += aug_l_term
            viol.append(viol_term)

        return x_t_1.detach(), ilqr_l, aug_l , viol

    def comp_derivs(self, alpha=1.0):  # smoothout factor
        _start2 = time.time()

        dfdx: list[torch.Tensor] = []
        dfdu: list[torch.Tensor] = []
        t = 0
        for x_t, u_t, x_t_1 in zip(self.x_rollout_traj_0, self.u_rollout_traj, self.x_rollout_traj_1):
            dfdx_t, dfdu_t = self.plant.get_derivatives(x_t_1, x_t, u_t, t)

            # Smoothing
            # if t > 0:
            #     dfdx_t = alpha * dfdx_t + (1.0 - alpha) * dfdx[t - 1]
            #     dfdu_t = alpha * dfdu_t + (1.0 - alpha) * dfdu[t - 1]
            dfdx.append(dfdx_t)
            dfdu.append(dfdu_t)
            t += 1

        _end2 = time.time() - _start2
        return dfdx, dfdu, _end2

class ILQRDynSysClosedLoop(BaseILQRDynSys):
    def __init__(self, T, plant: DynamicSystem, controller: Controller) -> None:
        super().__init__(plant, T)
        self.controller = controller
        self._u_plant_rollout: list[torch.Tensor] = []
        self._x_plant_rollout_0: list[BaseState] = []
        self._x_plant_rollout_1: list[BaseState] = []
        self.u_rollout_traj: list[BaseState] = []

    @property
    def feedback_controlled(self):
        return True

    # ---------- Implement abstract methods --------------
    def get_plant_rollouts(self):
        return (
            self._x_plant_rollout_0 + [self.x_rollout_traj_0[-1].states_list[0]],
            self._u_plant_rollout,
        )

    def get_rollouts(self) -> Tuple[list[BaseState], list[BaseState]]:  # overload
        x_traj = self.x_rollout_traj_0  # x0, .. xT   # len T
        u_traj = self.u_rollout_traj  # u0, .. uT-1   # len T
        return x_traj, u_traj

    def reset_rollouts(self):
        if self.backup is None:
            return
        (
            self.u_rollout_traj,
            self.x_rollout_traj_0,
            self.x_rollout_traj_1,
            self._u_plant_rollout,
            self._x_plant_rollout_0,
            self._x_plant_rollout_1,
        ) = self.backup

    def backup_rollouts(self):
        self.backup = (
            self.u_rollout_traj,
            self.x_rollout_traj_0,
            self.x_rollout_traj_1,
            self._u_plant_rollout,
            self._x_plant_rollout_0,
            self._x_plant_rollout_1,
        )

    def clear_rollouts(self):  # overload
        self._u_plant_rollout = []
        self._x_plant_rollout_0 = []
        self._x_plant_rollout_1 = []

        self.u_rollout_traj = []
        self.x_rollout_traj_0 = []
        self.x_rollout_traj_1 = []

    def _step_rollout(
        self, x_hat: StackedState, u_hat: BaseState, t: int, req_grad: bool
    ):
        # Stacked states check
        if t == 0:
            print("INIT CONTROLLER!")
            # x_hat = StackedState([x_hat, self.controller.init_state()])
            x_hat.states_list[1] = self.controller.init_state()
        x_hat_ = x_hat.detach().requires_grad_(req_grad)
        u_hat_ = u_hat.detach().requires_grad_(req_grad)
        x_plant_old = x_hat_.states_list[0]

        # Controller stuff
        u_plant, x_c_new = self.controller.compute_inpute_update_state(u_hat_, *x_hat_.states_list)


        x_plant_new = self.plant.step(x_hat_.states_list[0], u_plant, t)
        x_hat_new_ = StackedState([x_plant_new, x_c_new])

        self.append_step_rollout(
            u_hat_, x_hat_, x_hat_new_, u_plant, x_plant_old, x_plant_new
        )
        ilqr_l, aug_l, viol = self.compute_step_costs(t, x_hat_, x_plant_old, u_hat_, u_plant)

        viol = [viol]
        if t == self.T-1:
            print("TERMINALKA")
            ilqr_l_term, aug_l_term , viol_term = self.compute_step_costs(self.T, x_hat_new_, x_plant_new)

            ilqr_l += ilqr_l_term
            aug_l += aug_l_term
            viol.append(viol_term)
        return x_hat_new_.detach(), ilqr_l, aug_l, viol

    # Helper
    def append_step_rollout(
        self, u_hat_t, x_hat_t, x_hat_t1, u_plant_t, x_plant_t, x_plant_t1
    ):
        self.u_rollout_traj.append(u_hat_t)
        self.x_rollout_traj_0.append(x_hat_t)
        self.x_rollout_traj_1.append(x_hat_t1)
        self._u_plant_rollout.append(u_plant_t)
        self._x_plant_rollout_0.append(x_plant_t)
        self._x_plant_rollout_1.append(x_plant_t1)

    def comp_derivs(self, alpha=1.0):  # smoothout factor
        _start2 = time.time()

        dfdx: list[torch.Tensor] = []
        dfdu: list[torch.Tensor] = []
        t = 0
        for x_t, u_t, x_t_1 in zip(self.x_rollout_traj_0, self.u_rollout_traj, self.x_rollout_traj_1):
            dfdx_t = dfdx_vmap(x_t_1, x_t)
            dfdu_t = dfdx_vmap(x_t_1, u_t)

            dfdx.append(dfdx_t)
            dfdu.append(dfdu_t)
            t += 1
        # Smoothing ( does not work)
        # dfdx = torch.stack(dfdx)
        # dfdu = torch.stack(dfdu)
        # if True:
        #     dfdx_ = filter_me(dfdx, 4)
        #     dfdu_ = filter_me(dfdu, 4)

        _end2 = time.time() - _start2
        return list(dfdx), list(dfdu), _end2


# --------------------------------------------------------------------------------------
# Controll definition
# --------------------------------------------------------------------------------------
class Controller(ABC):
    def __init__(self, base_tensor: torch.Tensor) -> None:
        self.base_tensor = base_tensor
        self.has_cost = type(self).compute_traj_cost != Controller.compute_traj_cost

    @abstractmethod
    def init_state(self) -> BaseState:
        pass

    @abstractmethod
    def _get_err(self, x_des: BaseState, x: BaseState) -> torch.Tensor:
        pass

    @abstractmethod
    def _compute_input(self, err: torch.Tensor, x: BaseState, x_c: BaseState) -> torch.Tensor:
        pass

    @abstractmethod
    def _update_state(self, err, x_c) -> BaseState:
        pass

    def compute_inpute_update_state(self, x_des: BaseState, x: BaseState, x_c: BaseState):
        err = self._get_err(x_des, x)
        u_plant = self._compute_input(err, x, x_c).view(-1)
        x_c_new = self._update_state(err, x_c)
        return u_plant, x_c_new

    def compute_cost(self, x_t: BaseState, t: int, terminal=False) -> torch.Tensor:
        return x_t._base_tensor.new_zeros(1)

    def compute_traj_cost(self, x_rollout: list[BaseState]) -> torch.Tensor:
        return x_rollout[0]._base_tensor.new_zeros(1)

    # @abstractmethod
    # def state_derivatives(self, x_des:BaseState, x:BaseState, x_c:BaseState, req_partial_grad=False):
    #     pass

    # @abstractmethod
    # def input_derivatives(self, x_des:BaseState, x:BaseState, x_c:BaseState, req_partial_grad=False):
    #     pass





def test_gradients(dyn_sys: DynamicSystem, x: BaseState, u:torch.Tensor, eps = 1e-15):
    # Implementation
    _x0 = x.clone().detach().requires_grad_(True)
    _u0 = u.clone().detach().requires_grad_(True)
    x1 = dyn_sys.step_abs(_x0, _u0, 0)
    dx1_x0, dx1_u0 = dyn_sys.get_derivatives(x1, _x0, _u0, 0)


    # Finite differences check
    nx = dyn_sys.nx
    nu = dyn_sys.nu

    dx1_x0_fd, dx1_u0_fd = torch.zeros((nx, nx)), torch.zeros((nx, nu))

    # tqdm print progress

    for n in range(nx):
        dx = torch.zeros(nx)
        dx[n] += eps
        x1 = dyn_sys.step_abs(x + dx, u, 0)
        x0 = dyn_sys.step_abs(x + (-dx), u, 0)
        dx1_x0_fd[:, n] = (x1 - x0) / (2 * eps)

    for n in range(nu):
        du = torch.zeros(nu)
        du[n] += eps
        x1 = dyn_sys.step_abs(x, u + du, 0)
        x0 = dyn_sys.step_abs(x, u + (-du), 0)
        dx1_u0_fd[:, n] = (x1 - x0) / (2 * eps)

    return dx1_x0, dx1_u0, dx1_x0_fd, dx1_u0_fd