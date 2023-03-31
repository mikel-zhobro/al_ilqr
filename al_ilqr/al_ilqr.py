"""
AL-ILQR implementation in PyTorch based on [1](https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf)
"""
from __future__ import print_function, annotations
from typing import Callable, Any, TYPE_CHECKING, Union

import time
import pickle
import numpy as np
import torch

from .utils import dfdx_vmap, print_dict, col

from .config import iLQRConfigDefault, ALConfigDefault

if TYPE_CHECKING:
    from .helpers import Constraint, BaseILQRDynSys, LossFunctionBase, BaseState

    Plant = Callable[[BaseState, torch.Tensor, int, Any], BaseState]


class AugmentedElements:
    def __init__(self, constraints: list[Constraint] = list()) -> None:
        # Hyperparameters
        self.conf = ALConfigDefault()

        # Augmentations
        self.current_it: int = 1
        self.lamda: torch.Tensor  # T_steps * N_c
        self.mu: torch.Tensor  # T_steps * N_c, penalty

        # Cosntraint related
        self.N_c: int = 0
        self.N: int = 0
        self.constraints: list[Constraint] = list()
        self.eq_ix: list[int] = []
        self.ineq_ix: list[int] = []
        if constraints:
            self.add_constraints(constraints)

    def add_constraints(self, constraints: list[Constraint]):
        _constraints = self.constraints + constraints

        # Constraints
        self.N_c = len(_constraints)
        self.N = sum([C.n_out() for C in _constraints])
        self.constraints = _constraints

        _n = 0
        for c in self.constraints:
            if c.type == c.EQ:
                self.eq_ix.extend(range(_n, _n + c.n_out()))
            elif c.type == c.INEQ:
                self.ineq_ix.extend(range(_n, _n + c.n_out()))
            else:
                assert False, f"Constraing of type: {c.type} is not valid."
            _n += c.n_out()

    @property
    def not_empty(self):
        return self.N > 0

    def initialize(self, base_tensor: torch.Tensor):
        self.lamda = base_tensor.new_zeros(
            (self.T + 1, self.N)
        )  # T_steps+1 * N_c
        self.mu = self.conf.al_mu_init * base_tensor.new_ones(
            (self.T + 1, self.N)
        )  # T_steps+1 * N_c

    def initialize2(self, x_array: list[BaseState], u_array: list[Any]):
        if self.not_empty:
            assert (
                len(x_array) - 1 == len(u_array) == self.T
            ), f"{len(x_array)-1} == {len(u_array)} == {self.T}"

            self.lamda = x_array[0]._base_tensor.new_zeros(
                (self.T + 1, self.N)
            )  # T_steps+1 * N_c
            self.mu = self.conf.al_mu_init * x_array[0]._base_tensor.new_ones(
                (self.T + 1, self.N)
            )  # T_steps+1 * N_c

    @property
    def T(self):
        return self.conf.T

    def _evaluate(self, x_plant: BaseState, u_plant, t):
        return torch.cat([C(x_plant, u_plant, t) for C in self.constraints])  # (N_c, )

    def _get_aug_lag_matrices(
        self, x_plant: BaseState, u_plant: Union[torch.Tensor, BaseState], t
    ):
        lamda_t = self.lamda[t].view(-1, 1)
        c_t = self._evaluate(x_plant, u_plant, t).view(-1, 1)
        tmp = self.mu[t].clone().view(-1, 1)
        tmp[torch.logical_and(c_t < 0, torch.abs(lamda_t) < 1e-5)] = 0
        tmp[self.eq_ix] = self.mu[t][self.eq_ix].view(-1, 1)
        Iuk = tmp.view(-1).diag()  # (N_c, N_c)
        return c_t, lamda_t, Iuk

    def _jacobian(self, x: BaseState, u: Union[torch.Tensor, BaseState], t: int):
        cx: list[torch.Tensor] = []
        cu: list[torch.Tensor] = []

        for C in self.constraints:
            cxi, cui = C.jacobian(x, u, t)
            cx.append(cxi)
            cu.append(cui)
        return torch.row_stack(cx), torch.row_stack(cu)  # (N_c, n_x), (N_c, n_u)

    def aug_step_cost(self, x: BaseState, u, t:int):
        cost, violation = x._base_tensor.new_zeros(1), x._base_tensor.new_zeros(1)
        if self.not_empty:
            c_t, lamda_t, Iuk = self._get_aug_lag_matrices(x, u, t)
            cost = lamda_t.T @ c_t + 0.5 * c_t.T @ Iuk @ c_t
            violation = c_t.detach().abs()
        return cost.view(-1), violation

    # @torch.no_grad()
    def compute_aug_cost(
        self,
        x_plant_array: list[BaseState],
        u_plant_array: list[torch.Tensor],
    ):
        u_plant_array_ =  u_plant_array + [u_plant_array[-1].detach()]
        aug_cost_ = x_plant_array[0]._base_tensor.new_zeros(1)
        cost_arr = []
        violations = []
        if self.not_empty:
            for t, x, u in zip(range(self.T + 1), x_plant_array, u_plant_array_):
                cost, viol = self.aug_step_cost(x, u, t)
                cost_arr.append(cost)
                violations.append(viol)

            aug_cost_ = torch.sum(torch.stack(cost_arr))
        return aug_cost_, torch.stack(
            violations
        ).squeeze().detach() if violations else torch.empty(0)

    def augment_t(
        self,
        t,
        Qx: torch.Tensor,
        Qu: torch.Tensor,
        Qxx: torch.Tensor,
        Quu: torch.Tensor,
        Qux: torch.Tensor,
        x: BaseState,
        u: Union[torch.Tensor, BaseState],
        x_plant_t: BaseState,
        u_plant_t: torch.Tensor,
    ):
        if self.not_empty:
            with torch.enable_grad():
                x_plant_t.requires_grad_()
                u_plant_t.requires_grad_()
                c_t, lamda_t, Iuk = self._get_aug_lag_matrices(x_plant_t, u_plant_t, t)
                # c_t = torch.cat([C(x, u, t) for C in self.constraints])
                cx = dfdx_vmap(c_t.view(-1), x)
                cu = dfdx_vmap(c_t.view(-1), u, allow_unused=True)
                # cx, cu = self._jacobian(x, u, t) # (N_c, n_x), (N_c, n_u)

            Qx += cx.T @ (lamda_t + Iuk @ c_t)
            Qu += cu.T @ (lamda_t + Iuk @ c_t)
            Qxx += cx.T @ Iuk @ cx
            Quu += cu.T @ Iuk @ cu
            Qux += cu.T @ Iuk @ cx

        return Qx, Qu, Qxx, Quu, Qux

    @torch.no_grad()
    def update_batch(
        self,
        x_plant_array: list[BaseState],
        u_plant_array: Union[list[torch.Tensor], list[BaseState]],
    ):
        assert len(x_plant_array) - 1 == len(u_plant_array) == self.T

        if self.not_empty > 0:
            lambda_tmp = self.lamda + self.mu * torch.stack(
                [
                    self._evaluate(x, y, t)
                    for t, (x, y) in enumerate(zip(x_plant_array, u_plant_array))
                ]
                + [
                    self._evaluate(
                        x_plant_array[-1], u_plant_array[-1].detach(), self.T
                    )
                ]
            )
            self.lamda[:, self.eq_ix] = lambda_tmp[:, self.eq_ix]
            self.lamda[:, self.ineq_ix] = torch.clamp(
                lambda_tmp[:, self.ineq_ix], min=0.0
            )
            self.mu = self.mu * self.conf.al_psi
            self.current_it += 1

    def print_info(self, n_tab=0):
        print("\t"*n_tab + f"--- {col.HEADER}Augmented Lagrangian Info{col.ENDC}---")
        print_dict(self.conf, n_tab)
        print()
        print("\t"*n_tab + f"\t- {col.HEADER}Constraints Info{col.ENDC} -")
        for i, c in enumerate(self.constraints):
            print("\t"*n_tab + f"\t{i}. {c.get_log()}")

class PyLQR_iLQRSolver:
    """
    Discrete time finite horizon iLQR solver
    """

    def __init__(
        self,
        plant_dyn: BaseILQRDynSys,
        cost: LossFunctionBase,
        limits=None,
        log_dir: Union[str, None] = None,
        **kwargs,
    ):
        """
        T:              Length of horizon
        plant_dyn:      Discrete time plant dynamics, can be nonlinear
        cost:           instaneous cost function; the terminal cost can be defined by judging the time index
        constraints:    ineq_constraints on state/control; ineq_constraints(x, u, t) <= 0
        constraints:    eq_constraints on state/control; eq_constraints(x, u, t) == 0

        All the functions should accept (x, u, t) but not necessarily depend on all of them.
        """
        self.debug = True
        self.log_dir = log_dir
        # Hyperparameters
        self.reg = 0.1
        self.conf = iLQRConfigDefault()

        # Constraints
        self.limits = limits
        self.augmented = AugmentedElements()

        # Update hyperparams
        self.reset(**kwargs)

        # Backups
        self.lqr_sys_backup: dict[str, list[torch.Tensor]]
        self.prev_rollouts: Union[tuple[ list[BaseState], Union[list[BaseState], list[torch.Tensor]]], None] = None

        # plant and cost
        self.cost = cost  # (BaseState, U, t) -> l
        self.plant_dyn = plant_dyn  # (BaseState, U, t) -> BaseState
        self.plant_dyn.set_step_cost(self.evaluate_step_cost)

        # Logging:
        self.ff_logs: dict[int, Any]
        self.J_hist_total: list[float]
        self.J_hist_ilqr: list[float]
        self.J_hist_aug: list[float]

        # iLQR entities
        self.k_array: list[torch.Tensor] = []
        self.K_array: list[torch.Tensor] = []
        self.delta_V_u: float
        self.delta_V_uu: float

    @property
    def T(self):
        return self.conf.T  # timesteps

    def reset(self, **kwargs):
        self.conf.update(**kwargs)
        self.augmented.conf.update(**kwargs)

    def evaluate_step_cost(self, t, x: BaseState, u, xp, up):
        terminal = (t == self.conf.T)

        opt_cost = self.cost.evaluate(xp, up, t, terminal)

        # controller state cost
        ctrl_cost = opt_cost.new_zeros(1)
        # if self.plant_dyn.feedback_controlled:
        #     ctrl_cost = self.plant_dyn.compute_controller_cost(x, t, terminal)

        # ilqr cost
        alpha = 1e-3
        ilqr_cost = opt_cost.new_zeros(1)
        dx = 0.
        du = 0.
        if self.prev_rollouts is not None and u is not None:
            du = self.conf.dx_alpha*torch.sum((self.prev_rollouts[0][t] - x)**2)
            du = self.conf.du_alpha*torch.sum((self.prev_rollouts[1][t] - u)**2)
            ilqr_cost = dx + du

        # augmented cost
        aug_cost, viols = self.augmented.aug_step_cost(xp, up ,t)
        return  opt_cost, ctrl_cost, ilqr_cost.view(-1), aug_cost, viols

    def evaluate_trajectory_cost(self):
        x_plant_traj, u_plant_traj = self.plant_dyn.get_plant_rollouts()
        x_traj, u_traj = self.plant_dyn.get_rollouts()
        assert len(x_plant_traj) - 1 == len(u_plant_traj), f"Input array must be one shorter. {len(x_plant_traj) - 1} == {len(u_plant_traj)}"
        assert len(x_traj) - 1 == len(u_traj), f"Input array must be one shorter. {len(x_traj) - 1} == {len(u_traj)}"

        vals = (x_traj, u_traj+[None], x_plant_traj, u_plant_traj+[None])
        opt_c, ctr_c, lqr_c, aug_c, viols = zip(*[self.evaluate_step_cost(t, x, u, xp, up) for t, (x, u, xp, up) in enumerate(zip(*vals))])


        # Diffredmax
        deltas_squared = torch.stack([((x1-x0)/self.plant_dyn.plant.dt)**2 for x0, x1 in zip(x_traj[:-1], x_traj[1:])])
        opt_vel_c = 1e-6 * deltas_squared.sum()
        print("vel_cost:", opt_vel_c.item())

        return torch.stack(opt_c).sum() + opt_vel_c, torch.stack(lqr_c).sum(), torch.stack(aug_c).sum(), torch.stack(ctr_c).sum(), torch.stack(viols)

    def evaluate_total_cost(self):
        # Optim cost + controller cost + ilqr_increments cost
        opt_c, lqr_c, aug_c, ctrl_c, viols = self.evaluate_trajectory_cost()

        not_aug_c = opt_c + ctrl_c + lqr_c
        total_c = not_aug_c + aug_c
        return (
            (total_c).item(),
            not_aug_c.item(),
            aug_c.item(),
            viols
        )

    def solve(
        self,
        x0: BaseState,
        u_init: Union[list[torch.Tensor], list[BaseState]],
        constraints: Union[list[Constraint], None] = None,
        initializer: Union[BaseILQRDynSys, None] = None,
        initializer_u_init: Union[list[BaseState], None] = None,
        verbose=True,
    ):
        """
        Args:
            x0 (BaseState): initial state
            u_init (torch.Tensor): initial input trajectory
            constraints:    ineq_constraints on state/control; ineq_constraints(x, u, t) <= 0
            constraints:    eq_constraints on state/control; eq_constraints(x, u, t) == 0
        """
        # Prepare augmented AL
        if constraints is not None:
            self.augmented.add_constraints(constraints)

        self.print_info()

        _start_time = time.time()

        self.reg = self.conf.reg_init
        # Logging:
        self.ff_logs = {}

        # Initialize
        self.augmented.initialize(x0._base_tensor)

        if initializer is not None:
            assert initializer_u_init is not None, "If initializer is given make sure that its corresponding input is also inputed."
            initializer.set_step_cost(self.evaluate_step_cost)
            J_total_new, J_ilqr_new, J_aug_new, violations_new, lqr_sys_f = self.rollout(initializer, 1e10, [x0], initializer_u_init)
            _, u_init = initializer.get_plant_rollouts() # TODO: can also precompute derivatives, no need for second forw pass

        J_total_new, J_ilqr_new, J_aug_new, violations_new, lqr_sys_f = self.rollout(self.plant_dyn, 1e10, [x0], u_init)

        self.iteration_log(-1, -1, violations_new)

        # 0. Iterations
        self.J_hist_total = [J_total_new]
        self.J_hist_ilqr = [J_ilqr_new]
        self.J_hist_aug = [J_aug_new]

        _ii_max = self.augmented.conf.al_max_iter
        _i_max = self.conf.iter_max
        _i, _ii = 0, 0
        converged = False
        # accept = True
        for _ii in range(_ii_max):
            J_total, J_ilqr, J_aug, violations = self.evaluate_total_cost()
            reg_N = 0
            accept = True
            for _i in range(_i_max):
                print(
                    f"iLQR iteration: ------------------ Al-it: {_ii+1}/{_ii_max}  iLQR-it: {_i+1}/{_i_max} ------------------"
                )

                self.back_propagation(lqr_sys_f, accept)

                (
                    J_new_total,
                    J_new_ilqr,
                    J_new_aug,
                    violations_new,
                    lqr_sys_f,
                    accept,
                    converged,
                ) = self.forward_propagation(J_total, J_ilqr, J_aug)


                if accept:  # successful forward pass ilqr_accept_forward_pass
                    J_total = J_new_total
                    J_ilqr = J_new_ilqr
                    J_aug = J_new_aug
                    violations = violations_new

                    self.J_hist_total.append(J_total)
                    self.J_hist_ilqr.append(J_ilqr)
                    # self.J_hist_aug.append(J_aug)
                    self.J_hist_aug.append(violations.sum())

                    self.iteration_log(_i, _ii, violations)

                # Convergence and regularization
                if converged:  # iLQR converged, line_search is not improving that much ilqr_converged
                    if verbose:
                        print(
                            f"iLQR Converged at iteration {_i + 1}; J = {J_total}; J_aug = {self.J_hist_aug[-1]}; {reg_N}th reg = {self.reg}"
                        )
                    break

                if not accept:  # not a successful forward pass ilqr_max_reg_reached
                    if self.reg > self.conf.reg_max:
                        print(
                            "Exceeds regularization limit at iteration {0}; terminate the iterations of inner-loop".format(
                                _i + 1
                            )
                        )
                        break
                    if J_total > self.conf.max_cost: # ilqr_max_cost_exceeded
                        print(
                            f"Exceeds maximal allowed cost({J_total}>{self.conf.max_cost}); terminate the inner-loop"
                        )
                        break

                    # need to increase regularization
                    self.reg = self.reg * self.conf.reg_factor
                    reg_N += 1
                    if verbose:
                        print(
                            "Reject the control perturbation. Increase the regularization term."
                        )

            if not self.augmented.not_empty:
                print("No constraints, no AL iteration required.")
                break
            elif torch.all(violations < self.augmented.conf.al_cmax): # al_no_violation
                print("end_violation:", violations[-1])
                print("AL: Constraint tolerance met.")
                break
            elif torch.all(self.augmented.mu > self.augmented.conf.al_mu_max): # al_max_mu_reached
                print("AL: Maximal penalty:{self.augmented.conf.al_mu_max}, reached.")
                break
            print(
                f"Converged at iteration {_i + 1}; J = {J_total}; J_aug = {self.J_hist_aug[-1]}; reg = {self.reg}"
            )
            # Update lambda_al according to the x and u trajectories
            # and increase mu for the next iLQR iteration
            self.augmented.update_batch(*self.plant_dyn.get_plant_rollouts())

        # Final Log
        res_dict = self.final_log( _i, _ii, time.time()-_start_time, x0)
        return res_dict

    def forward_propagation(
        self,
        J_total: float,
        J_ilqr: float,
        J_aug: float,
    ):
        # Satisfy mypy
        # -------------
        J_new_total = J_total
        J_new_ilqr = J_ilqr
        J_new_aug = J_aug
        violations = torch.empty(0)
        lqr_sys = lambda: {'': [violations]}
        # -------------

        self.plant_dyn.backup_rollouts() # backup in case of linesearch failure

        x_array, u_array = self.plant_dyn.get_rollouts()
        xp_array, up_array = self.plant_dyn.get_plant_rollouts()
        self.prev_rollouts = ([x.detach() for x in x_array],
                              [u.detach() for u in u_array],
                              [x.detach() for x in xp_array],
                              [u.detach() for u in up_array]
                              ) # type: ignore

        accept = False  # whether forward pass was successfull
        converged = False  # whether iLQR converged
        # 1. Line search (continue after the first alpha that improves the trajectory loss)
        alpha = self.conf.alpha_init
        for jj in range(self.conf.iter_line_search):
            J_new_total, J_new_ilqr, J_new_aug, violations, lqr_sys = self.rollout(self.plant_dyn, J_total, x_array, u_array, alpha)

            delta_J_ratio = (J_new_total - J_total) / (
                self.delta_V_u * alpha + self.delta_V_uu * alpha**2 + 1e-6
            )
            # condition for line_search
            accept_armijo = (delta_J_ratio >= self.conf.beta_1) and (delta_J_ratio <= self.conf.beta_2) and (J_new_total - J_total)<0.
            accept = (J_new_total - J_total)< -0.001 # J_total/1000.

            print(
                f"\t{jj+1}th/{self.conf.iter_line_search} alpha={alpha:.5f} J_total(new/old)={J_new_total:.3f}/{J_total:.3f}, J_ilqr={J_new_ilqr:.3f}/{J_ilqr:.3f}, J_aug={J_new_aug:.3f}/{J_aug:.3f}, reg = {self.reg:.5f}"
            )
            print(f"\t    delta_J_ratio: {delta_J_ratio}: {J_new_total - J_total}/{self.delta_V_u * alpha + self.delta_V_uu * alpha**2}")
            print(f"\t    linesearch success: {accept}")

            if accept:
                # the rollouts are saved in the plant
                # successful step, decrease the regularization term, momentum like adaptive regularization
                self.reg = max([self.conf.reg_min, self.reg / self.conf.reg_factor])
                print(
                    f"  J = {J_new_total:.5f}; {jj}th/{self.conf.iter_line_search} alpha={alpha:.5f}; reg = {self.reg:.5f}"
                )
                break
            else:
                alpha = 0.5 * alpha
        if not accept:
            self.plant_dyn.reset_rollouts()

        # see if it is converged
        converged = (
            np.abs((J_new_total - J_total) / J_total) < self.conf.eps_cost
            )

        return (
            J_new_total,
            J_new_ilqr,
            J_new_aug,
            violations,
            lqr_sys,
            accept,
            converged,
        )

    def rollout(
        self,
        plant_dyn: BaseILQRDynSys,
        J_total: float,
        x_array: list[BaseState],
        u_array: Union[list[torch.Tensor], list[BaseState]],
        alpha=1.0,
    ):

        J_total_new, J_ilqr_new, J_aug_new, violations_new = plant_dyn.rollout(J_total, x_array, u_array, self.k_array, self.K_array, alpha)

        def deriv_dict():
            dfdx, dfdu, t1 = plant_dyn.comp_derivs(alpha=self.conf.smooth_grad_a)
            dldx, dldu, dldxx, dlduu, dldux, t2 = self.compute_loss_derivs()

            res_dict: dict[str, list] = {
                "dfdx": dfdx,
                "dfdu": dfdu,
                "dldx": dldx,
                "dldu": dldu,
                "dldxx": dldxx,
                "dlduu": dlduu,
                "dldux": dldux,
            }
            print(f"  - DynSys/Losses deriv times({t1:.5f}/{t2:.5f})")
            return res_dict

        return J_total_new.item(), J_ilqr_new.item(), J_aug_new.item(), violations_new, deriv_dict

    def compute_loss_derivs(self):
        _start2 = time.time()
        x_array, u_array = self.plant_dyn.get_rollouts()
        if not self.plant_dyn.requires_grad:
            x_array = [x.requires_grad_() for x in x_array]
            u_array = [u.requires_grad_() for u in u_array]
        dx = []
        du = []
        dxx = []
        duu = []
        dux = []

        opt_c, ctrl_c, lqr_c, aug_c, _ = self.evaluate_trajectory_cost()
        ll = opt_c + ctrl_c + lqr_c + aug_c

        ll.backward(retain_graph=True, create_graph=True)
        for i, (x, u) in enumerate(zip(x_array, u_array)):
            dx.append(x.grad)
            du.append(u.grad)
            dxx.append(dfdx_vmap(dx[-1], x))
            duu.append(dfdx_vmap(du[-1], u))
            dux.append(dfdx_vmap(du[-1], x, True))
        dx.append(x_array[-1].grad)
        dxx.append(dfdx_vmap(dx[-1], x_array[-1]))
        _end2 = time.time() - _start2
        return dx, du, dxx, duu, dux, _end2

    def compute_loss_derivs2(self):

        x_array, u_array = self.plant_dyn.get_rollouts()
        opt_c, ctrl_c, lqr_c, aug_c, _ = self.evaluate_trajectory_cost()
        ll = opt_c + ctrl_c + lqr_c + aug_c


        _start2 = time.time()
        # ll.backward(retain_graph=True, create_graph=True)
        # dx = [x.grad for x in x_array]
        # du = [u.grad for u in u_array]
        dx = [dfdx_vmap(ll, x, create_graph=True).view(-1) for i, x in enumerate(x_array)]
        du = [dfdx_vmap(ll, u, create_graph=True).view(-1) for i, u in enumerate(u_array)]
        dxx = [dfdx_vmap(dfdx, x) for i, (dfdx, x) in enumerate(zip(dx, x_array))]
        duu = [dfdx_vmap(dfdu, u, True) for i, (dfdu, u) in enumerate(zip(du, u_array))]
        # duu.append(torch.zeros_like(duu[-1]))
        dux = [dfdx_vmap(dfdu, x, True) for i, (dfdu, x) in enumerate(zip(du, x_array))]
        # dux.append(torch.zeros_like(dux[-1]))
        _end2 = time.time() - _start2

        # dx, du, dxx, duu, dux, _end2 = self.cost.comp_trajectory_derivs(x_array, u_array)

        return dx, du, dxx, duu, dux, _end2

    def back_propagation(self, lqr_sys_f, accept=True):
        """
        Back propagation along the given state and control trajectories to solve
        the Riccati equations for the error system (delta_x, delta_u, t)
        Need to approximate the dynamics/costs/constraints along the given trajectory
        dynamics needs a time-varying first-order approximation
        costs and constraints need time-varying second-order approximation

        lqr_sys: if called returns a dict with first order state derivatives df_dx, df_du
        and loss derivatives up to second order dl_dx, dl_du, dl_dxx, dl_duu, dl_duxs.

        In case we are using a closed loop approach the state is x_hat and includes bot the
        plants state and the state of the controller. On the other hand the input is
        the desired trajectory denoted by u_hat.

        So, in the case of closed loop constrained_ilqr x_array and u_array
        represent instead by x_hat_array and u_hat_array, i.e. rollout trajectories of the augmented state
        and contorller's input.


        To simplify setting up the problem we still allow to define the loss and constraints w.r.t
        the input & state of the plant.

        """
        lqr_sys: dict[str, list[torch.Tensor]] = (
            lqr_sys_f() if accept else self.lqr_sys_backup
        )
        x_array, u_array = self.plant_dyn.get_rollouts()
        x_plant_array, u_plant_array = self.plant_dyn.get_plant_rollouts()

        self.k_array = []
        self.K_array = []
        _start = time.time()
        with torch.no_grad():

            # initialize with the terminal cost parameters to prepare the backpropagation
            Vx = lqr_sys["dldx"][-1].view(-1, 1)
            Vxx = lqr_sys["dldxx"][-1]
            # (Vx, _, Vxx, _, _,) = self.augmented.augment_t(
            #     len(u_array),
            #     Vx,
            #     0,
            #     Vxx,
            #     0,
            #     0,
            #     x_array[-1],
            #     u_array[-1],
            #     x_plant_array[-1],
            #     u_plant_array[-1],
            # )
            self.delta_V_u = 0.0
            self.delta_V_uu = 0.0
            for t in reversed(range(self.T)):
                Qx = lqr_sys["dldx"][t].view(-1, 1) + lqr_sys["dfdx"][t].T @ (Vx)
                Qu = lqr_sys["dldu"][t].view(-1, 1) + lqr_sys["dfdu"][t].T @ (Vx)
                Qxx = lqr_sys["dldxx"][t] + lqr_sys["dfdx"][t].T @ (Vxx) @ (
                    lqr_sys["dfdx"][t]
                )
                Quu = lqr_sys["dlduu"][t] + lqr_sys["dfdu"][t].T @ (Vxx) @ (
                    lqr_sys["dfdu"][t]
                )
                Qux = lqr_sys["dldux"][t] + lqr_sys["dfdu"][t].T @ (Vxx) @ (
                    lqr_sys["dfdx"][t]
                )

                # Augmented Lagrangian
                x_t, u_t = x_array[t], u_array[t]
                x_plant_t, u_plant_t = x_plant_array[t], u_plant_array[t]
                # (Qx, Qu, Qxx, Quu, Qux,) = self.augmented.augment_t(
                #     t, Qx, Qu, Qxx, Quu, Qux, x_t, u_t, x_plant_t, u_plant_t
                # )

                # Solve quadratic problem to find k and K
                # -------------------------
                # k = Quu-1 Qu, K_ = -Quu-1 Qux
                _k, K_ = qp_solve(
                    u_t,  # only for the boxlqr case
                    Quu,
                    Qu,
                    Qux,
                    limits=self.limits,
                    reg=self.reg,
                    _k0=None if len(self.k_array) == 0 else self.k_array[-1],
                    verbose=False,
                )
                # -------------------------

                self.k_array.append(_k)
                self.K_array.append(K_)

                # update value function for the previous time step
                Vx = Qx + K_.T @ (Quu) @ (_k) + K_.T @ Qu + Qux.T @ _k
                Vxx = Qxx + K_.T @ (Quu) @ (K_) + K_.T @ Qux + Qux.T @ K_
                # Vxx = 0.5 * (Vxx + Vxx.T)
                self.delta_V_u += (_k.T @ Qu).item()
                self.delta_V_uu += (_k.T @ Quu @ _k).item()

        self.k_array.reverse()
        self.K_array.reverse()

        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        # DEBUG PLOTS
        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        if False:
            import matplotlib.pyplot as plt
            SAVE = False
            _log_dir = "/tmp/" if self.log_dir is None else self.log_dir
            plt.rcParams["axes.grid"] = False
            plt.rcParams['lines.linewidth'] = 1.
            ixss = list(set(self.plant_dyn.plant.ixs_controlled + self.plant_dyn.plant.ixs_goal))
            N_bodies_plot = len(ixss)
            n_bodies = self.plant_dyn.plant.n_bodies

            x0 = x_array[0].detach()

            # 1. Before and desired
            x_before_array = np.stack([x.detach().numpy() for x in x_array]) # x0 -- xN
            x_desired = np.stack([x.detach().numpy() for x in self.cost.x_des])

            # 2. Feedforward
            u_feedforward_impact = torch.stack([Jxu @ k for Jxu, k in zip(lqr_sys["dfdu"], self.k_array)]) # u0 -- uN
            u_feedforward_accumulated_impact  = torch.cumsum(u_feedforward_impact, dim=0).view(self.T, n_bodies, -1) #u_acc_i = sum_i(ui)
            x_feedforward_expected = [x0] + [x.detach().increment(u_impact) for x, u_impact in zip(x_array, u_feedforward_impact)]  # x0 -- xN+1
            x_after_feedforward_impacted_array = np.stack([x.numpy() for x in x_feedforward_expected]) # x1 -- xN+1

            # 3. Linear System
            delta_x_array = [x0._base_tensor.new_zeros(x_array[0].size)]
            delta_u_array = []
            i=0
            for Jxx, Jxu, k, K, x_old, u_old in zip(lqr_sys["dfdx"], lqr_sys["dfdu"], self.k_array, self.K_array, x_array, u_array):

                delta_x = delta_x_array[-1]
                delta_u = k.view(-1) + K @ delta_x
                delta_x_next = Jxx @ delta_x + Jxu @ delta_u

                delta_x_array.append(delta_x_next)
                delta_u_array.append(delta_u)

                i=+1

            x_linear_expected = np.stack([x.detach().increment(delta_x).numpy() for x, delta_x in zip(x_array, delta_x_array)])
            u_linear_impact = torch.stack([Jxu @ delta_u for Jxu, delta_u in zip(lqr_sys["dfdu"], delta_u_array)], dim=0)
            u_linear_accumulated_impact  = torch.cumsum(u_linear_impact, dim=0).view(self.T, n_bodies, -1) #u_acc_i = sum_i(ui)


            # PLOT
            dnames = ['dx', 'dy', 'dz', 'dvx', 'dvy', 'dvz', 'drx', 'dry', 'drz', 'dw_x', 'dw_y', 'dw_z' ]
            names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'rx', 'ry', 'rz', 'rw', 'w_x', 'w_y', 'w_z']


            x_size = u_feedforward_accumulated_impact.shape[-1]
            cols = 3
            rows = (x_size // cols) + (x_size % cols)


            # ---------------------------------------------------------------------------------------------------
            # ---------------------------------------------------------------------------------------------------
            # svd_xu = np.stack([np.linalg.svd(Jxu.detach().numpy())[1] for Jxu in lqr_sys["dfdu"]])
            # svd_xx = np.stack([np.linalg.svd(Jxx.detach().numpy())[1] for Jxx in lqr_sys["dfdx"]])


            # _ixs_xu = []
            # _svd_xu = []
            # for t, Jxu in enumerate(lqr_sys["dfdu"]):
            #     u, s, vh = np.linalg.svd(Jxu.detach().numpy())
            #     _ixs_xu.append(np.argmax(np.abs(vh), 1))
            #     _svd_xu.append(s)
            #     # print(f"{t}: singular values: {s}")
            #     # print(f"\t uncontrolled {u[:,6:]}")

            # _svd_xu_sorted = np.stack(s_xu[ix] for s_xu, ix in zip(_svd_xu, _ixs_xu))

            # plt.figure(figsize=(20, 12))
            # for i in range(6):
            #     plt.subplot(2,3,i+1)
            #     plt.plot(svd_xu[:,i], label=f"u_{i}")
            #     plt.ylim(-1, 1)
            #     plt.legend()

            # plt.figure(figsize=(20, 12))
            # for i in range(12):
            #     plt.subplot(6,2,i+1)
            #     plt.plot(svd_xx[:,i], label=f"x_{i}")
            #     plt.ylim(-1.2, 1.2)
            #     plt.legend()
            # ---------------------------------------------------------------------------------------------------
            # ---------------------------------------------------------------------------------------------------


            plt.figure(figsize=(20, 12))
            plt.suptitle(f"u impact to: x")
            for i in range(N_bodies_plot):
                for r in range(rows):
                    for c in range(cols):
                        ix = r * cols + c
                        print(r,c)
                        plt.subplot(rows,cols,ix+1)
                        plt.plot(u_feedforward_accumulated_impact[:, ixss[i], ix].numpy(), label=f"body_{ixss[i]}_feedforward")
                        plt.plot(u_linear_accumulated_impact[:, ixss[i], ix].numpy(), label=f"body_{ixss[i]}_linear")
                        plt.title(f"u impact to: {dnames[ix]}")
                        plt.legend()
            if SAVE:
                plt.savefig(fname=_log_dir+f"Predicted input inpact to components of the state")

            plt.figure(figsize=(20, 12))
            plt.subplots_adjust(left=0.2)
            plt.subplots_adjust(right=0.8)
            plt.suptitle(f"Predicted states from the linearized system")
            for i in range(N_bodies_plot):
                plt.subplot(3,N_bodies_plot,i+1)
                plt.title("Body: " + str(ixss[i]))
                for jj in range(3):
                    plt.subplot(3,N_bodies_plot,jj*N_bodies_plot+i+1)
                    plt.plot(x_after_feedforward_impacted_array[:,ixss[i], jj], label=f"body_{ixss[i]}: {names[jj]}_feedforward_expected", alpha=0.5)
                    plt.plot(x_linear_expected[:,ixss[i], jj], label=f"body_{ixss[i]}: {names[jj]}_linear_expected", alpha=0.5)
                    plt.plot(x_before_array[:,ixss[i], jj], label=f"body_{ixss[i]}: {names[jj]}_before", linestyle='dashed')
                    plt.plot(x_desired[:,ixss[i], jj], label=f"body_{ixss[i]}: {names[jj]}_desired", linestyle=':')

                    if i ==0:
                        plt.legend(loc='center right', bbox_to_anchor=(-0.1, 0.5), borderaxespad=0)
                    else:
                        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0)

            if SAVE:
                plt.savefig(fname=_log_dir+f"Predicted Body-pose influence")
            if not SAVE:
                plt.show()


        self.lqr_sys_backup = lqr_sys
        # self.log_data(lqr_sys, 'lqr_sys')
        print(f"  - Back-propogation({time.time() - _start:.5f})")


    def log_data(self, data, name):
        _log_dir = "/tmp/" if self.log_dir is None else self.log_dir
        with open(_log_dir + name + ".p", "wb") as f:
            pickle.dump(data, f)

    def final_log(self, _i, _ii, time_taken, x0):
        nr_iters = (_ii + 1) * (_i + 1)
        # prepare result dictionary
        xp_array, up_array = self.plant_dyn.get_plant_rollouts()
        x_hat_arr, u_hat_arr = self.plant_dyn.get_rollouts()
        res_dict = {
            "feedback_ctrl": self.plant_dyn.feedback_controlled,
            "x0": x0,
            "x_plant_array_opt": np.stack([x.numpy() for x in xp_array]),
            "u_plant_array_opt": torch.stack(up_array).cpu().detach().numpy(),
            "x_hat_array_opt": [x.clone() for x in x_hat_arr],
            "u_hat_array_opt": [u.clone() for u in u_hat_arr],
            "J_hist_total": np.array(self.J_hist_total),
            "J_hist_ilqr": np.array(self.J_hist_ilqr),
            "J_hist_aug": np.array(self.J_hist_aug),
            "k_array_opt": torch.stack(self.k_array).cpu().detach().numpy(),
            "K_array_opt": torch.stack(self.K_array).cpu().detach().numpy(),
            "constraints": {},
            "constraints_info": [c.get_log() for c in self.augmented.constraints],
            "ff_logs": self.ff_logs,
            "time": time_taken,
            "time_iteration": time_taken / nr_iters,
            "reg": self.reg,
        }
        for i, C in enumerate(self.augmented.constraints):
            costs = (
                torch.cat(
                    [
                        C(x, u, t).detach()
                        for t, x, u in zip(range(self.T), xp_array, up_array)
                    ]
                )
                .cpu()
                .numpy()
            )
            res_dict["constraints"][f'{i}_{"eq" if C.type == C.EQ else "ineq"}'] = costs
        return res_dict

    def iteration_log(self, _i, _ii, violations):
        xp_array, up_array = self.plant_dyn.get_plant_rollouts()
        x_hat_arr, u_hat_arr = self.plant_dyn.get_rollouts()
        if _ii not in self.ff_logs.keys():
            self.ff_logs[_ii] = {}
        self.ff_logs[_ii][_i] = {
            "violations": violations.cpu().detach().numpy(),
            "x_plant_array_opt": np.stack([x.detach().numpy() for x in xp_array]),
            "u_plant_array_opt": torch.stack(up_array).cpu().detach().numpy(),
            "x_hat_array_opt": [x.clone() for x in x_hat_arr],
            "u_hat_array_opt": np.stack([u.detach().numpy() for u in u_hat_arr]),
            # "k_array": torch.stack(self.k_array).cpu().detach().numpy(),
            # "K_array": torch.stack(self.K_array).cpu().detach().numpy(),
            }

    def print_info(self):
        print(f"----------------------------------------------------------------------------")
        print(f" {col.HEADER}Solving iLQR{col.ENDC} with T: {self.T}, nx: {self.plant_dyn.plant.nx}, nu:, {self.plant_dyn.plant.nu}")
        print()
        print(f"--- {col.HEADER}iLQR Info{col.ENDC}---")
        print_dict(self.conf, 0)
        print()
        self.augmented.print_info()
        print()
        self.cost.print_info()
        print()
        self.plant_dyn.print_info()
        print(f"----------------------------------------------------------------------------")

def qp_solve(
    u,
    Quu: torch.Tensor,
    Qu: torch.Tensor,
    Qux: torch.Tensor,
    limits,
    reg: float,
    _k0: Union[torch.Tensor, None] = None,
    verbose=False,
):
    """
    min_u u*Quu*u + Qu*u + u*Qux*x ->  u* = k + K*x
    """
    if limits is None:
        inv_Quu = regularized_persudo_inverse_(Quu, reg=reg)
        _k = -inv_Quu @ (Qu)
        K_ = -inv_Quu @ (Qux)
    else:
        # _k0 = _k0.cpu().numpy() if _k0 is not None else None
        # Quu, Qu, b_lower, b_upper = Quu.cpu().numpy(), Qu.cpu().numpy(), limits[0]-u.cpu().numpy(), limits[1]-u.cpu().numpy()
        b_lower, b_upper = limits[0] - u, limits[1] - u
        K_ = torch.zeros_like(Qux)

        _k, Qff_inv, free_ix = box_QP(
            Quu, Qu, b_lower, b_upper, _k0, reg=reg, verbose=verbose
        )

        # _k = torch.from_numpy(_k).to(Quu).view(-1,1)
        # K_[free_ix] = torch.from_numpy(-Qff_inv @ Qux[free_ix].cpu().numpy()).to(K_)

        K_[free_ix] = -Qff_inv @ Qux[free_ix]
    # print(torch.allclose(_k.view(-1,1), -inv_Quu @ (Qu)), torch.allclose(K_, -inv_Quu @ (Qux)))
    # print((_k.view(-1,1)+inv_Quu @ (Qu)).T)
    # print(all(_k+u<=limits[1]))
    # print(all(_k+u>=limits[0]))
    return _k.view(-1, 1), K_


def box_QP(
    Q: torch.Tensor,
    q: torch.Tensor,
    b_lower: torch.Tensor,
    b_upper: torch.Tensor,
    x: Union[torch.Tensor, None] = None,
    reg=1e-5,
    verbose=False,
):
    """Projected-Newton QP Solution

            min_x  f(x) = 0.5 x^T*Q*x + q^T*x
            s.t.    b_lower <= x <= b_upper

    Args:
            H (nxn): hessian
            q (nx1): gradient
            b_lower/b_upper (nx1): low,upper boundary
            x (nx1): warm start
    """

    q, b_lower, b_upper = (
        torch.squeeze(q),
        torch.squeeze(b_lower),
        torch.squeeze(b_upper),
    )
    if x is None:
        # x = torch.zeros_like(q)
        x = 0.5 * (b_lower + b_upper)
    x = torch.squeeze(x)
    # f = lambda x: (1/2)*x.T @ (Q @ x) + q.T @ x
    f = lambda x: (1 / 2) * x @ (Q @ x) + q @ x

    converged = False
    alpha_array = 1.1 ** (-torch.arange(15) ** 2)

    # TODO: not sure if we can compute the cholesky factorization only once
    #       and then change it through iterations
    # VT, diag_s_inv, UT = regularized_persudo_inverse_(Q, reg, True)

    it = 0
    J_s_opt = f(x)

    all_ix_s = set(range(x.numel()))

    # Satisfz mypy
    # ---------------
    fail_linesearch = True
    alpha = alpha_array[0]
    j = 0
    Qff_inv = torch.empty(0)
    free_ix = []
    # ---------------
    while not converged and it < 200:
        # 1. Get indexes
        grad = q + Q @ x
        clamped_ix_s = set(
            np.where(
                torch.logical_or(
                    torch.logical_and(x <= b_lower + 1e-5, grad > 0),
                    torch.logical_and(x >= b_upper - 1e-5, grad < 0),
                ).tolist()
            )[0]
        )
        free_ix_s = all_ix_s - clamped_ix_s
        free_ix, clamped_ix = list(free_ix_s), list(clamped_ix_s)

        xf, xc, qf = x[free_ix], x[clamped_ix], q[free_ix]
        Qff, Qfc = Q[free_ix][:, free_ix], Q[free_ix][:, clamped_ix]

        Qff_inv = regularized_persudo_inverse_(Qff, reg)
        delta_x_f = -Qff_inv @ (qf + Qfc @ xc) - xf
        delta_x = -torch.clone(grad)
        delta_x[free_ix] = delta_x_f

        # TODO
        # Q_tmp, diag_s_inv_tmp = torch.clone(Q), torch.clone(diag_s_inv)
        # Q_tmp[clamped_ix] = 0
        # diag_s_inv_tmp[clamped_ix] = 0
        # delta_x_2 = -(VT @ diag_s_inv_tmp.diag() @ UT) @ (q + Q_tmp @ x)

        # assert grad.T @ delta_x <=1e-7
        fail_linesearch = True
        for j, alpha in enumerate(alpha_array):
            x_new = torch.clamp(x + alpha * delta_x, min=b_lower, max=b_upper)
            J_new = f(x_new)
            if J_s_opt >= J_new:
                converged = np.abs((J_s_opt - J_new) / J_s_opt) < 1e-6
                x = x_new
                J_s_opt = J_new
                fail_linesearch = False
                break
        it += 1
        if fail_linesearch:
            break
    if verbose:
        print(
            f"Did {'not ' if not converged or fail_linesearch else ''}converge after {it} iterations and {j}th alpha={alpha} with loss {J_s_opt} and reg={reg}."
        )
    return x, Qff_inv, free_ix

@torch.no_grad()
def regularized_persudo_inverse_(
    mat: torch.Tensor,
    reg,  # split: bool = False
) -> torch.Tensor:
    """
    Use SVD to realize persudo inverse by perturbing the singularity values
    to ensure its positive-definite properties
    """
    # if split:
    #     u, s, v = torch.linalg.svd(mat)
    #     s[s < 0] = 0.0  # truncate negative values...
    #     # diag_s_inv = mat.new_zeros((v.shape[0], u.shape[1]))
    #     diag_s_inv = 1.0 / (s + reg)
    #     return v.T, diag_s_inv, u.T
    # return v @ (diag_s_inv.diag()) @ (u.T)

    # if not NPD.isPD(mat.numpy()):
    #     print("MAT is not positive definite")
    #     mat = torch.tensor(NPD.nearestPD(mat.numpy()))

    return torch.linalg.pinv(mat + torch.diag((torch.diag(mat) * 0.0 + reg)), hermitian=True)
    # return torch.linalg.inv(mat + torch.diag((torch.diag(mat) * 0.0 + reg)))
    # return torch.linalg.pinv(mat, atol=reg*1e-4, hermitian=True)
    # return torch.linalg.inv(mat + torch.diag((torch.diag(mat) * 0.0 + reg)))
