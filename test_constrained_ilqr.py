"""
An Inverted Pendulum test for the AL-iLQR implementation
"""
from __future__ import print_function, annotations
import torch
import matplotlib.pyplot as plt
from typing import Union

from al_ilqr.helpers import (
    DynamicSystem,
    LossFunctionBase,
    Controller,
    Optim,
)
from al_ilqr.state import BasicState, StackedState
from al_ilqr.constraints import EqConstraint, InEqConstraint, Constraint
from al_ilqr.utils import dfdx_vmap

class MyLossAuto(LossFunctionBase):
    def __init__(self, T) -> None:
        super().__init__(1, 1)
        self.T = T
        self.Q = 100
        self.R = 0.01

        # terminal Q to regularize final speed
        self.Q_T = 0.1

    def evaluate(
        self,
        x: BasicState,
        u: torch.Tensor,
        t: int,
        terminal=False
    ) -> torch.Tensor:
        if terminal:
            return (x[0] - torch.pi) ** 2 * self.Q + x[1] ** 2 * self.Q_T
        else:
            return (x[0] - torch.pi) ** 2 * self.Q + u[0] ** 2 * self.R


class MyLoss(LossFunctionBase):
    def __init__(self, T, test: bool = True) -> None:
        super().__init__(1, 1)
        self.test = test
        self.T = T
        self.Q = 100.0
        self.R = 0.01

        # terminal Q to regularize final speed
        self.Q_T = 0.1

    def evaluate(
        self,
        x: BasicState,
        u: torch.Tensor,
        t: int,
        terminal=False
    ) -> torch.Tensor:
        if terminal:
            return (x[0] - torch.pi) ** 2 * self.Q + x[1] ** 2 * self.Q_T
        else:
            return (x[0] - torch.pi) ** 2 * self.Q + u[0] ** 2 * self.R

    def _dldx(self, x: BasicState, u: torch.Tensor, t: int, terminal=False):
        if terminal:
            dldx = torch.tensor([2 * (x[0] - torch.pi) * self.Q, 2 * x[1] * self.Q_T]).T
        else:
            # terminal cost
            dldx = torch.tensor([2 * (x[0] - torch.pi) * self.Q, 0]).T

        if self.test:
            _x = x.detach().requires_grad_()
            autograd_dldx = dfdx_vmap(
                self.evaluate(_x, u, t).view(-1), _x.d_in()
            ).detach()
            assert torch.allclose(dldx, autograd_dldx), f"{dldx} \n {autograd_dldx}"
        return dldx

    def _dldu(self, x: BasicState, u: torch.Tensor, t: int, terminal=False):
        dldu = torch.tensor([2 * u[0] * self.R])
        if self.test:
            _u = u.detach().requires_grad_()
            autograd_dldu = dfdx_vmap(self.evaluate(x, _u, t, terminal).view(-1), _u).detach()
            assert torch.allclose(dldu, autograd_dldu), f"{dldu} \n {autograd_dldu}"
        return dldu

    def _dldxx(
        self,
        dldx: torch.Tensor,
        x: BasicState,
        u: torch.Tensor,
        t: int,
        terminal=False
    ):
        if terminal:
            dldxx = torch.tensor([[2.0 * self.Q, 0.0], [0, 2.0 * x[1] * self.Q_T]])
        else:
            dldxx = torch.tensor([[2.0, 0.0], [0.0, 0.0]]) * self.Q
        return dldxx

    def _dlduu(
        self,
        dldu: torch.Tensor,
        x: BasicState,
        u: torch.Tensor,
        t: int,
        terminal=False
    ):
        dlduu = torch.tensor([[2 * self.R]])
        return dlduu

    def _dldux(
        self,
        dldu: torch.Tensor,
        x: BasicState,
        u: torch.Tensor,
        t: int,
        terminal=False
    ):
        dldux = torch.tensor([0, 0])
        return dldux


class InvPendulumDynSysAuto(DynamicSystem):
    def __init__(self):
        # parameters
        self.m_ = 1
        self.l_ = 0.5
        self.b_ = 0.1
        self.lc_ = 0.5
        self.g_ = 9.81
        self.dt_ = 0.01

        self.I_ = self.m_ * self.l_**2
        super().__init__(self.dt_, 2, 1)

    def state(self):
        pass

    def step_abs(self, x: BasicState, u, t):
        # torch.autograd.set_detect_anomaly(True)
        m_ = 1
        l_ = 0.5
        b_ = 0.1
        lc_ = 0.5
        g_ = 9.81
        dt_ = 0.01

        I_ = m_ * l_**2
        xdd = (u[0] - m_ * g_ * lc_ * torch.sin(x.val[0]) - b_ * x.val[1]) / I_

        # dont need term +0.5*(qdd**2)*self.dt_?
        # x_new = x.clone()
        x_new = BasicState(torch.zeros_like(x.val))
        tmp = torch.stack([x[1], xdd])
        x_new.val = x.val + dt_ * tmp
        # x_new = x.clone()
        # x_new.val[0] = x.val[0] + dt_ * x.val[1]
        # x_new.val[1] = x.val[1] + dt_ * xdd
        return x_new


class InvPendulumDynSys(DynamicSystem):
    def __init__(self, test: bool = False):
        # parameters
        self.m_ = 1
        self.l_ = 0.5
        self.b_ = 0.1
        self.lc_ = 0.5
        self.g_ = 9.81
        self.dt_ = 0.01

        self.I_ = self.m_ * self.l_**2
        self.test = test
        super().__init__(self.dt_, 2, 1)

    def state(self):
        pass

    def step_abs(self, x: BasicState, u, t):
        m_ = 1
        l_ = 0.5
        b_ = 0.1
        lc_ = 0.5
        g_ = 9.81
        dt_ = 0.01

        I_ = m_ * l_**2
        xdd = (u[0] - m_ * g_ * lc_ * torch.sin(x.val[0]) - b_ * x.val[1]) / I_

        # dont need term +0.5*(qdd**2)*self.dt_?
        # x_new =  x + dt_ * torch.tensor([x[1], xdd])
        x_new = x.clone()
        x_new.val[0] = x.val[0] + dt_ * x.val[1]
        x_new.val[1] = x.val[1] + dt_ * xdd
        return x_new

    def step_abs2(self, x: BasicState, u, t):
        xdd = (
            u[0]
            - self.m_ * self.g_ * self.lc_ * torch.sin(x.val[0])
            - self.b_ * x.val[1]
        ) / self.I_

        # dont need term +0.5*(qdd**2)*self.dt_?
        x_new = x.clone()
        x_new.val = x.val + self.dt_ * torch.tensor([x.val[1], xdd])
        return x_new

    def dfdx(self, x_1, x, u, t):
        dfdx = torch.tensor(
            [
                [1, self.dt_],
                [
                    -self.m_
                    * self.g_
                    * self.lc_
                    * torch.cos(x[0])
                    * self.dt_
                    / self.I_,
                    1 - self.b_ * self.dt_ / self.I_,
                ],
            ]
        )

        if self.test:
            _x = x.detach().requires_grad_()
            autograd_dfdx = dfdx_vmap(self.step(_x, u, t).d_out(), _x.d_in()).detach()
            assert torch.allclose(dfdx, autograd_dfdx), f"{dfdx} \n {autograd_dfdx}"
        return dfdx

    def dfdu(self, x_1, x, u, t):
        dfdu = torch.tensor([[0], [self.dt_ / self.I_]])

        if self.test:
            _u = u.detach().requires_grad_()
            autograd_dfdu = dfdx_vmap(self.step(x, _u, t).d_out(), _u).detach()
            assert torch.allclose(dfdu, autograd_dfdu), f"{dfdu} \n {autograd_dfdu}"
        return dfdu


class PIController(Controller):
    def __init__(self, base_tensor: torch.Tensor) -> None:
        self.ki = 0.004 * base_tensor.new_ones(2)
        self.kp = 2.0 * base_tensor.new_ones(2)
        super().__init__(base_tensor)

    def init_state(self):
        return BasicState(self.base_tensor.new_zeros(2))

    def _compute_input(self, err: torch.Tensor, x: BasicState, x_c: BasicState):
        return self.ki @ x_c.val + self.kp @ err

    def _update_state(self, err, x_c: BasicState):
        self.state = x_c.clone().increment(err)
        return self.state

    def _get_err(self, x_des: BasicState, x: BasicState):
        return x_des.diff(x)

    def state_derivatives(self, x_des, x, x_c, req_partial_grad=False):
        dxc_dxc = dfdx_vmap(self.state, x_c)
        dxc_dxdes = dfdx_vmap(self.state, x_des)
        dxc_dx = dfdx_vmap(self.state, x)
        return dxc_dxc, dxc_dx, dxc_dxdes

    def input_derivatives(self, u, x_des, x, x_c, req_partial_grad=False):
        du_dxc = dfdx_vmap(u, x_c)
        du_dxdes = dfdx_vmap(u, x_des)
        du_dx = dfdx_vmap(u, x)
        return du_dxc, du_dx, du_dxdes


def plot_ilqr_result(res):
    if res is not None:
        # draw cost evolution and phase chart
        dt = res["dt"]
        fig = plt.figure(figsize=(16, 8), dpi=80)
        ax_cost = fig.add_subplot(231)
        n_itrs = len(res["J_hist_total"])
        ax_cost.plot(torch.arange(n_itrs), res["J_hist_total"], "r", linewidth=3.5)
        ax_cost.set_xlabel("Number of Iterations", fontsize=20)
        ax_cost.set_ylabel("Trajectory Cost")

        ax_input = fig.add_subplot(232)
        n_steps = len(res["u_plant_array_opt"])
        ax_input.plot(
            torch.arange(n_steps) * dt,
            res["u_plant_array_opt"],
            "b",
            linewidth=3.5,
        )
        ax_input.set_xlabel("Time step", fontsize=20)
        ax_input.set_ylabel("Input trajectory")

        ax_phase = fig.add_subplot(233)
        theta = res["x_plant_array_opt"][:, 0]
        theta_dot = res["x_plant_array_opt"][:, 1]
        ax_phase.plot(theta, theta_dot, "k", linewidth=3.5)
        ax_phase.set_xlabel("theta (rad)", fontsize=20)
        ax_phase.set_ylabel("theta_dot (rad/s)", fontsize=20)
        ax_phase.set_title("Phase Plot", fontsize=20)

        ax_phase.plot([theta[-1]], [theta_dot[-1]], "b*", markersize=16)
        ax_input = fig.add_subplot(234)
        n_steps = len(res["x_plant_array_opt"][:, 0])
        ax_input.plot(
            torch.arange(n_steps) * dt,
            res["x_plant_array_opt"][:, 0],
            label="theta",
        )
        ax_input.plot(
            torch.arange(n_steps) * dt,
            res["x_plant_array_opt"][:, 1],
            label="theta_dot",
        )
        ax_input.legend()
        ax_input.set_xlabel("Time step", fontsize=20)
        ax_input.set_ylabel("OUt 1 trajectory")


if __name__ == "__main__":
    T = 150
    analytical = False
    test = False  # for this, analytical has to also be yes
    controlled = True

    # --------------------------------------------------------------------------------------
    # 1. Building
    # --------------------------------------------------------------------------------------

    # --# --------------------------------------------------------------------------------------
    # --# Create dynamic system and loss
    # --# --------------------------------------------------------------------------------------
    dyn_sys = InvPendulumDynSys(test) if analytical else InvPendulumDynSysAuto()
    my_controller: Union[Controller, None] = (
        PIController(torch.empty(1)) if controlled else None # TODO
    )
    loss = MyLoss(T, test) if analytical else MyLossAuto(T)

    # --# --------------------------------------------------------------------------------------
    # --# Build LQR problem
    # --# --------------------------------------------------------------------------------------
    problem = Optim()
    problem.build_ilqr_problem(T, dyn_sys, loss, my_controller)

    # --------------------------------------------------------------------------------------
    # 2. Solving
    # --------------------------------------------------------------------------------------

    # --# --------------------------------------------------------------------------------------
    # --# Define desired trajectory
    # --# --------------------------------------------------------------------------------------
    x0 = BasicState(torch.tensor([torch.pi - 2, 0]))
    x_des_traj = [BasicState(torch.tensor([torch.pi / 2.0, torch.pi / 2.0]))] * T
    my_constraints: list[Constraint] = []

    def eq_c(x: BasicState, u, t):
        if t >= T // 3 and t < T // 2:
            return x.diff(x_des_traj[0]).view(-1)  # end state constraint
        else:
            return x.val.new_zeros(2)

    my_constr = EqConstraint()
    my_constr.build(N_out=2, c=eq_c)
    my_constraints.append(my_constr)

    def ineq_c(x: BasicState, u, t):
        return (x.val[1] - 15.0).view(-1)  # end state constraint

    my_constr2 = InEqConstraint()
    my_constr2.build(N_out=1, c=ineq_c)
    my_constraints.append(my_constr2)
    # --# --------------------------------------------------------------------------------------

    # prepare initial guess
    u_init_traj: Union[list[torch.Tensor], list[BasicState]]
    if my_controller is not None:
        x0 = StackedState([x0, my_controller.init_state()])
        # u_init_traj = [u.detach().clone() for u in x_des_traj[:-1]]
        u_init_traj = [x_des_traj[-1].detach()]*T
    else:
        u_init_traj = list(
            torch.rand(
                (T, problem.ilqr_dyn_sys.plant.nu), # dyn_sys.nu),
                dtype=x0._base_tensor.dtype,
                device=x0._base_tensor.device,
            )
        )


    # --# --------------------------------------------------------------------------------------
    # --# Solve iLQR
    # --# --------------------------------------------------------------------------------------
    res = problem.solve_ilqr_problem(
        x0, x_des_traj, u_init_traj, constraints=my_constraints, verbose=True  # type: ignore
    )

    # --------------------------------------------------------------------------------------
    # 3. Visualize
    # --------------------------------------------------------------------------------------
    print(
        ("Analytical derivatives" if analytical else "Autodifferentiation derivatives")
        + f" in {res['time']} seconds, time_iteration: {res['time_iteration']}"
    )
    plot_ilqr_result(res)
    plt.show()
