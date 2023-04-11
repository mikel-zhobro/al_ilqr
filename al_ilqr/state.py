from __future__ import print_function, annotations
from abc import ABC, abstractmethod

import torch
import numpy as np
from pytorch3d.transforms import so3_log_map, so3_exp_map, matrix_to_quaternion


def get_body_state(state: MultiBodyState, ixs: list[int]):
    obj = MultiBodyState(
        state.R[ixs],
        state.p[ixs],
        state.v_w[ixs],
        state.requires_grad,
    )
    return obj

# --------------------------------------------------------------------------------------
# State definition
# --------------------------------------------------------------------------------------
class BaseState(ABC):
    def __init__(self, base_tensor: torch.Tensor, requires_grad=False):
        # required for backprop
        self._requires_grad = False  # it is setted below, together with elements
        self._base_tensor = base_tensor
        self._din: torch.Tensor = base_tensor.new_empty(0)
        self.requires_grad = requires_grad

    # Pytorch-Tensor like properties
    # ----------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    def grad(self):
        return self.d_in().grad

    def requires_grad_(self, req_grad=True):
        # sets requires_grad to True
        self.requires_grad = req_grad
        return self

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._requires_grad = val
        self.requires_grad_setter(val)

    @abstractmethod
    def _set_din(self):
        pass

    def requires_grad_setter(self, val: bool):
        if val:
            self._set_din()
            self.increment(self._din)

    @abstractmethod
    def detach(self) -> BaseState:
        """ """
        pass

    @abstractmethod
    def clone(self) -> BaseState:
        pass

    # Required for gradient calculation
    # ----------------------------------------------------------------------------------------------
    def d_in(self):
        return self._din

    @abstractmethod
    def d_out(self) -> torch.Tensor:
        pass

    # Difference
    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def increment(self, dstate)->BaseState:
        pass

    @abstractmethod
    def diff(self, other_state) -> torch.Tensor:
        pass

    def __sub__(self, other_state):
        return self.diff(other_state)

    def __add__(self, dstate):
        assert not isinstance(
            dstate, BaseState
        ), "Additon is not defined for between states but between a state and a diff."
        obj = self.clone()
        obj.increment(dstate)
        return obj

    # Numpy
    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def numpy(self) -> np.ndarray:
        pass


class StackedState(BaseState):
    def __init__(self, states_list: list[BaseState], requires_grad=False) -> None:
        self.states_list = states_list
        super().__init__(states_list[0]._base_tensor, requires_grad)

    # Pytorch-Tensor like properties
    # ----------------------------------------------------------------------------------------------
    @property
    def size(self):
        return sum([s.size for s in self.states_list])

    def _set_din(self):
        self._din = self._base_tensor.new_zeros(self.size).requires_grad_()
        start = 0
        for s in self.states_list:
            s._din = self._din[start : start + s.size]
            start = s.size

    def detach(self) -> StackedState:
        """ """
        obj = StackedState([s.detach() for s in self.states_list], requires_grad=False)
        return obj

    def clone(self) -> StackedState:
        obj = StackedState(
            [s.clone() for s in self.states_list], requires_grad=self.requires_grad
        )
        return obj

    # Required for gradient calculation
    # ----------------------------------------------------------------------------------------------
    def d_out(self) -> torch.Tensor:
        tmp = []
        for s in self.states_list:
            tt = s.d_out()
            tmp.append(tt)
        ret = torch.cat(tmp)
        return ret

    # Difference
    # ----------------------------------------------------------------------------------------------
    def increment(self, dstate):
        start = 0
        for s in self.states_list:
            s.increment(dstate[start : start + s.size])
            start = s.size
        return self

    def diff(self, other_state: StackedState) -> torch.Tensor:
        return torch.cat(
            [s1.diff(s2) for s1, s2 in zip(self.states_list, other_state.states_list)]
        )

    # Numpy
    # ----------------------------------------------------------------------------------------------
    def numpy(self):
        return [s.numpy() for s in self.states_list]


class MultiBodyState(BaseState):
    """ Saves multibody states as a list of R, p, v_w
        d_out, diff and increment use the convention: [dp, dv, dr, dw]
    """
    def __init__(
        self,
        R: torch.Tensor,
        p: torch.Tensor,
        v_w: torch.Tensor,
        requires_grad=False,
    ) -> None:
        self.n_body = R.shape[0]
        self.n_state = 12
        self.R = R
        self.p = p  # N x 3
        self.v_w = v_w  # N x 6

        # required for backprop
        self._din: torch.Tensor = p.new_empty(0)
        super().__init__(base_tensor=self.p.new_empty(1), requires_grad=requires_grad)

    # Pytorch-Tensor like properties
    # ----------------------------------------------------------------------------------------------
    @property
    def size(self):
        return self.n_body * self.n_state

    def detach(self):
        obj = MultiBodyState(
            self.R.clone().detach(),
            self.p.clone().detach(),
            self.v_w.clone().detach(),
            False,
        )  # Does not call __init__
        return obj

    def clone(self):
        return MultiBodyState(
            self.R.clone(),
            self.p.clone(),
            self.v_w.clone(),
            self.requires_grad,
        )

    # Required for gradient calculation
    # ----------------------------------------------------------------------------------------------
    def d_in(self):
        return self._din

    def d_out(self):
        log_d_g_R = so3_log_map(
            self.R @ self.R.clone().detach().transpose(1, 2)
        )  # global frame: R - R.detach()

        tmp = torch.cat([self.p, self.v, log_d_g_R, self.w], dim=1).view(-1)
        return tmp

    def _set_din(self):
        ret = [torch.zeros_like(self.p), torch.zeros_like(self.v),
               torch.zeros_like(self.w), torch.zeros_like(self.w)]
        tmp = torch.cat(ret, dim=1)
        self._din = tmp.view(-1).requires_grad_()

    # Difference in expresed in so3
    # ----------------------------------------------------------------------------------------------
    def increment(self, d_state: torch.Tensor):
        tmp_dste = d_state.view(self.n_body, -1)
        assert tmp_dste.shape[1] == 12

        self.p = self.p + tmp_dste[:, :3]
        self.v = self.v + tmp_dste[:, 3:6]
        self.R = (
            so3_exp_map(tmp_dste[:, 6:9]) @ self.R
        )  # global frame: self.R + d_state
        self.w = self.w + tmp_dste[:, 9:]
        return self

    def add(self, other_state: MultiBodyState):
        assert (
            self.n_body == other_state.n_body
        ), "make sure the state have the same nr of bodies"
        self.p = self.p + other_state.p
        self.v = self.v + other_state.v
        self.R = other_state.R @ self.R  # global frame: self.R + d_state
        self.w = self.w + other_state.w

    # Difference in expresed in so3
    # ----------------------------------------------------------------------------------------------
    def diff(self, other_state: MultiBodyState):
        ret = [self._diff_trans(other_state), self._diff_orient(other_state)]
        return torch.cat(ret, dim=1).view(-1)  # (N, 12)

    def _diff_trans(self, other_state: MultiBodyState):
        diff_p = self.p - other_state.p
        diff_v = self.v - other_state.v
        return torch.cat([diff_p, diff_v], dim=1)  # (N, 6)

    def _diff_orient(self, other_state: MultiBodyState):
        diff_orient = so3_log_map(self.R @ other_state.R.transpose(1, 2))
        diff_w = self.w - other_state.w
        return torch.cat([diff_orient, diff_w], dim=1)  # (N, 6)

    # Helpers
    # ----------------------------------------------------------------------------------------------
    @property
    def v(self):
        return self.v_w[:, :3]

    @v.setter
    def v(self, v):
        self.v_w[:, :3] = v

    @property
    def w(self):
        return self.v_w[:, 3:]

    @w.setter
    def w(self, w):
        self.v_w[:, 3:] = w

    # Numpy
    # ----------------------------------------------------------------------------------------------
    def numpy(self):  # N x 12
        return np.concatenate(
            (self._get_numpy_pose(), self._get_numpy_velocities()), axis=1
        )

    def _get_numpy_pose(self):
        return (
            torch.cat(
                [self.p.detach(), matrix_to_quaternion(self.R.detach())], dim=1
            )  # N x 7
            .cpu()
            .numpy()
        )

    def _get_numpy_velocities(self):
        return self.v_w.detach().cpu().numpy()  # N x 6


class BasicState(BaseState):
    def __init__(self, value: torch.Tensor, requires_grad=False) -> None:
        self.val = value
        super().__init__(self.val.new_empty(1), requires_grad)

    # Pytorch-Tensor like properties
    # ----------------------------------------------------------------------------------------------
    def __getitem__(self, key):
        return self.val[key]

    def __setitem__(self, key, val):
        self.val[key] = val

    @property
    def size(self):
        return self.val.size(0)

    def _set_din(self):
        self._din = torch.zeros_like(self.val).requires_grad_()

    def detach(self):
        obj = BasicState(self.val.clone().detach(), False)  # Does not call __init__
        return obj

    def clone(self):
        return BasicState(self.val.clone(), self.requires_grad)

    # Required for gradient calculation
    # ----------------------------------------------------------------------------------------------
    def d_out(self):
        return self.val

    # Difference
    # ----------------------------------------------------------------------------------------------
    def increment(self, dstate: torch.Tensor):
        self.val = self.val + dstate
        return self

    def diff(self, other_state: BasicState):
        return self.val - other_state.val

    # Numpy
    # ----------------------------------------------------------------------------------------------
    def numpy(self):
        return self.val.view(-1).detach().cpu().numpy()
