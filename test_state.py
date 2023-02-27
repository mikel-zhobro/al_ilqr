import torch
from al_ilqr.state import StackedState, BasicState, MultiBodyState
from al_ilqr.utils import dfdx_vmap

from pytorch3d.transforms import random_rotations

a = BasicState(torch.ones(11))
b = MultiBodyState(R=random_rotations(5), p=torch.rand((5, 3)), v_w=torch.rand((5, 6)))
ab = StackedState([a, b])


c = BasicState(2 * torch.ones(11))
d = MultiBodyState(R=random_rotations(5), p=torch.rand((5, 3)), v_w=torch.rand((5, 6)))

cd = StackedState([c, d])


# test diff, add,increment,()
diff = cd - ab
dd = ab + diff
diffnew = dd.diff(cd)
print(torch.allclose(diffnew, torch.zeros_like(diffnew), atol=1e-4))

#  din, dout not numerically)
ab.requires_grad_()
diff2 = torch.exp(cd - ab)
res = ab + diff2
d_res_d_diff2 = dfdx_vmap(res, diff2)
d_res_d_ab = dfdx_vmap(res, ab)

pass


# testing pytorch3d
# def close_en(d):
#     return torch.allclose(d, torch.zeros_like(d), atol=1e-5)
# for i in range(52):
#     R1 = random_rotations(1)
#     R2 = random_rotations(1)

#     dd = so3_log_map(R1 @ R2.transpose(1,2))
#     dR = so3_exp_map(dd)


#     R2_new = dR @ R2
#     dd_new = so3_log_map(R1 @ R2_new.transpose(1,2))

#     if not close_en(dd_new):
#         print(dd_new)
