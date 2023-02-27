from __future__ import print_function, annotations
import time
import torch
from functorch import vmap
from scipy.signal.windows import gaussian
from numpy import linalg as la
import numpy as np

from typing import TYPE_CHECKING, Any

from lcp_physics.physics.forces import yield_body_multiforce, get_ix_force_torque_list

if TYPE_CHECKING:
    from .helpers import BaseILQRDynSys

class col:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    OKYELLOW = "\33[33m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def dfdx_vmap(y, x, allow_unused=False, create_graph=False) -> torch.Tensor:
    _x: torch.Tensor = x if torch.is_tensor(x) else x.d_in()
    _y: torch.Tensor = y if torch.is_tensor(y) else y.d_out()

    def get_vjp(v):
        if _y.requires_grad:

            ret_tmp = torch.autograd.grad(
                _y,
                _x,
                v,
                retain_graph=True,
                allow_unused=allow_unused,
                create_graph=create_graph,
            )[0]
        else:
            ret_tmp = _y.new_zeros(_x.numel())
        return ret_tmp if ret_tmp is not None else _y.new_zeros(_x.numel())

    N = _y.shape[0]
    I_N = torch.eye(N, device=_x[0].device)
    return vmap(get_vjp)(I_N)

def filter_me(sig, kernel_size=23, mode='replicate'):
    # sig: [N, W, H]
    assert sig.dim() > 1, "make sure the singal has at least 2 dims"


    # prepare kernel
    pad=(kernel_size-1)//2
    pr, pl = pad, pad if (kernel_size-1)%2==0 else pad+1
    kernel = torch.tensor(gaussian(kernel_size, 8.51)).view(-1)
    kernel = torch.nn.functional.normalize(kernel, dim=0, eps=1e-5,p=1).reshape(1,1,-1)


    # prepare input [N, W, H] -> [N, WH] -> [WH, N] -> [WH, N+Padd]
    in_ = sig.view(sig.shape[0], -1).T
    in_ = torch.nn.functional.pad(in_, (pl,pr), mode=mode).view(in_.shape[0],1,-1)

    # prepare output [WH, N] -> N, WH -> N, W, H
    out = torch.nn.functional.conv1d(in_, kernel).T.view(sig.shape)
    return out

class NPD:
    @staticmethod
    def isPD(B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False

    @staticmethod
    def nearestPD(A):
        """Find the nearest positive-definite matrix to input
        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].
        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if NPD.isPD(A3):
            return A3

        spacing = np.spacing(la.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not NPD.isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3


def print_dict(data_dict: dict[str, Any], n_tab=0):
    for key in data_dict.keys():
        print(
            "\t"*n_tab + f"{col.OKYELLOW} {key: ^15}: {col.ENDC}{str(data_dict[key]):^15}"
        )

if __name__ == '__main__':
    st = time.time()
    for i in range(10):
        for j in range(2, 100):
            A = np.random.randn(j, j)
            B = NPD.nearestPD(A)
            assert(NPD.isPD(B))
    print(time.time()-st)
    print('unit test passed!')