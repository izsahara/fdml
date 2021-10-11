import numpy as np
from .parameters import Parameter
from .kernels import Kernel, SquaredExponential
from ._base_models import GPR as _GPR
# from ._base_models import CGPNode as _GPNode
from numpy import ndarray
from typing import Union, Optional, Tuple

class Model:

    def __init__(self, name: str):
        self.name = name

class GPR(_GPR, Model):

    def __init__(self, data: Tuple[ndarray, ndarray], kernel: Optional[Kernel] = None, likelihood_variance: Optional[Parameter] = 1e-6):
        assert (data[0].ndim, data[1].ndim) == (2, 2), "Inputs and Outputs must be a 2D array"
        kernel =  kernel if isinstance(kernel, Kernel) else SquaredExponential(length_scale=np.ones(data[0].shape[1]))
        _GPR.__init__(self, kernel=kernel, inputs=data[0], outputs=data[1], likelihood_variance=likelihood_variance)
        self._kernel = super(GPR, self).kernel
        Model.__init__(self, name='GPR')

    def train(self) -> None:
        super(GPR, self).train()

    def gradients(self) -> ndarray:
        return super(GPR, self).gradients()

    def log_likelihood(self) -> float:
        return super(GPR, self).log_likelihood()

    def log_marginal_likelihood(self) -> float:
        return super(GPR, self).objective_fxn()

    def predict(self, X : ndarray, return_var : bool = False) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        return super(GPR, self).predict(X, return_var)

    @property
    def kernel(self):
        return self._kernel

# class GPNode(_GPNode, Model):
#     def __init__(self, kernel: Optional[Kernel] = None, likelihood_variance: Optional[Parameter] = 1e-6, scale: Optional[Parameter] = 1.0):
#         kernel =  kernel if isinstance(kernel, Kernel) else SquaredExponential(length_scale=np.ones(data[0].shape[1]))
#         _GPNode.__init__(self, kernel=kernel, likelihood_variance=likelihood_variance, scale=scale)
#         Model.__init__(self, name='GPNode')
#
#     def train(self) -> None:
#         super(GPNode, self).train()
#
#     def gradients(self) -> ndarray:
#         return super(GPNode, self).gradients()
#
#     def log_likelihood(self) -> float:
#         return super(GPNode, self).log_likelihood()
#
#     def log_marginal_likelihood(self) -> float:
#         return super(GPNode, self).objective_fxn()
#
#     def predict(self, X : ndarray, return_var : bool = False) -> Union[ndarray, Tuple[ndarray, ndarray]]:
#         return super(GPNode, self).predict(X, return_var)

