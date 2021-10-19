import numpy as np
from .parameters import Parameter
from .kernels import SquaredExponential
from .src._base_models import GPR as _GPR
from numpy import ndarray
from typing import Union, Optional, Tuple

TNone = type(None)

class GPR(_GPR):

    def __init__(self,
                 data: Tuple[ndarray, ndarray],
                 kernel: Optional[SquaredExponential] = None,
                 likelihood_variance: Optional[Union[float, Parameter]] = 1e-6):
        assert (data[0].ndim, data[1].ndim) == (2, 2), "Inputs and Outputs must be a 2D array"
        # kernel =  kernel if isinstance(kernel, Kernel) else SquaredExponential(length_scale=np.ones(data[0].shape[1]))
        kernel = SquaredExponential(length_scale=np.ones(data[0].shape[1])) if isinstance(kernel, TNone) else kernel
        _GPR.__init__(self, kernel=kernel, inputs=data[0], outputs=data[1], likelihood_variance=likelihood_variance)
        # self._kernel = super(GPR, self).kernel

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
