from __future__ import annotations
import numpy as np

from typing import Tuple, List, Optional, Union
from .kernels import SquaredExponential
from .parameters import Parameter
from .base_models2 import LBFGSB
from .src._deep_models2 import CGPNode, CGPLayer, CSIDGP

TNone = type(None)

class GPNode(CGPNode):
    def __init__(self,
                 kernel: Optional[SquaredExponential] = None,
                 likelihood_variance: Optional[Parameter] = 1e-6, scale: Optional[Parameter] = 1.0,
                 solver: Optional[LBFGSB] = None):
        kernel = SquaredExponential(length_scale=1.0) if isinstance(kernel, TNone) else kernel
        if not isinstance(solver, TNone):
            CGPNode.__init__(self, kernel=kernel, likelihood_variance=likelihood_variance, scale=scale, solver=solver)
        else:
            CGPNode.__init__(self, kernel=kernel, likelihood_variance=likelihood_variance, scale=scale)

    def train(self) -> None:
        super(GPNode, self).train()

    def gradients(self) -> np.ndarray:
        return super(GPNode, self).gradients()

    def log_likelihood(self) -> float:
        return super(GPNode, self).log_likelihood()

    def log_marginal_likelihood(self) -> float:
        return super(GPNode, self).log_marginal_likelihood()

    def predict(self, X : np.ndarray, return_var : bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return super(GPNode, self).predict(X, return_var)

class GPLayer(CGPLayer):

    def __init__(self, nodes : List[GPNode], initialize = True):
        super(GPLayer, self).__init__(nodes=nodes, initialize=initialize)

    def __call__(self, layer : GPLayer) -> None:
        super(GPLayer, self).propagate(layer)

    def set_outputs(self, Y : np.ndarray):
        super(GPLayer, self).set_outputs_(Y, False)

    def train(self) -> None:
        super(GPLayer, self).train()

    def predict(self, X : np.ndarray) -> None:
        return super(GPLayer, self).predict(X)

class SIDGP(CSIDGP):

    def __init__(self, layers : List[GPLayer], initialize = True):
        super(SIDGP, self).__init__(layers, initialize)

    def train(self, n_iter : int = 500, ess_burn : int = 10) -> None:
        super(SIDGP, self).train(n_iter=n_iter, ess_burn=ess_burn)

    def estimate(self, n_burn : int = 0):
        super(SIDGP, self).estimate(n_burn=n_burn)

    def predict(self, X : np.ndarray, n_impute: int = 50, n_thread : int = 1) -> Tuple[np.ndarray, np.ndarray]:
        return super(SIDGP, self).predict(X=X, n_impute=n_impute, n_thread=n_thread)
