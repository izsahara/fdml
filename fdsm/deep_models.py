from __future__ import annotations
import numpy as np

from typing import Tuple, List, Optional
from os.path import exists
from .base_models import GPNode
from ._deep_models import *


class GPLayer(CGPLayer):

    def __init__(self, nodes : List[GPNode]):
        super(GPLayer, self).__init__(nodes=nodes)

    def __call__(self, layer : GPLayer) -> None:
        super(GPLayer, self).propagate(layer)

    def set_outputs(self, Y : np.ndarray):
        # NOTE: RECONSTRUCT ONLY USED IN LOADING MODEL
        # TODO: HIDE RECONSTRUCT FROM USER
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

    def predict(self, X : np.ndarray, n_impute: int = 50, n_thread : int = 1) -> Tuple[ndarray, ndarray]:
        return super(SIDGP, self).predict(X=X, n_impute=n_impute, n_thread=n_thread)
