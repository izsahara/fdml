import numpy as np
from numpy import ndarray, any
from typing import Tuple, Optional
from ._parameters import FloatParameter as _FP
from ._parameters import VectorParameter as _VP

AVAILABLE_TRANSFORMS = ["none", "logexp"]

class Parameter:
    def __init__(self):
        pass


class VectorParameter(_VP, Parameter):

    def __init__(self, name: str, value :ndarray, transform : str = "none", bounds : Optional[Tuple[ndarray, ndarray]] = None):
        Parameter.__init__(self)
        assert transform in AVAILABLE_TRANSFORMS, "Unrecognized Transform"
        bounds = (np.ones(value.shape) * -np.inf, np.ones(value.shape) * np.inf) if isinstance(bounds, type(None)) else bounds
        _VP.__init__(self, name, value, transform, bounds)


class FloatParameter(_FP, Parameter):

    def __init__(self, name: str, value: float, transform : str = "none", bounds : Optional[Tuple[float, float]] = None):
        Parameter.__init__(self)
        assert transform in AVAILABLE_TRANSFORMS, "Unrecognized Transform"
        bounds = (-np.inf, np.inf) if isinstance(bounds, type(None)) else bounds
        _FP.__init__(self, name, value, transform, bounds)