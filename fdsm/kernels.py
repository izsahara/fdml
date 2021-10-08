from .parameters import Parameter
from ._kernels import SquaredExponential as _SE
from ._kernels import Matern32 as _M32
from ._kernels import Matern52 as _M52

from typing import Union
from numpy import array, ndarray

class Kernel:

    def __init__(self, name: str):
        self.name = name

class SquaredExponential(_SE, Kernel):

    def __init__(self, length_scale: Union[Parameter, float, ndarray] = 1.0, variance: Union[Parameter, float] = 1.0):
        length_scale = array([length_scale]) if isinstance(length_scale, float) else length_scale
        length_scale = length_scale.reshape(-1, 1) if isinstance(length_scale, ndarray) else length_scale
        _SE.__init__(self, length_scale=length_scale, variance=variance)
        Kernel.__init__(self, name='SquaredExponential')

class Matern32(_M32, Kernel):

    def __init__(self, length_scale: Union[Parameter, float, ndarray] = 1.0, variance: Union[Parameter, float] = 1.0):
        length_scale = array([length_scale]) if isinstance(length_scale, float) else length_scale
        length_scale = length_scale.reshape(-1, 1) if isinstance(length_scale, ndarray) else length_scale
        _M32.__init__(self, length_scale=length_scale, variance=variance)
        Kernel.__init__(self, name='Matern32')

class Matern52(_M52, Kernel):

    def __init__(self, length_scale: Union[Parameter, float, ndarray] = 1.0, variance: Union[Parameter, float] = 1.0):
        length_scale = array([length_scale]) if isinstance(length_scale, float) else length_scale
        length_scale = length_scale.reshape(-1, 1) if isinstance(length_scale, ndarray) else length_scale
        _M52.__init__(self, length_scale=length_scale,variance=variance)
        Kernel.__init__(self, name='Matern52')