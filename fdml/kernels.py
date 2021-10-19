from .parameters import Parameter
from .src._kernels import SquaredExponential as _SE
from .src._kernels import Matern32 as _M32
from .src._kernels import Matern52 as _M52

from typing import Union
from numpy import array, ndarray

# class Kernel:
#
#     def __init__(self, name: str):
#         self.name = name

class SquaredExponential(_SE):

    def __init__(self, length_scale: Union[Parameter, float, ndarray] = 1.0, variance: Union[Parameter, float] = 1.0):
        length_scale = array([length_scale]) if isinstance(length_scale, float) else length_scale
        length_scale = length_scale.reshape(-1, 1) if isinstance(length_scale, ndarray) else length_scale
        _SE.__init__(self, length_scale=length_scale, variance=variance)

    @property
    def length_scale(self):
        return super().length_scale

    @length_scale.setter
    def length_scale(self, value: ndarray):
        self.length_scale.value = value

    @property
    def variance(self):
        return super().variance

    @variance.setter
    def variance(self, value: float):
        self.variance.value = value

class Matern32(_M32):

    def __init__(self, length_scale: Union[Parameter, float, ndarray] = 1.0, variance: Union[Parameter, float] = 1.0):
        length_scale = array([length_scale]) if isinstance(length_scale, float) else length_scale
        length_scale = length_scale.reshape(-1, 1) if isinstance(length_scale, ndarray) else length_scale
        _M32.__init__(self, length_scale=length_scale, variance=variance)

    @property
    def length_scale(self):
        return super().length_scale

    @length_scale.setter
    def length_scale(self, value: ndarray):
        self.length_scale.value = value

    @property
    def variance(self):
        return super().variance

    @variance.setter
    def variance(self, value: float):
        self.variance.value = value

class Matern52(_M52):

    def __init__(self, length_scale: Union[Parameter, float, ndarray] = 1.0, variance: Union[Parameter, float] = 1.0):
        length_scale = array([length_scale]) if isinstance(length_scale, float) else length_scale
        length_scale = length_scale.reshape(-1, 1) if isinstance(length_scale, ndarray) else length_scale
        _M52.__init__(self, length_scale=length_scale,variance=variance)

    @property
    def length_scale(self):
        return super().length_scale

    @length_scale.setter
    def length_scale(self, value: ndarray):
        self.length_scale.value = value

    @property
    def variance(self):
        return super().variance

    @variance.setter
    def variance(self, value: float):
        self.variance.value = value
