import numpy as np
from typing import Union, Optional, Tuple
from .parameters import Parameter
from .kernels import SquaredExponential
from .src._base_models2 import PSO as _PSO
from .src._base_models2 import LBFGSB as _LBFGSB
from .src._base_models2 import GradientDescent as _GD
from .src._base_models2 import DifferentialEvolution as _DE
from .src._base_models2 import GPR as _GPR

TNone = type(None)

# ================================= OPTIMIZERS =============================== #

class PSO(_PSO):

    def __init__(self, verbosity: int = 1, n_restarts: int = 1, sampling_method: str = "uniform"):
        _PSO.__init__(self, verbosity, n_restarts, sampling_method)
    # @property
    # def verbosity(self):
    #     return _PSO.verbosity
    # @property
    # def n_restarts(self):
    #     return _PSO.n_restarts
    # @property
    # def sampling_method(self):
    #     return _PSO.sampling_method
    # @property
    # def conv_failure_switch(self):
    #     return _PSO.conv_failure_switch
    # @property
    # def iter_max(self):
    #     return _PSO.iter_max
    # @property
    # def err_tol(self):
    #     return  _PSO.err_tol
    # @property
    # def vals_bound(self):
    #     return _PSO.vals_bound
    # @property
    # def center_particle(self):
    #     return _PSO.center_particle
    # @property
    # def n_pop(self):
    #     return _PSO.n_pop
    # @property
    # def n_gen(self):
    #     return _PSO.n_gen
    # @property
    # def inertia_method(self):
    #     return _PSO.inertia_method
    # @property
    # def initial_w(self):
    #     return _PSO.initial_w
    # @property
    # def w_damp(self):
    #     return _PSO.w_damp
    # @property
    # def w_min(self):
    #     return _PSO.w_min
    # @property
    # def w_max(self):
    #     return _PSO.w_max
    # @property
    # def velocity_method(self):
    #     return _PSO.velocity_method
    # @property
    # def c_cog(self):
    #     return _PSO.c_cog
    # @property
    # def c_soc(self):
    #     return _PSO.c_soc
    # @property
    # def initial_c_cog(self):
    #     return _PSO.initial_c_cog
    # @property
    # def final_c_cog(self):
    #     return _PSO.final_c_cog
    # @property
    # def initial_c_soc(self):
    #     return _PSO.initial_c_soc
    # @property
    # def final_c_soc(self):
    #     return _PSO.final_c_soc
    #
    # @verbosity.setter
    # def verbosity(self, value : int):
    #     _PSO.verbosity = value
    # @n_restarts.setter
    # def n_restarts(self, value : int):
    #     _PSO.n_restarts = value
    # @sampling_method.setter
    # def sampling_method(self, value : str):
    #     _PSO.sampling_method = value
    # @conv_failure_switch.setter
    # def conv_failure_switch(self, value : int):
    #     _PSO.conv_failure_switch = value
    # @iter_max.setter
    # def iter_max(self, value : int):
    #     _PSO.iter_max = value
    # @err_tol.setter
    # def err_tol(self, value : float):
    #     _PSO.err_tol = value
    # @vals_bound.setter
    # def vals_bound(self, value : bool):
    #     _PSO.vals_bound = value
    # @center_particle.setter
    # def center_particle(self, value: bool):
    #     _PSO.center_particle = value
    # @n_pop.setter
    # def n_pop(self, value: int):
    #     _PSO.n_pop = value
    # @n_gen.setter
    # def n_gen(self, value: int):
    #     _PSO.n_gen = value
    # @inertia_method.setter
    # def inertia_method(self, value: int):
    #     _PSO.inertia_method = value
    # @initial_w.setter
    # def initial_w(self, value: float):
    #     _PSO.initial_w = value
    # @w_damp.setter
    # def w_damp(self, value: float):
    #     _PSO.w_damp = value
    # @w_min.setter
    # def w_min(self, value: float):
    #     _PSO.w_min = value
    # @w_max.setter
    # def w_max(self, value: float):
    #     _PSO.w_max = value
    # @velocity_method.setter
    # def velocity_method(self, value: int):
    #     _PSO.velocity_method = value
    # @c_cog.setter
    # def c_cog(self, value: float):
    #     _PSO.c_cog = value
    # @c_soc.setter
    # def c_soc(self, value: float):
    #     _PSO.c_soc = value
    # @initial_c_cog.setter
    # def initial_c_cog(self, value: float):
    #     _PSO.initial_c_cog = value
    # @final_c_cog.setter
    # def final_c_cog(self, value: float):
    #     _PSO.final_c_cog = value
    # @initial_c_soc.setter
    # def initial_c_soc(self, value: float):
    #     _PSO.initial_c_soc = value
    # @final_c_soc.setter
    # def final_c_soc(self, value: float):
    #     _PSO.final_c_soc = value

class DifferentialEvolution(_DE):

    def __init__(self, verbosity: int = 1, n_restarts: int = 1, sampling_method: str = "uniform"):
        _DE.__init__(self, verbosity, n_restarts, sampling_method)
    # @property
    # def verbosity(self):
    #     return _PSO.verbosity
    # @property
    # def n_restarts(self):
    #     return _PSO.n_restarts
    # @property
    # def sampling_method(self):
    #     return _PSO.sampling_method
    # @property
    # def conv_failure_switch(self):
    #     return _PSO.conv_failure_switch
    # @property
    # def iter_max(self):
    #     return _PSO.iter_max
    # @property
    # def err_tol(self):
    #     return  _PSO.err_tol
    # @property
    # def vals_bound(self):
    #     return _PSO.vals_bound
    # @property
    # def center_particle(self):
    #     return _PSO.center_particle
    # @property
    # def n_pop(self):
    #     return _PSO.n_pop
    # @property
    # def n_gen(self):
    #     return _PSO.n_gen
    # @property
    # def inertia_method(self):
    #     return _PSO.inertia_method
    # @property
    # def initial_w(self):
    #     return _PSO.initial_w
    # @property
    # def w_damp(self):
    #     return _PSO.w_damp
    # @property
    # def w_min(self):
    #     return _PSO.w_min
    # @property
    # def w_max(self):
    #     return _PSO.w_max
    # @property
    # def velocity_method(self):
    #     return _PSO.velocity_method
    # @property
    # def c_cog(self):
    #     return _PSO.c_cog
    # @property
    # def c_soc(self):
    #     return _PSO.c_soc
    # @property
    # def initial_c_cog(self):
    #     return _PSO.initial_c_cog
    # @property
    # def final_c_cog(self):
    #     return _PSO.final_c_cog
    # @property
    # def initial_c_soc(self):
    #     return _PSO.initial_c_soc
    # @property
    # def final_c_soc(self):
    #     return _PSO.final_c_soc
    #
    # @verbosity.setter
    # def verbosity(self, value : int):
    #     _PSO.verbosity = value
    # @n_restarts.setter
    # def n_restarts(self, value : int):
    #     _PSO.n_restarts = value
    # @sampling_method.setter
    # def sampling_method(self, value : str):
    #     _PSO.sampling_method = value
    # @conv_failure_switch.setter
    # def conv_failure_switch(self, value : int):
    #     _PSO.conv_failure_switch = value
    # @iter_max.setter
    # def iter_max(self, value : int):
    #     _PSO.iter_max = value
    # @err_tol.setter
    # def err_tol(self, value : float):
    #     _PSO.err_tol = value
    # @vals_bound.setter
    # def vals_bound(self, value : bool):
    #     _PSO.vals_bound = value
    # @center_particle.setter
    # def center_particle(self, value: bool):
    #     _PSO.center_particle = value
    # @n_pop.setter
    # def n_pop(self, value: int):
    #     _PSO.n_pop = value
    # @n_gen.setter
    # def n_gen(self, value: int):
    #     _PSO.n_gen = value
    # @inertia_method.setter
    # def inertia_method(self, value: int):
    #     _PSO.inertia_method = value
    # @initial_w.setter
    # def initial_w(self, value: float):
    #     _PSO.initial_w = value
    # @w_damp.setter
    # def w_damp(self, value: float):
    #     _PSO.w_damp = value
    # @w_min.setter
    # def w_min(self, value: float):
    #     _PSO.w_min = value
    # @w_max.setter
    # def w_max(self, value: float):
    #     _PSO.w_max = value
    # @velocity_method.setter
    # def velocity_method(self, value: int):
    #     _PSO.velocity_method = value
    # @c_cog.setter
    # def c_cog(self, value: float):
    #     _PSO.c_cog = value
    # @c_soc.setter
    # def c_soc(self, value: float):
    #     _PSO.c_soc = value
    # @initial_c_cog.setter
    # def initial_c_cog(self, value: float):
    #     _PSO.initial_c_cog = value
    # @final_c_cog.setter
    # def final_c_cog(self, value: float):
    #     _PSO.final_c_cog = value
    # @initial_c_soc.setter
    # def initial_c_soc(self, value: float):
    #     _PSO.initial_c_soc = value
    # @final_c_soc.setter
    # def final_c_soc(self, value: float):
    #     _PSO.final_c_soc = value

class LBFGSB(_LBFGSB):

    def __init__(self, verbosity: int = 1, n_restarts: int = 10, sampling_method: str = "uniform"):
        _LBFGSB.__init__(self, verbosity, n_restarts, sampling_method)

class GradientDescent(_GD):

    def __init__(self, method : int = 1, verbosity: int = 1, n_restarts: int = 10, sampling_method: str = "uniform"):
        _GD.__init__(self, method, verbosity, n_restarts, sampling_method)

    # @property
    # def verbosity(self):
    #     return _LBFGSB.verbosity
    # @property
    # def n_restarts(self):
    #     return _LBFGSB.n_restarts
    # @property
    # def sampling_method(self):
    #     return _LBFGSB.sampling_method
    # @property
    # def conv_failure_switch(self):
    #     return _LBFGSB.conv_failure_switch
    # @property
    # def iter_max(self):
    #     return _LBFGSB.iter_max
    # @property
    # def err_tol(self):
    #     return _LBFGSB.err_tol
    # @property
    # def vals_bound(self):
    #     return _LBFGSB.vals_bound
    # @property
    # def M(self):
    #     return _LBFGSB.M
    #
    # @verbosity.setter
    # def verbosity(self, value: int):
    #     _LBFGSB.verbosity = value
    # @n_restarts.setter
    # def n_restarts(self, value: int):
    #     _LBFGSB.n_restarts = value
    # @sampling_method.setter
    # def sampling_method(self, value: str):
    #     _LBFGSB.sampling_method = value
    # @M.setter
    # def M(self, value: int):
    #     _LBFGSB.M = value
    # @conv_failure_switch.setter
    # def conv_failure_switch(self, value: int):
    #     _LBFGSB.conv_failure_switch = value
    # @iter_max.setter
    # def iter_max(self, value: int):
    #     _LBFGSB.iter_max = value
    # @err_tol.setter
    # def err_tol(self, value: float):
    #     _LBFGSB.err_tol = value
    # @vals_bound.setter
    # def vals_bound(self, value: bool):
    #     _LBFGSB.vals_bound = value

    # ================================= BASE MODELS =============================== #

class GPR(_GPR):

    def __init__(self,
                 data: Tuple[np.ndarray, np.ndarray],
                 kernel: Optional[SquaredExponential] = None,
                 likelihood_variance: Optional[Union[float, Parameter]] = 1e-6,
                 scale: Optional[Union[float, Parameter]] = 1.0,
                 solver: Optional[LBFGSB] = None):
        assert (data[0].ndim, data[1].ndim) == (2, 2), "Inputs and Outputs must be a 2D array"
        kernel = SquaredExponential(length_scale=np.ones(data[0].shape[1])) if isinstance(kernel, TNone) else kernel
        if not isinstance(solver, TNone):
            _GPR.__init__(self, kernel=kernel, inputs=data[0], outputs=data[1], likelihood_variance=likelihood_variance, scale=scale, solver=solver)
        else:
            _GPR.__init__(self, kernel=kernel, inputs=data[0], outputs=data[1], likelihood_variance=likelihood_variance, scale=scale)

    def train(self) -> None:
        super(GPR, self).train()
    def gradients(self) -> np.ndarray:
        return super(GPR, self).gradients()
    def log_marginal_likelihood(self) -> float:
        return super(GPR, self).log_marginal_likelihood()
    def predict(self, X : np.ndarray, return_var : bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return super(GPR, self).predict(X, return_var)
