from __future__ import annotations
import numpy as np

from typing import Tuple, List, Optional
# from os.path import exists
# from .base_models import GPNode
from .kernels import Kernel
from .parameters import Parameter
from .base_models import Model
from ._deep_models import *

class GPNode(CGPNode, Model):
    def __init__(self, kernel: Optional[Kernel] = None, likelihood_variance: Optional[Parameter] = 1e-6, scale: Optional[Parameter] = 1.0):
        kernel =  kernel if isinstance(kernel, Kernel) else SquaredExponential(length_scale=np.ones(data[0].shape[1]))
        CGPNode.__init__(self, kernel=kernel, likelihood_variance=likelihood_variance, scale=scale)
        Model.__init__(self, name='GPNode')

    def train(self) -> None:
        super(GPNode, self).train()

    def gradients(self) -> ndarray:
        return super(GPNode, self).gradients()

    def log_likelihood(self) -> float:
        return super(GPNode, self).log_likelihood()

    def log_marginal_likelihood(self) -> float:
        return super(GPNode, self).objective_fxn()

    def predict(self, X : ndarray, return_var : bool = False) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        return super(GPNode, self).predict(X, return_var)

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


# class GPNode(CGPNode):
#
#     def __init__(self, kernel: Optional[Kernel] = None, likelihood_variance : Optional[Parameter] = None):
#         kernel = kernel if isinstance(kernel, Kernel) else SquaredExponential(length_scale=1.0)
#         likelihood_variance = likelihood_variance if isinstance(likelihood_variance, Parameter) \
#             else FloatParameter(name="likelihood_variance", value=1e-8)
#         super().__init__(kernel=kernel, likelihood_variance=likelihood_variance)
#
#     @property
#     def parameter_history(self):
#         return super(GPNode, self).get_parameter_history()
#     @property
#     def inputs(self):
#         return super(GPNode, self).inputs
#     @property
#     def outputs(self):
#         return super(GPNode, self).outputs
#     @inputs.setter
#     def inputs(self, X : np.ndarray):
#         super(GPNode, self).inputs = X
#     @outputs.setter
#     def outputs(self, Y : np.ndarray):
#         super(GPNode, self).outputs = Y
#
#     def sample_mvn(self) -> float:
#         return super(GPNode, self).sample_mvn()


# def node_sample(current_node : GPNode, linked_nodes : List[GPNode]):
#     if not np.all(current_node.missing):
#         return
#
#     mean = np.zeros(current_node.inputs.shape).ravel()
#     current_K = current_node.compute_K()
#     nu = np.random.default_rng().multivariate_normal(mean=mean, cov=current_K, check_valid='ignore').reshape(-1, 1)
#     log_y: float = 0.0
#     for node in linked_nodes:
#         # node.update_cholesky()
#         # cov = node.compute_K()
#         # cov = node.kernel.K(node.inputs, node.inputs) + np.eye(len(node.inputs))*node.likelihood_variance.value
#         # log_y += log_likelihood_func(node.outputs.ravel(), node.K, node.kernel.variance.value)
#         log_y += node.log_likelihood()
#         # log_y += node.objective_fxn()
#     log_y += np.log(np.random.uniform())
#     theta: float = np.random.uniform(0., 2. * np.pi)
#     theta_min, theta_max = theta - 2. * np.pi, theta
#     while True:
#         fp: np.ndarray = update_f(current_node.outputs, nu, mean.reshape(-1, 1), theta)
#         log_yp: float = 0.0
#         for node in linked_nodes:
#             node.inputs = fp
#             # cov = node.compute_K()
#             # cov = node.kernel.K(node.inputs, node.inputs) + np.eye(len(node.inputs)) * node.likelihood_variance.value
#             # log_yp += log_likelihood_func(node.outputs.ravel(), node.K, node.kernel.variance.value)
#             log_yp += node.log_likelihood()
#             # log_yp += node.objective_fxn()
#         if log_yp > log_y:
#             current_node.outputs = fp
#             return
#         else:
#             if theta < 0:
#                 theta_min = theta
#             else:
#                 theta_max = theta
#             theta = np.random.uniform(theta_min, theta_max)
#
# class GPNode(CGPNode):
#
#     def __init__(self, kernel: Optional[Kernel] = None):
#         kernel = kernel if isinstance(kernel, Kernel) else SquaredExponential(length_scale=1.0)
#         super().__init__(kernel=kernel)
#
#     @property
#     def parameter_history(self):
#         return super(GPNode, self).get_parameter_history()
#
#     # def log_marginal_likelihood(self) -> float:
#     #     return super(GPNode, self).log_marginal_likelihood()
#
#     def train(self) -> None:
#         super(GPNode, self).train()
#
#     def predict(self, X : np.ndarray, return_var : bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
#         return super().predict(X, return_var)
#
# class GPLayer(CGPLayer):
#
#     def __init__(self, nodes : List[GPNode]):
#         super(GPLayer, self).__init__(nodes=nodes)
#
#     def __call__(self, layer : GPLayer):
#         pass
#
#     # def log_marginal_likelihood(self) -> float:
#     #     return super(GPLayer, self).log_marginal_likelihood()
#     def train(self) -> None:
#         super(GPLayer, self).train()
#     def predict(self, X : np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
#         return super(GPLayer, self).predict(X, return_var)
#     def set_missing(self):
#         for node in self.nodes:
#             node.missing = np.ones(node.outputs.shape)


# class SIDGP:
#
#     def __init__(self, layers : List[GPLayer]):
#         # Initialize layers
#         if len(layers[0].observed_inputs) == 0:
#             raise ValueError("First Layer Requires Observed Inputs")
#         if len(layers[-1].observed_outputs) == 0:
#             raise ValueError("Last Layer Requires Observed Outputs")
#         # Configure First Layer
#         layers[0].index = 1
#         if len(layers[0].observed_outputs) > 0:
#             # Do something about missingness
#             pass
#         else:
#             layers[0].observed_outputs = layers[0].observed_inputs
#             # layers[0].update_cholesky()
#             layers[0].set_missing()
#
#         index : int = 2
#         X = layers[0].observed_outputs
#         for ll in range(1, len(layers)):
#             layers[ll].index = index
#             if len(layers[ll].nodes) == len(layers[ll-1].nodes):
#                 Y = X
#             elif len(layers[ll].nodes) < len(layers[ll-1].nodes):
#                 PCA = KernelPCA(n_components=len(layer.nodes), kernel='sigmoid')
#                 Y = PCA.fit_transform(X)
#             else:
#                 Y = np.hstack([X, X[:,np.random.choice(X.shape[1], len(layers[ll].nodes) - X.shape[1])]])
#
#             layers[ll].observed_inputs = X
#             if len(layers[ll].observed_outputs) > 0:
#                 # Do something about missingness
#                 continue
#             else:
#                 layers[ll].observed_outputs = Y
#                 layers[ll].set_missing()
#             X = Y
#
#         # super(SIDGP, self).__init__(layers=layers)
#         self.layers = layers
#         self.sample(10)
#
#     def sample(self, n_burn : int = 1):
#         for i in range(n_burn):
#             for ll in range(len(self.layers) - 1):
#                 for node in self.layers[ll].nodes:
#                     node_sample(node, self.layers[ll+1].nodes)
#
#     def train(self, train_iter : int = 500, ess_burn : int = 10):
#         total = train_iter
#         bar_length = 50
#         for i in range(total + 1):
#             percent = 100.0 * i / total
#             self.sample(ess_burn)
#             for layer in self.layers:
#                 layer.train()
#                 # for node in layer.nodes:
#                 #     node.train()
#             sys.stdout.write('\r')
#             sys.stdout.write("Training: [{:{}}] {:>3}%".format('=' * int(percent / (100.0 / bar_length)), bar_length, int(percent)))
#             sys.stdout.flush()
#         sys.stdout.write("\n")
#         for layer in self.layers:
#             layer.estimate_parameters()
#
#     # def predict(self, X : np.ndarray, n_impute : int = 50, n_thread : int = 2):
#     #     pass
#
#
#     # @property
#     # def observed_inputs(self) -> np.ndarray:
#     #     return self._observed_inputs
#     # @property
#     # def observed_outputs(self) -> np.ndarray:
#     #     return self._observed_outputs
#     # @observed_inputs.setter
#     # def observed_inputs(self, value : np.ndarray):
#     #     assert value.ndim > 1, "Expected 2D Array"
#     #     self._observed_inputs = value
#     # @observed_outputs.setter
#     # def observed_outputs(self, value : np.ndarray):
#     #     assert value.ndim > 1, "Expected 2D Array"
#     #     self._observed_outputs = value
#
