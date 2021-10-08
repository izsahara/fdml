import numpy as np
import h5py as hdf
from ._utilities import KernelPCA as _KPCA
from .parameters import FloatParameter as _FP
from .parameters import VectorParameter as _VP
from .kernels import SquaredExponential as _SE
from .kernels import Matern52 as _M52
from .kernels import Matern32 as _M32
from .base_models import GPNode as _GPNode
from .deep_models import GPLayer as _GPLayer
from .deep_models import SIDGP as _SIDGP

class KernelPCA(_KPCA):
    def __init__(self, n_components : int, kernel : str = "sigmoid"):
        super(KernelPCA, self).__init__(n_components=n_components, kernel=kernel)

    def transform(self, X : np.ndarray):
        return super(KernelPCA, self).transform(X)

"""
TODO: ADD PICKLE SUPPORT FROM C++
      ADD NAME ATTRB IN C++ CONSTRUCTOR
"""
def save_model(model : _SIDGP, root_path : str, label : str, update : bool = False):
    file = hdf.File(root_path, "a")
    if not update:
        state = file.create_group(label + "/MODEL")
        state.attrs.create("train_iter", model.train_iter)
        for ll, layer in enumerate(model.layers):
            lg = state.create_group(f'LAYER_{ll+1}')
            lg.attrs.create("observed_inputs", layer.observed_inputs())
            lg.attrs.create("observed_outputs", layer.observed_outputs())
            for nn, node in enumerate(layer.nodes):
                ng = lg.create_group(f'NODE_{nn+1}')
                # Input/Output
                ng.attrs.create("inputs", node.inputs)
                ng.attrs.create("outputs", node.outputs)
                # Likelihood Variance
                likelihood_variance = ng.create_group("likelihood_variance")
                likelihood_variance.attrs.create("value", node.likelihood_variance.value)
                likelihood_variance.attrs.create("bounds", node.likelihood_variance.bounds)
                likelihood_variance.attrs.create("fixed", node.likelihood_variance.fixed)
                likelihood_variance.attrs.create("constraint", node.likelihood_variance.constraint)
                # Scale
                scale = ng.create_group("scale")
                scale.attrs.create("value", node.scale.value)
                scale.attrs.create("bounds", node.scale.bounds)
                scale.attrs.create("fixed", node.scale.fixed)
                scale.attrs.create("constraint", node.scale.constraint)
                # Solver Settings
                solver_settings = ng.create_group("solver_settings")
                solver_settings.attrs.create("verbosity", node.solver_settings.verbosity)
                solver_settings.attrs.create("n_restarts", node.solver_settings.n_restarts)
                solver_settings.attrs.create("solver_iterations", node.solver_settings.solver_iterations)
                solver_settings.attrs.create("gradient_norm", node.solver_settings.gradient_norm)
                solver_settings.attrs.create("x_delta", node.solver_settings.x_delta)
                solver_settings.attrs.create("f_delta", node.solver_settings.f_delta)
                solver_settings.attrs.create("x_delta_violations", node.solver_settings.x_delta_violations)
                solver_settings.attrs.create("f_delta_violations", node.solver_settings.f_delta_violations)
                # Kernel Properties
                ng_kernel = ng.create_group("kernel")
                ng_kernel.attrs.create("type", str(type(node.kernel))[21:-2])
                # Kernel Length Scale
                length_scale = ng_kernel.create_group("length_scale")
                length_scale.attrs.create("value", node.kernel.length_scale.value)
                length_scale.attrs.create("bounds", node.kernel.length_scale.bounds)
                length_scale.attrs.create("fixed", node.kernel.length_scale.fixed)
                length_scale.attrs.create("constraint", node.kernel.length_scale.constraint)
                # Kernel Variance
                variance = ng_kernel.create_group("variance")
                variance.attrs.create("value", node.kernel.variance.value)
                variance.attrs.create("bounds", node.kernel.variance.bounds)
                variance.attrs.create("fixed", node.kernel.variance.fixed)
                variance.attrs.create("constraint", node.kernel.variance.constraint)
    else:
        state = file[label + "/MODEL"]
        state.attrs["train_iter"][:] = model.train_iter
        for ll, layer in enumerate(model.layers):
            lg = state[f'LAYER_{ll+1}']
            lg.attrs["observed_inputs"][:] = layer.observed_inputs()
            lg.attrs["observed_outputs"][:] = layer.observed_outputs()
            for nn, node in enumerate(layer.nodes):
                ng = lg[f'NODE_{nn + 1}']
                # Input/Output
                ng["inputs"][:] = node.inputs
                ng["outputs"][:] = node.outputs
                # Likelihood Variance
                likelihood_variance = ng["likelihood_variance"]
                likelihood_variance["value"][:] = node.likelihood_variance.value
                likelihood_variance["bounds"][:] = node.likelihood_variance.bounds
                likelihood_variance["fixed"][:] = node.likelihood_variance.fixed
                likelihood_variance["constraint"][:] = node.likelihood_variance.constraint
                # Scale
                scale = ng["scale"]
                scale.attrs["value"][:] =  node.scale.value
                scale.attrs["bounds"][:] =  node.scale.bounds
                scale.attrs["fixed"][:] =  node.scale.fixed
                scale.attrs["constraint"][:] =  node.scale.constraint
                # Solver Settings
                solver_settings = ng["solver_settings"]
                solver_settings.attrs["verbosity"][:] =  node.solver_settings.verbosity
                solver_settings.attrs["n_restarts"][:] =  node.solver_settings.n_restarts
                solver_settings.attrs["solver_iterations"][:] =  node.solver_settings.solver_iterations
                solver_settings.attrs["gradient_norm"][:] =  node.solver_settings.gradient_norm
                solver_settings.attrs["x_delta"][:] =  node.solver_settings.x_delta
                solver_settings.attrs["f_delta"][:] =  node.solver_settings.f_delta
                solver_settings.attrs["x_delta_violations"][:] =  node.solver_settings.x_delta_violations
                solver_settings.attrs["f_delta_violations"][:] =  node.solver_settings.f_delta_violations
                # Kernel Properties
                ng_kernel = ng["kernel"]
                ng_kernel.attrs["type"][:] = str(type(node.kernel))[21:-2]
                # Kernel Length Scale
                length_scale = ng_kernel["length_scale"]
                length_scale.attrs["value"][:] = node.kernel.length_scale.value
                length_scale.attrs["bounds"][:] = node.kernel.length_scale.bounds
                length_scale.attrs["fixed"][:] = node.kernel.length_scale.fixed
                length_scale.attrs["constraint"][:] = node.kernel.length_scale.constraint
                # Kernel Variance
                variance = ng_kernel["variance"]
                variance.attrs["value"][:] = node.kernel.variance.value
                variance.attrs["bounds"][:] = node.kernel.variance.bounds
                variance.attrs["fixed"][:] = node.kernel.variance.fixed
                variance.attrs["constraint"][:] = node.kernel.variance.constraint
    file.close()

def load_model(root_path : str, label : str):
    # Reconstruct model
    call_kernel = {'SquaredExponential': _SE, 'Matern52': _M52, 'Matern32': _M32}
    state = hdf.File(root_path, "r")[label]["MODEL"]
    layers = list()
    visit_layer = [_ for _ in state.values()]
    # visit_layer = list(filter(None, [g if not isinstance(g, hdf.Dataset)  else None for g in state.values()]))
    for idx, ll in enumerate(visit_layer):
        nodes = list()
        visit_node = [_ for _ in ll.values()]
        for nn in visit_node:
            # Kernel Properties
            ng_kernel = nn.get("kernel")
            # Kernel Length Scale
            ng_kernel_ls = ng_kernel.get("length_scale")
            length_scale = _VP(name="length_scale", value=ng_kernel_ls.attrs.get("value"))
            length_scale.bounds = ng_kernel_ls.attrs.get("bounds")
            length_scale.constraint = ng_kernel_ls.attrs.get("constraint")
            if ng_kernel_ls.attrs.get("fixed"):
                length_scale.fix()
            # Kernel Variance
            ng_kernel_var = ng_kernel.get("variance")
            variance = _FP(name="variance", value=ng_kernel_var.attrs.get("value"))
            variance.bounds = ng_kernel_var.attrs.get("bounds")
            variance.constraint = ng_kernel_var.attrs.get("constraint")
            if ng_kernel_var.attrs.get("fixed"):
                variance.fix()
            kernel = call_kernel[ng_kernel.attrs.get("type")](length_scale=length_scale, variance=variance)
            # Node Properties
            inputs = nn.attrs.get("inputs")
            outputs = nn.attrs.get("outputs")
            # Likelihood Variance
            nn_lv = nn.get("likelihood_variance")
            likelihood_variance = _FP(name="likelihood_variance", value=nn_lv.attrs.get("value"))
            likelihood_variance.bounds = nn_lv.attrs.get("bounds")
            likelihood_variance.constraint = nn_lv.attrs.get("constraint")
            if nn_lv.attrs.get("fixed"):
                likelihood_variance.fix()
            # Scale
            nn_scale = nn.get("scale")
            scale = _FP(name="scale", value=nn_scale.attrs.get("value"))
            scale.bounds = nn_scale.attrs.get("bounds")
            scale.constraint = nn_scale.attrs.get("constraint")
            if nn_scale.attrs.get("fixed"):
                scale.fix()
            # Set up Node
            node = _GPNode(kernel=kernel, likelihood_variance=likelihood_variance, scale=scale)
            node.inputs = inputs
            node.outputs = outputs
            node.scale = scale
            node.likelihood_variance = likelihood_variance
            # Solver Settings
            solver_settings = nn.get("solver_settings")
            node.solver_settings.verbosity = solver_settings.attrs.get("verbosity")
            node.solver_settings.n_restarts = solver_settings.attrs.get("n_restarts")
            node.solver_settings.solver_iterations = solver_settings.attrs.get("solver_iterations")
            node.solver_settings.gradient_norm = solver_settings.attrs.get("gradient_norm")
            node.solver_settings.x_delta = solver_settings.attrs.get("x_delta")
            node.solver_settings.f_delta = solver_settings.attrs.get("f_delta")
            node.solver_settings.x_delta_violations = solver_settings.attrs.get("x_delta_violations")
            node.solver_settings.f_delta_violations = solver_settings.attrs.get("f_delta_violations")
            nodes.append(node)

        layer = _GPLayer(nodes=nodes)
        layer.reconstruct_observed(ll.attrs.get("observed_inputs"), ll.attrs.get("observed_outputs"))
        layer.index = idx
        layers.append(layer)
    return _SIDGP(layers=layers, initialize=False)