""" ================ RELIABILITY 4 BRANCH ============== """
"""
/RELIABILITY/PROBLEM_PATH
    * name.fdmlreliability [HDF file]
        ** [G] experiment_label 
            *** [G] AdaptiveModel.model_type_samples 
                **** [D] update, new_sample
            *** [G] AdaptiveModel.model_type_stats 
                **** [D] update, cov, pf                        
    * /MODELS
        ** AdaptiveModel.model_type_initial.fdmlmodeltype
        ** AdaptiveModel.model_type_update.fdmlmodeltype
    * /PLOTS
        ** name-experiment_label-AdaptiveModel.model_type-update.png

[G] hdf.Group ; [D] hdf.Dataset  
"""
import numpy as np
import sys
sys.path.insert(0, "../../")

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor as SKGPR
from fdml.kernels import Matern52
from fdml.deep_models2 import GPLayer, SIDGP, GPNode
from fdml.adaptive import AdaptiveModel, Reliability, Analytic

four_branch = Analytic("FOUR_BRANCH")
X_train = np.random.standard_normal((10, 2))
Z_train = four_branch(X_train)
nftr = X_train.shape[1]

# GPR Model
# skkernel = Matern(nu=2.5)
# sk_model = SKGPR(kernel=skkernel)
# sk_adaptive = AdaptiveModel(data=(X_train, Z_train), base_model=sk_model)

# SIDGP Model
node1 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
node2 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
node3 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
node4 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
node5 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))

node1.likelihood_variance.fix()
node2.likelihood_variance.fix()
node3.likelihood_variance.fix()
node4.likelihood_variance.fix()
node5.likelihood_variance.fix()

node1.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 10.0 * np.ones(nftr))
node2.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 10.0 * np.ones(nftr))
node3.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 10.0 * np.ones(nftr))
node4.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 10.0 * np.ones(nftr))
node5.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 10.0 * np.ones(nftr))

layer1 = GPLayer(nodes=[node1, node2])
layer2 = GPLayer(nodes=[node3, node4])
layer3 = GPLayer(nodes=[node5])

layer1.set_inputs(X_train)
layer3.set_outputs(Z_train)

dgp_model = SIDGP(layers=[layer1, layer2, layer3])

problem = Reliability(  X_initial=X_train,
                        Z_initial=Z_train,
                        model=dgp_model, name="DGP",
                        reference_model=four_branch, log_path=".",
                        n_mcs=int(1E5),
                        experiment_label="1")

dgp_adaptive = AdaptiveModel(problem=problem, plot=True)
dgp_adaptive.update(n_update=100, analytical=True, n_impute=100, n_thread=300)
