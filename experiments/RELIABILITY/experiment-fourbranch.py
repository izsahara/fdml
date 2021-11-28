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
from fdml.base_models2 import LBFGSB, PSO
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

# Layer 1

node11 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
node12 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
node11.likelihood_variance.fix()
node12.likelihood_variance.fix()
node11.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))
node12.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))

# Layer 2
node22 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
node21 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
node21.likelihood_variance.fix()
node22.likelihood_variance.fix()
node21.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))
node22.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))


# Layer 3
node31 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
node31.likelihood_variance.fix()
node31.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))

node11.solver.solver_iterations = 15
node12.solver.solver_iterations = 15
node21.solver.solver_iterations = 15
node22.solver.solver_iterations = 15
node31.solver.solver_iterations = 15


layer1 = GPLayer(nodes=[node11, node12])
layer2 = GPLayer(nodes=[node21, node22])
layer3 = GPLayer(nodes=[node31])

layer1.set_inputs(X_train)
layer3.set_outputs(Z_train)

dgp_model = SIDGP(layers=[layer1, layer2, layer3])

problem = Reliability(  X_initial=X_train,
                        Z_initial=Z_train,
                        model=dgp_model, name="DGP",
                        reference_model=four_branch, log_path=".",
                        n_mcs=int(1E6),
                        experiment_label="1")

dgp_adaptive = AdaptiveModel(problem=problem, plot=True)
dgp_adaptive.update(n_update=100, analytical=True, n_impute=100, n_thread=300)
