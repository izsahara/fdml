"""
AIRCRAFT-ENGINE EXPERIMENT
CFG1 best so far
CFG4 == CFG1 increase length_scale bounds
"""

import os
import sys
import numpy as np
import h5py as hdf
from pickle import dump, load
sys.path.insert(0, "../../")
from fdml.kernels import SquaredExponential, Matern52
from fdml.deep_models2 import GPNode, GPLayer, SIDGP
from fdml.base_models2 import PSO, LBFGSB
from sklearn.metrics import mean_squared_error, r2_score
rmse = lambda yt, yp : mean_squared_error(yt, yp, squared=False)

class Config:
    def __init__(self, name):
        self.name = name

    def __call__(self, X_train, Y_train):
        raise NotImplementedError

class Config1(Config):
    def __init__(self):
        super(Config1, self).__init__(name="CFG1")

    def __call__(self, X_train, Y_train):
        n_samp, n_ftr = X_train.shape

        # ====================== Layer 1 ======================= #
        node11 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node12 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node13 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node11.solver.solver_iterations = 20
        node12.solver.solver_iterations = 20
        node13.solver.solver_iterations = 20

        node11.likelihood_variance.fix()
        node12.likelihood_variance.fix()
        node13.likelihood_variance.fix()

        node11.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node12.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node13.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Layer 2 ======================= #

        node21 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node22 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node23 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node21.solver.solver_iterations = 15
        node22.solver.solver_iterations = 15
        node23.solver.solver_iterations = 15

        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()
        node23.likelihood_variance.fix()

        node21.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node22.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node23.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Layer 3 ======================= #
        node31 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node31.solver.solver_iterations = 15
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Model ======================= #

        layer1 = GPLayer(nodes=[node11, node12, node13])
        layer2 = GPLayer(nodes=[node21, node22, node23])
        layer3 = GPLayer(nodes=[node31])

        layer1.set_inputs(X_train)
        layer3.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2, layer3])
        model.train(n_iter=250, ess_burn=100)
        model.estimate()
        return model

class Config2(Config):
    def __init__(self):
        super(Config2, self).__init__(name="CFG2")

    def __call__(self, X_train, Y_train):
        n_samp, n_ftr = X_train.shape

        # ====================== Layer 1 ======================= #
        node11 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node12 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node13 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node11.solver.solver_iterations = 30
        node12.solver.solver_iterations = 30
        node13.solver.solver_iterations = 30

        node11.likelihood_variance.fix()
        node12.likelihood_variance.fix()
        node13.likelihood_variance.fix()

        node11.kernel.length_scale.bounds = (1e-3 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node12.kernel.length_scale.bounds = (1e-3 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node13.kernel.length_scale.bounds = (1e-3 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Layer 2 ======================= #

        node21 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node22 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node23 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node21.solver.solver_iterations = 30
        node22.solver.solver_iterations = 30
        node23.solver.solver_iterations = 30

        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()
        node23.likelihood_variance.fix()

        node21.kernel.length_scale.bounds = (1e-3 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node22.kernel.length_scale.bounds = (1e-3 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node23.kernel.length_scale.bounds = (1e-3 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Layer 3 ======================= #
        node31 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node31.solver.solver_iterations = 30
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-3 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Model ======================= #

        layer1 = GPLayer(nodes=[node11, node12, node13])
        layer2 = GPLayer(nodes=[node21, node22, node23])
        layer3 = GPLayer(nodes=[node31])

        layer1.set_inputs(X_train)
        layer3.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2, layer3])
        model.train(n_iter=250, ess_burn=100)
        model.estimate()
        return model

class Config3(Config):
    def __init__(self):
        super(Config3, self).__init__(name="CFG3")

    def __call__(self, X_train, Y_train):
        n_samp, n_ftr = X_train.shape

        # ====================== Layer 1 ======================= #
        node11 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node12 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node13 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node11.solver.solver_iterations = 30
        node12.solver.solver_iterations = 30
        node13.solver.solver_iterations = 30

        node11.likelihood_variance.fix()
        node12.likelihood_variance.fix()
        node13.likelihood_variance.fix()

        node11.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node12.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node13.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Layer 2 ======================= #

        node21 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node22 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node23 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node21.solver.solver_iterations = 30
        node22.solver.solver_iterations = 30
        node23.solver.solver_iterations = 30

        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()
        node23.likelihood_variance.fix()

        node21.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node22.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node23.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Layer 3 ======================= #
        node31 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node31.solver.solver_iterations = 30
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Model ======================= #

        layer1 = GPLayer(nodes=[node11, node12, node13])
        layer2 = GPLayer(nodes=[node21, node22, node23])
        layer3 = GPLayer(nodes=[node31])

        layer1.set_inputs(X_train)
        layer3.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2, layer3])
        model.train(n_iter=250, ess_burn=100)
        model.estimate()
        return model

class Config4(Config):
    def __init__(self):
        super(Config4, self).__init__(name="CFG4")

    def __call__(self, X_train, Y_train):
        n_samp, n_ftr = X_train.shape

        # ====================== Layer 1 ======================= #
        node11 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node12 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node13 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node11.solver.solver_iterations = 30
        node12.solver.solver_iterations = 30
        node13.solver.solver_iterations = 30

        node11.likelihood_variance.fix()
        node12.likelihood_variance.fix()
        node13.likelihood_variance.fix()

        node11.kernel.length_scale.bounds = (1e-7 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))
        node12.kernel.length_scale.bounds = (1e-7 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))
        node13.kernel.length_scale.bounds = (1e-7 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))

        # ====================== Layer 2 ======================= #

        node21 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node22 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node23 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node21.solver.solver_iterations = 30
        node22.solver.solver_iterations = 30
        node23.solver.solver_iterations = 30

        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()
        node23.likelihood_variance.fix()

        node21.kernel.length_scale.bounds = (1e-7 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))
        node22.kernel.length_scale.bounds = (1e-7 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))
        node23.kernel.length_scale.bounds = (1e-7 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))

        # ====================== Layer 3 ======================= #
        node31 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node31.solver.solver_iterations = 30
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-7 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))

        # ====================== Model ======================= #

        layer1 = GPLayer(nodes=[node11, node12, node13])
        layer2 = GPLayer(nodes=[node21, node22, node23])
        layer3 = GPLayer(nodes=[node31])

        layer1.set_inputs(X_train)
        layer3.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2, layer3])
        model.train(n_iter=250, ess_burn=100)
        model.estimate()
        return model

class Config5(Config):
    def __init__(self):
        super(Config5, self).__init__(name="CFG5")

    def __call__(self, X_train, Y_train):
        n_samp, n_ftr = X_train.shape

        # ====================== Layer 1 ======================= #
        node11 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node12 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node13 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node11.solver.solver_iterations = 20
        node12.solver.solver_iterations = 20
        node13.solver.solver_iterations = 20

        node11.likelihood_variance.fix()
        node12.likelihood_variance.fix()
        node13.likelihood_variance.fix()

        node11.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node12.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node13.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Layer 2 ======================= #

        node21 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node22 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node23 = GPNode(kernel=SquaredExponential(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node21.solver.solver_iterations = 15
        node22.solver.solver_iterations = 15
        node23.solver.solver_iterations = 15

        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()
        node23.likelihood_variance.fix()

        node21.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node22.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node23.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Layer 3 ======================= #
        node31 = GPNode(kernel=Matern52(length_scale=np.ones(n_ftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node31.solver.solver_iterations = 15
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Model ======================= #

        layer1 = GPLayer(nodes=[node11, node12, node13])
        layer2 = GPLayer(nodes=[node21, node22, node23])
        layer3 = GPLayer(nodes=[node31])

        layer1.set_inputs(X_train)
        layer3.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2, layer3])
        model.train(n_iter=500, ess_burn=100)
        model.estimate()
        return model



def run_experiment(config: Config, n_thread : int):
    print(f"==================== {config.name} ====================")
    for exp in range(1, 6):
        print(f"EXPERIMENT {exp}")
        X_train = np.loadtxt("X_train.dat", delimiter="\t")
        Y_train = np.loadtxt("Y_train.dat", delimiter="\t")
        X_test = np.loadtxt("X_test.dat", delimiter="\t")
        Y_test = np.loadtxt("Y_test.dat", delimiter="\t")

        model = config(X_train, Y_train)
        filename = f"{config.name}/{exp}.fdmlmodel"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as modelfile:
            dump(model, modelfile)        
   
        mean, var = model.predict(X_test, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)

        pred_path = os.path.abspath(f"{config.name}")
        np.savetxt(f"{pred_path}/Z{exp}.dat", np.hstack([mean, var]), delimiter='\t')

        rr = rmse(Y_test.ravel(), mean.ravel())
        nrmse = rr / (np.max(Y_test) - np.min(Y_test))
        print(f"NRMSE = {np.mean(nrmse)}")


if __name__ == "__main__":
    # run_experiment(Config1(), n_thread=50)
    # run_experiment(Config2(), n_thread=100)
    # run_experiment(Config4(), n_thread=100)
    run_experiment(Config5(), n_thread=100)



