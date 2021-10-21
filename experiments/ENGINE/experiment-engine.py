"""
AIRCRAFT-ENGINE EXPERIMENT
ENGINE.hdf
    - 100
        - 1 - 2 - ..... 25
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

N_SAMPLES = 100
EXPERIMENTS = 50

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
        node1 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0), solver=LBFGSB(verbosity=0))
        node2 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0), solver=LBFGSB(verbosity=0))
        node3 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0), solver=LBFGSB(verbosity=0))
        node4 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0), solver=LBFGSB(verbosity=0))
        node5 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0), solver=LBFGSB(verbosity=0))

        node1.solver.solver_iterations = 15
        node2.solver.solver_iterations = 15
        node3.solver.solver_iterations = 15
        node4.solver.solver_iterations = 15
        node5.solver.solver_iterations = 15

        node1.likelihood_variance.fix()
        node2.likelihood_variance.fix()
        node3.likelihood_variance.fix()
        node4.likelihood_variance.fix()
        node5.likelihood_variance.fix()

        node1.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node2.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node3.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node4.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node5.kernel.length_scale.bounds = (1e-5 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        layer1 = GPLayer(nodes=[node1, node2, node3])
        layer2 = GPLayer(nodes=[node4, node5])

        layer1.set_inputs(X_train)
        layer2.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2])
        model.train(n_iter=150, ess_burn=100)
        model.estimate()
        return model


def run_experiment(config: Config, n_thread : int):
    for exp in range(24, 31):
        print(f"EXPERIMENT {exp}")
        X_train = np.loadtxt(f"{N_SAMPLES}/{exp}/X_train.dat", delimiter="\t")
        Y_train = np.loadtxt(f"{N_SAMPLES}/{exp}/Y_train.dat", delimiter="\t")
        X_test = np.loadtxt(f"{N_SAMPLES}/{exp}/X_test.dat", delimiter="\t")

        model = config(X_train, Y_train)
        modelfile = open(f"{N_SAMPLES}/{exp}/{config.name}.fdmlmodel", 'wb')
        dump(model, modelfile)
        modelfile.close()

        mean, var = model.predict(X_test, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"{N_SAMPLES}/{exp}/Z.dat", np.hstack([mean, var]), delimiter='\t')


if __name__ == "__main__":
    run_experiment(Config1(), n_thread=100)



