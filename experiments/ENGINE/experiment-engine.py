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
        node1 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0), solver=LBFGSB(verbosity=2))
        node2 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0), solver=LBFGSB(verbosity=2))
        node3 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0), solver=LBFGSB(verbosity=2))
        node4 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0), solver=LBFGSB(verbosity=2))

        node1.solver.solver_iterations = 15
        node2.solver.solver_iterations = 15
        node3.solver.solver_iterations = 15
        node4.solver.solver_iterations = 15

        ext_solver = PSO(verbosity=2, n_restarts=3)
        ext_solver.n_gen = 25
        ext_solver.n_pop = 1000
        node5 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0), solver=ext_solver)

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
        model.train(n_iter=500, ess_burn=100)
        model.estimate()
        return model


def run_experiment(config: Config, n_thread : int):
    foldername = f"{config.name}"
    os.mkdir(foldername)
    file = hdf.File("ENGINE.hdf", "r")
    for exp in file[f"{N_SAMPLES}"]:
        print("EXPERIMENT " + exp)
        dataset = file[f"{N_SAMPLES}"][exp]
        X_train, Y_train = dataset["X_train"][:], dataset["Y_train"][:]
        X_test = dataset["X_test"][:]

        model = config(X_train, Y_train)
        modelfile = open(f"{foldername}/{config.name}_{int(exp)}.fdmlmodel", 'wb')
        dump(model, modelfile)
        modelfile.close()

        mean, var = model.predict(X_test, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"{foldername}/{int(exp)}.dat", np.hstack([mean, var]), delimiter='\t')


if __name__ == "__main__":
    run_experiment(Config1(), n_thread=4)