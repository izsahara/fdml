"""
Outputs : "rf4_TwrBsMyt", "rf4_Anch1Ten", "rf4_Anch3Ten"
"""

import numpy as np
import h5py as hdf
import sys
sys.path.insert(0, "../../")
from fdml.kernels import SquaredExponential, Matern52
from fdml.base_models2 import LBFGSB, PSO
from fdml.deep_models2 import GPNode, GPLayer, SIDGP
from pickle import dump


CFG = 1

class Config:
    def __init__(self, name):
        self.name = name

    def __call__(self, X_train, Y_train):
        raise NotImplementedError

class Config1(Config):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, X_train, Y_train):
        nftr = X_train.shape[1]
        node11 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=PSO(verbosity=0))
        node12 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node13 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node14 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node15 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node11.solver.solver_iterations = 50
        node12.solver.solver_iterations = 50
        node13.solver.solver_iterations = 50
        node14.solver.solver_iterations = 50
        node15.solver.solver_iterations = 50

        node11.likelihood_variance.fix()
        node12.likelihood_variance.fix()
        node13.likelihood_variance.fix()
        node14.likelihood_variance.fix()
        node15.likelihood_variance.fix()

        node11.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 10.0 * np.ones(nftr))
        node12.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 10.0 * np.ones(nftr))
        node13.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 10.0 * np.ones(nftr))
        node14.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 10.0 * np.ones(nftr))
        node15.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 10.0 * np.ones(nftr))

        node21 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0), solver=LBFGSB(verbosity=0))
        layer1 = GPLayer(nodes=[node11, node12, node13, node14, node15])
        layer2 = GPLayer(nodes=[node21])

        layer1.set_inputs(X_train)
        layer2.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2])
        print("Train Model")
        model.train(n_iter=500, ess_burn=100)
        model.estimate()
        return model


def rf4_TwrBsMyt(config : Config, n_thread):
    X_train = np.loadtxt("100/X_train.dat")
    X_test = np.loadtxt("100/X_test.dat")
    Y_train = np.loadtxt("100/Y1_train.dat").reshape(-1, 1)
    model = config(X_train, Y_train)
    modelfile = open(f"CFG{CFG}/{config.name}.fdmlmodel", 'wb')
    dump(model, modelfile)
    modelfile.close()
    mean, var = model.predict(X_test, n_impute=100, n_thread=n_thread)
    mean = mean.reshape(-1, 1)
    var = var.reshape(-1, 1)
    np.savetxt(f"CFG{CFG}/Z1.dat", np.hstack([mean, var]), delimiter='\t')    

def rf4_Anch1Ten(config : Config, n_thread):
    X_train = np.loadtxt("100/X_train.dat")
    X_test = np.loadtxt("100/X_test.dat")
    Y_train = np.loadtxt("100/Y2_train.dat").reshape(-1, 1)
    model = config(X_train, Y_train)
    modelfile = open(f"CFG{CFG}/{config.name}.fdmlmodel", 'wb')
    dump(model, modelfile)
    modelfile.close()
    mean, var = model.predict(X_test, n_impute=100, n_thread=n_thread)
    mean = mean.reshape(-1, 1)
    var = var.reshape(-1, 1)
    np.savetxt(f"CFG{CFG}/Z2.dat", np.hstack([mean, var]), delimiter='\t')   

def rf4_Anch3Ten(config : Config, n_thread):
    X_train = np.loadtxt("100/X_train.dat")
    X_test = np.loadtxt("100/X_test.dat")
    Y_train = np.loadtxt("100/Y3_train.dat").reshape(-1, 1)
    model = config(X_train, Y_train)
    modelfile = open(f"CFG{CFG}/{config.name}.fdmlmodel", 'wb')
    dump(model, modelfile)
    modelfile.close()
    mean, var = model.predict(X_test, n_impute=100, n_thread=n_thread)
    mean = mean.reshape(-1, 1)
    var = var.reshape(-1, 1)
    np.savetxt(f"CFG{CFG}/Z3.dat", np.hstack([mean, var]), delimiter='\t')   


if __name__ == "__main__":

    rf4_TwrBsMyt(Config1(name="rf4_TwrBsMyt"), n_thread=200)
    rf4_Anch1Ten(Config1(name="rf4_Anch1Ten"), n_thread=200)
    rf4_Anch3Ten(Config1(name="rf4_Anch3Ten"), n_thread=200)
