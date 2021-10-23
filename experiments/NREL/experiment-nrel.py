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
        # Layer 1
        node11 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node12 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node13 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node14 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node15 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        
        node11.solver.solver_iterations = 15
        node12.solver.solver_iterations = 15
        node13.solver.solver_iterations = 15
        node14.solver.solver_iterations = 15
        node15.solver.solver_iterations = 15

        node11.likelihood_variance.fix()
        node12.likelihood_variance.fix()
        node13.likelihood_variance.fix()
        node14.likelihood_variance.fix()
        node15.likelihood_variance.fix()

        node11.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))
        node12.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))
        node13.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))
        node14.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))
        node15.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))

        # Layer 2
        node21 = GPNode(kernel=SquaredExponential(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node22 = GPNode(kernel=SquaredExponential(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node23 = GPNode(kernel=SquaredExponential(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node24 = GPNode(kernel=SquaredExponential(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node25 = GPNode(kernel=SquaredExponential(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        
        node21.solver.solver_iterations = 15
        node22.solver.solver_iterations = 15
        node23.solver.solver_iterations = 15
        node24.solver.solver_iterations = 15
        node25.solver.solver_iterations = 15

        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()
        node23.likelihood_variance.fix()
        node24.likelihood_variance.fix()
        node25.likelihood_variance.fix()

        node21.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))
        node22.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))
        node23.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))
        node24.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))
        node25.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))   

        node31 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-4 * np.ones(nftr), 10.0 * np.ones(nftr))


        layer1 = GPLayer(nodes=[node11, node12, node13, node14, node15])
        layer2 = GPLayer(nodes=[node21, node22, node23, node24, node25])
        layer3 = GPLayer(nodes=[node31])

        layer1.set_inputs(X_train)
        layer3.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2, layer3])
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
    # mean, var = model.predict(X_test, n_impute=100, n_thread=n_thread)
    # mean = mean.reshape(-1, 1)
    # var = var.reshape(-1, 1)
    # np.savetxt(f"CFG{CFG}/Z1.dat", np.hstack([mean, var]), delimiter='\t')    

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

    rf4_TwrBsMyt(Config1(name="Y1"), n_thread=4)
    # rf4_Anch1Ten(Config1(name="rf4_Anch1Ten"), n_thread=200)
    # rf4_Anch3Ten(Config1(name="rf4_Anch3Ten"), n_thread=200)
