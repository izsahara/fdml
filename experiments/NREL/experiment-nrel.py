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
from pickle import dump, load


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
        node11 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=2))
        node12 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=2))
        node13 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=2))
        node14 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=2))
        node15 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=2))
        
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

        node11.scale.fix()
        node12.scale.fix()
        node13.scale.fix()
        node14.scale.fix()
        node15.scale.fix()

        node11.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 3.0 * np.ones(nftr))
        node12.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 3.0 * np.ones(nftr))
        node13.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 3.0 * np.ones(nftr))
        node14.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 3.0 * np.ones(nftr))
        node15.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 3.0 * np.ones(nftr))

        # Layer 2
        node21 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=2))
        node22 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=2))
        node23 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=2))
        node24 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=2))
        node25 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=2))
        
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

        node21.scale.fix()
        node22.scale.fix()
        node23.scale.fix()
        node24.scale.fix()
        node25.scale.fix()

        node21.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 3.0 * np.ones(nftr))
        node22.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 3.0 * np.ones(nftr))
        node23.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 3.0 * np.ones(nftr))
        node24.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 3.0 * np.ones(nftr))
        node25.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 3.0 * np.ones(nftr))   

        node31 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=2))
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 3.0 * np.ones(nftr))


        layer1 = GPLayer(nodes=[node11, node12, node13, node14, node15])
        layer2 = GPLayer(nodes=[node21, node22, node23, node24, node25])
        layer3 = GPLayer(nodes=[node31])

        layer1.set_inputs(X_train)
        layer3.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2, layer3])
        print("Train Model")
        model.train(n_iter=200, ess_burn=100)
        model.estimate()
        return model

class Config2(Config):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, X_train, Y_train):
        # Layer 1
        nftr_l1 = X_train.shape[1]
        node11 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=2))
        node12 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=2))
        node13 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=2))
        node14 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=2))
        node15 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=2))
        
        node11.solver.solver_iterations = 30
        node12.solver.solver_iterations = 30
        node13.solver.solver_iterations = 30
        node14.solver.solver_iterations = 30
        node15.solver.solver_iterations = 30

        node11.likelihood_variance.fix()
        node12.likelihood_variance.fix()
        node13.likelihood_variance.fix()
        node14.likelihood_variance.fix()
        node15.likelihood_variance.fix()

        node11.scale.fix()
        node12.scale.fix()
        node13.scale.fix()
        node14.scale.fix()
        node15.scale.fix()

        node11.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 3.0 * np.ones(nftr_l1))
        node12.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 3.0 * np.ones(nftr_l1))
        node13.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 3.0 * np.ones(nftr_l1))
        node14.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 3.0 * np.ones(nftr_l1))
        node15.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 3.0 * np.ones(nftr_l1))

        # Layer 2
        nftr_l2 = 5
        node21 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l2), variance=1.0), solver=LBFGSB(verbosity=2))
        node22 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l2), variance=1.0), solver=LBFGSB(verbosity=2))        
        node21.solver.solver_iterations = 30
        node22.solver.solver_iterations = 30     
        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()   

        node21.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l2), 5.0 * np.ones(nftr_l2))
        node22.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l2), 5.0 * np.ones(nftr_l2)) 

        # Layer 3
        nftr_l3 = 2
        node31 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l3), variance=1.0), solver=LBFGSB(verbosity=2))
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l3), 5.0 * np.ones(nftr_l3))     
        #                                   
        layer1 = GPLayer(nodes=[node11, node12, node13, node14, node15])
        layer2 = GPLayer(nodes=[node21, node22])
        layer3 = GPLayer(nodes=[node31])

        layer1.set_inputs(X_train)
        layer3.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2, layer3])
        print("Train Model")
        model.train(n_iter=500, ess_burn=100)
        model.estimate()
        return model

def rf4_TwrBsMyt(config : Config, n_thread):
    X_train = np.loadtxt("data/Xsc_train.dat")
    X_test = np.loadtxt("data/Xsc_test.dat")
    Y_train = np.loadtxt("data/Y_train.dat")[:, 0].reshape(-1, 1)

    model = config(X_train, Y_train)
    modelfile = open(f"{config.name}.fdmlmodel", 'wb')
    dump(model, modelfile)
    modelfile.close()
    
    # modelfile = open(f"{config.name}.fdmlmodel", 'rb')
    # model = load(modelfile)
    # modelfile.close()

    mean, var = model.predict(X_test, n_impute=100, n_thread=n_thread)
    mean = mean.reshape(-1, 1)
    var = var.reshape(-1, 1)
    np.savetxt(f"{config.name}_Z1.dat", np.hstack([mean, var]), delimiter='\t')    

def rf4_Anch1Ten(config : Config, n_thread):
    X_train = np.loadtxt("data/Xsc_train.dat")
    X_test = np.loadtxt("data/Xsc_test.dat")
    Y_train = np.loadtxt("data/Y_train.dat")[:, 1].reshape(-1, 1)

    model = config(X_train, Y_train)
    modelfile = open(f"{config.name}.fdmlmodel", 'wb')
    dump(model, modelfile)
    modelfile.close()

    mean, var = model.predict(X_test, n_impute=100, n_thread=n_thread)
    mean = mean.reshape(-1, 1)
    var = var.reshape(-1, 1)
    np.savetxt(f"{config.name}_Z2.dat", np.hstack([mean, var]), delimiter='\t')   

def rf4_Anch3Ten(config : Config, n_thread):
    X_train = np.loadtxt("data/Xsc_train.dat")
    X_test = np.loadtxt("data/Xsc_test.dat")
    Y_train = np.loadtxt("data/Y_train.dat")[:, 2].reshape(-1, 1)

    model = config(X_train, Y_train)
    modelfile = open(f"{config.name}.fdmlmodel", 'wb')
    dump(model, modelfile)
    modelfile.close()

    mean, var = model.predict(X_test, n_impute=100, n_thread=n_thread)
    mean = mean.reshape(-1, 1)
    var = var.reshape(-1, 1)
    np.savetxt(f"{config.name}_Z3.dat", np.hstack([mean, var]), delimiter='\t')   


if __name__ == "__main__":
    rf4_TwrBsMyt(Config2(name="rf4_TwrBsMyt2-2"), n_thread=300)
    rf4_Anch1Ten(Config2(name="rf4_Anch1Ten2-1"), n_thread=300)
    rf4_Anch3Ten(Config2(name="rf4_Anch3Ten2-1"), n_thread=300)
