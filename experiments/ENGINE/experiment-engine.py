"""
AIRCRAFT-ENGINE EXPERIMENT
CFG1 best so far
CFG4 == CFG1 increase length_scale bounds
CFG5 == CFG1 increase train iter
CFG6 == CFG1 fix scales
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
    def __init__(self, name=None):
        if isinstance(name, type(None)):
            super(Config1, self).__init__(name="CFG1")
        else:
            super(Config1, self).__init__(name=name)

    def __call__(self, X_train, Y_train):
        n_samp, n_ftr = X_train.shape

        # ====================== Layer 1 ======================= #
        node11 = GPNode(kernel=Matern52(length_scale=np.ones(1), variance=1.0), solver=LBFGSB(verbosity=0))
        node12 = GPNode(kernel=Matern52(length_scale=np.ones(1), variance=1.0), solver=LBFGSB(verbosity=0))
        node13 = GPNode(kernel=Matern52(length_scale=np.ones(1), variance=1.0), solver=LBFGSB(verbosity=0))

        node11.solver.solver_iterations = 45
        node12.solver.solver_iterations = 45
        node13.solver.solver_iterations = 45

        node11.likelihood_variance.value = 1e-8
        node12.likelihood_variance.value = 1e-8
        node13.likelihood_variance.value = 1e-8

        node11.likelihood_variance.fix()
        node12.likelihood_variance.fix()
        node13.likelihood_variance.fix()

        node11.kernel.length_scale.bounds = (0.8 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node12.kernel.length_scale.bounds = (0.8 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node13.kernel.length_scale.bounds = (0.8 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Layer 2 ======================= #

        node21 = GPNode(kernel=Matern52(length_scale=np.ones(1), variance=1.0), solver=LBFGSB(verbosity=0))
        node22 = GPNode(kernel=Matern52(length_scale=np.ones(1), variance=1.0), solver=LBFGSB(verbosity=0))
        node23 = GPNode(kernel=Matern52(length_scale=np.ones(1), variance=1.0), solver=LBFGSB(verbosity=0))

        node21.solver.solver_iterations = 45
        node22.solver.solver_iterations = 45
        node23.solver.solver_iterations = 45

        node21.likelihood_variance.value = 1e-8
        node22.likelihood_variance.value = 1e-8
        node23.likelihood_variance.value = 1e-8

        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()
        node23.likelihood_variance.fix()

        node21.kernel.length_scale.bounds = (0.8 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node22.kernel.length_scale.bounds = (0.8 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node23.kernel.length_scale.bounds = (0.8 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Layer 3 ======================= #
        node31 = GPNode(kernel=Matern52(length_scale=np.ones(1), variance=1.0), solver=LBFGSB(verbosity=0))
        node31.solver.solver_iterations = 45
        node31.likelihood_variance.value = 1e-8
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (0.8 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))

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

class FConfig1(Config):
    def __init__(self, name=None):
        if isinstance(name, type(None)):
            super(FConfig1, self).__init__(name="FCFG1")
        else:
            super(FConfig1, self).__init__(name=name)

    def __call__(self, X_train, Y_train):
        
        # ========================== Layer 1 ============================ #

        node11 = GPNode(kernel=Matern52(length_scale=np.array([1.7557918122579712]), variance=1.0), solver=LBFGSB(verbosity=0))
        node11.scale.value = 0.492672602931985
        node11.likelihood_variance.value = 1e-8
        node11.scale.fix()
        node11.likelihood_variance.fix()

        nodeio = np.loadtxt(f"FCFG1/LAYER0/NODE0-io.dat")
        node11.inputs = nodeio[:, :-1]
        node11.outputs = nodeio[:, -1].reshape(-1, 1)
        #
        node12 = GPNode(kernel=Matern52(length_scale=np.array([1.7516119840435718]), variance=1.0), solver=LBFGSB(verbosity=0))        
        node12.scale.value = 0.5680049230224802
        node12.likelihood_variance.value = 1e-8
        node12.scale.fix() 
        node12.likelihood_variance.fix()

        nodeio = np.loadtxt(f"FCFG1/LAYER0/NODE1-io.dat")
        node12.inputs = nodeio[:, :-1]
        node12.outputs = nodeio[:, -1].reshape(-1, 1)        
        #      
        node13 = GPNode(kernel=Matern52(length_scale=np.array([1.743076192620457]), variance=1.0), solver=LBFGSB(verbosity=0))        
        node13.scale.value = 0.3370656594867721
        node13.likelihood_variance.value = 1e-8
        node13.scale.fix() 
        node13.likelihood_variance.fix()

        nodeio = np.loadtxt(f"FCFG1/LAYER0/NODE2-io.dat")
        node13.inputs = nodeio[:, :-1]
        node13.outputs = nodeio[:, -1].reshape(-1, 1)               

        # ====================== Layer 2 ======================= #

        node21 = GPNode(kernel=Matern52(length_scale=np.array([1.2021038551447338]), variance=1.0), solver=LBFGSB(verbosity=0))        
        node21.scale.value = 0.6489817460297123
        node21.likelihood_variance.value = 1e-8
        node21.scale.fix()
        node21.likelihood_variance.fix()

        nodeio = np.loadtxt(f"FCFG1/LAYER1/NODE0-io.dat")
        node21.inputs = nodeio[:, :-1]
        node21.outputs = nodeio[:, -1].reshape(-1, 1)        
        #
        node22 = GPNode(kernel=Matern52(length_scale=np.array([0.9924855755867731]), variance=1.0), solver=LBFGSB(verbosity=0))        
        node22.scale.value = 0.1367212103045102
        node22.likelihood_variance.value = 1e-8
        node22.scale.fix() 
        node22.likelihood_variance.fix()

        nodeio = np.loadtxt(f"FCFG1/LAYER1/NODE1-io.dat")
        node22.inputs = nodeio[:, :-1]
        node22.outputs = nodeio[:, -1].reshape(-1, 1)        
        #        
        node23 = GPNode(kernel=Matern52(length_scale=np.array([1.0588223125364282]), variance=1.0), solver=LBFGSB(verbosity=0))        
        node23.scale.value = 0.3793237043326677
        node23.likelihood_variance.value = 1e-8
        node23.scale.fix() 
        node23.likelihood_variance.fix()

        nodeio = np.loadtxt(f"FCFG1/LAYER1/NODE2-io.dat")
        node23.inputs = nodeio[:, :-1]
        node23.outputs = nodeio[:, -1].reshape(-1, 1)        
        #        

        # ====================== Layer 3 ======================= #
        node31 = GPNode(kernel=Matern52(length_scale=np.array([13.195894677508779]), variance=1.0), solver=LBFGSB(verbosity=0))        
        node31.scale.value = 0.0004991974350229773
        node31.likelihood_variance.value = 1e-8
        node31.scale.fix() 
        node31.likelihood_variance.fix()
        nodeio = np.loadtxt(f"FCFG1/LAYER2/NODE0-io.dat")
        node31.inputs = nodeio[:, :-1]
        node31.outputs = nodeio[:, -1].reshape(-1, 1)        
        # 

        # ====================== Model ======================= #

        layer1 = GPLayer(nodes=[node11, node12, node13], initialize = False)
        layer2 = GPLayer(nodes=[node21, node22, node23], initialize = False)
        layer3 = GPLayer(nodes=[node31], initialize = False)

        model = SIDGP(layers=[layer1, layer2, layer3], initialize=False)
        return model


def run_fixed(fconfig : Config, n_thread):
    print(f"==================== {fconfig.name} ====================")
    X_test = np.loadtxt("X_test.dat", delimiter="\t")
    Y_test = np.loadtxt("Y_test.dat", delimiter="\t")
    model = fconfig(None, None)
    filename = f"{fconfig.name}/model.fdmlmodel"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as modelfile:
        dump(model, modelfile)  

    mean, var = model.predict(X_test, n_impute=100, n_thread=n_thread)
    mean = mean.reshape(-1, 1)
    var = var.reshape(-1, 1)

    pred_path = os.path.abspath(f"{fconfig.name}")
    np.savetxt(f"{pred_path}/Z.dat", np.hstack([mean, var]), delimiter='\t')

    rr = rmse(Y_test.ravel(), mean.ravel())
    nrmse = rr / (np.max(Y_test) - np.min(Y_test))
    print(f"NRMSE = {np.mean(nrmse)}")            


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
    run_fixed(FConfig1(name="FCFG1"), n_thread=5)
    # run_experiment(Config1(name="CFG1-1"), n_thread=100)
    # run_experiment(Config2(), n_thread=100)
    # run_experiment(Config4(), n_thread=100)
    # run_experiment(Config5(), n_thread=100)
    # run_experiment(Config6(), n_thread=100)
    # run_experiment(Config7(), n_thread=5)



