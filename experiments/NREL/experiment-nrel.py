"""
Outputs : ['RootMxc1', 'RootMyc1', 'TwrBsMxt', 'TwrBsMyt', 'Anch1Ten', 'Anch2Ten', 'Anch3Ten']
modes : 0 - Train+Predict, 1 - Train Only, 2 - Predict Only
for each new config, make a new folder config{n} -> models -> pred

Config2 == Config1 except with scale fixed
"""

import numpy as np
import h5py as hdf
import os
import sys
sys.path.insert(0, "../../")
from sklearn.metrics import mean_squared_error
from fdml.kernels import SquaredExponential, Matern52
from fdml.base_models2 import LBFGSB, PSO
from fdml.deep_models2 import GPNode, GPLayer, SIDGP
from pickle import dump, load

rmse = lambda yt, yp : mean_squared_error(yt, yp, squared=False)

N_TRAIN = 250
DATA_PATH = f"data/{N_TRAIN}"
X_TRAIN = f"{DATA_PATH}/X_train.dat"
X_TEST = f"{DATA_PATH}/X_TEST.dat"


class Config:
    def __init__(self, name, index):
        self.name = name
        self.index = index

    def __call__(self, X_train, Y_train):
        raise NotImplementedError

class Config1(Config):
    def __init__(self, name):
        super().__init__(name, 1)

    def __call__(self, X_train, Y_train):
        # Layer 1
        nftr_l1 = X_train.shape[1]
        node11 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=0))
        node12 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=0))
        node13 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=0))
        node14 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=0))
        node15 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=0))
        
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

        node11.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 2.0 * np.ones(nftr_l1))
        node12.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 2.0 * np.ones(nftr_l1))
        node13.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 2.0 * np.ones(nftr_l1))
        node14.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 2.0 * np.ones(nftr_l1))
        node15.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 2.0 * np.ones(nftr_l1))

        # Layer 2
        nftr_l2 = 5
        node21 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l2), variance=1.0), solver=LBFGSB(verbosity=0))
        node22 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l2), variance=1.0), solver=LBFGSB(verbosity=0))        
        node21.solver.solver_iterations = 30
        node22.solver.solver_iterations = 30     
        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()   

        node21.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l2), 2.0 * np.ones(nftr_l2))
        node22.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l2), 2.0 * np.ones(nftr_l2)) 

        # Layer 3
        nftr_l3 = 2
        node31 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l3), variance=1.0), solver=LBFGSB(verbosity=0))
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l3), 2.0 * np.ones(nftr_l3))     
        #                                   
        layer1 = GPLayer(nodes=[node11, node12, node13, node14, node15])
        layer2 = GPLayer(nodes=[node21, node22])
        layer3 = GPLayer(nodes=[node31])

        layer1.set_inputs(X_train)
        layer3.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2, layer3])
        print("Train Model")
        model.train(n_iter=100, ess_burn=100)
        model.estimate()
        filename = f"results/{N_TRAIN}/config1/models/{self.name}.fdmlmodel"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(f"results/{N_TRAIN}/config1/models/{self.name}.fdmlmodel", 'wb') as modelfile:
            dump(model, modelfile)

        return model

class Config2(Config):
    def __init__(self, name):
        super().__init__(name, 2)

    def __call__(self, X_train, Y_train):
        # Layer 1
        nftr_l1 = X_train.shape[1]
        node11 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=0))
        node12 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=0))
        node13 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=0))
        node14 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=0))
        node15 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l1), variance=1.0), solver=LBFGSB(verbosity=0))
        
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

        node11.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 2.0 * np.ones(nftr_l1))
        node12.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 2.0 * np.ones(nftr_l1))
        node13.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 2.0 * np.ones(nftr_l1))
        node14.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 2.0 * np.ones(nftr_l1))
        node15.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l1), 2.0 * np.ones(nftr_l1))

        # Layer 2
        nftr_l2 = 5
        node21 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l2), variance=1.0), solver=LBFGSB(verbosity=0))
        node22 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l2), variance=1.0), solver=LBFGSB(verbosity=0))        
        node21.solver.solver_iterations = 30
        node22.solver.solver_iterations = 30     
        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()   

        node21.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l2), 2.0 * np.ones(nftr_l2))
        node22.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l2), 2.0 * np.ones(nftr_l2)) 

        # Layer 3
        nftr_l3 = 2
        node31 = GPNode(kernel=Matern52(length_scale=np.ones(nftr_l3), variance=1.0), solver=LBFGSB(verbosity=0))
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-6 * np.ones(nftr_l3), 2.0 * np.ones(nftr_l3))     
        #                                   
        layer1 = GPLayer(nodes=[node11, node12, node13, node14, node15])
        layer2 = GPLayer(nodes=[node21, node22])
        layer3 = GPLayer(nodes=[node31])

        layer1.set_inputs(X_train)
        layer3.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2, layer3])
        print("Train Model")
        model.train(n_iter=100, ess_burn=100)
        model.estimate()
        filename = f"results/{N_TRAIN}/config2/models/{self.name}.fdmlmodel"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(f"results/{N_TRAIN}/config2/models/{self.name}.fdmlmodel", 'wb') as modelfile:
            dump(model, modelfile)

        return model


def RootMxc1(config : Config, n_thread, mode = 0):
    Y_train = np.loadtxt(f"{DATA_PATH}/TR-RootMxc1.dat")
    Y_test = np.loadtxt(f"{DATA_PATH}/TS-RootMxc1.dat")
    if mode == 0:
        model = config(X_TRAIN, Y_train)
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        
    elif mode == 1:
        model = config(X_TRAIN, Y_train)
    else:
        modelfile = open(f"results/{N_TRAIN}/config{config.index}/models/{config.name}.fdmlmodel", 'rb')
        model = load(modelfile)
        modelfile.close()        
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        


def RootMyc1(config : Config, n_thread, mode = 0):
    Y_train = np.loadtxt(f"{DATA_PATH}/TR-RootMyc1.dat")
    Y_test = np.loadtxt(f"{DATA_PATH}/TS-RootMyc1.dat")
    if mode == 0:
        model = config(X_TRAIN, Y_train)
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        
    elif mode == 1:
        model = config(X_TRAIN, Y_train)
    else:
        modelfile = open(f"results/{N_TRAIN}/config{config.index}/models/{config.name}.fdmlmodel", 'rb')
        model = load(modelfile)
        modelfile.close()        
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        


def TwrBsMxt(config : Config, n_thread, mode = 0):
    Y_train = np.loadtxt(f"{DATA_PATH}/TR-TwrBsMxt.dat")
    Y_test = np.loadtxt(f"{DATA_PATH}/TS-TwrBsMxt.dat")
    if mode == 0:
        model = config(X_TRAIN, Y_train)
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        
    elif mode == 1:
        model = config(X_TRAIN, Y_train)
    else:
        modelfile = open(f"results/{N_TRAIN}/config{config.index}/models/{config.name}.fdmlmodel", 'rb')
        model = load(modelfile)
        modelfile.close()        
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        


def TwrBsMyt(config : Config, n_thread, mode = 0):
    Y_train = np.loadtxt(f"{DATA_PATH}/TR-TwrBsMyt.dat")
    Y_test = np.loadtxt(f"{DATA_PATH}/TS-TwrBsMyt.dat")
    if mode == 0:
        model = config(X_TRAIN, Y_train)
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        
    elif mode == 1:
        model = config(X_TRAIN, Y_train)
    else:
        modelfile = open(f"results/{N_TRAIN}/config{config.index}/models/{config.name}.fdmlmodel", 'rb')
        model = load(modelfile)
        modelfile.close()        
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        


def Anch1Ten(config : Config, n_thread, mode = 0):
    Y_train = np.loadtxt(f"{DATA_PATH}/TR-Anch1Ten.dat")
    Y_test = np.loadtxt(f"{DATA_PATH}/TS-Anch1Ten.dat")
    if mode == 0:
        model = config(X_TRAIN, Y_train)
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        
    elif mode == 1:
        model = config(X_TRAIN, Y_train)
    else:
        modelfile = open(f"results/{N_TRAIN}/config{config.index}/models/{config.name}.fdmlmodel", 'rb')
        model = load(modelfile)
        modelfile.close()        
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        


def Anch2Ten(config : Config, n_thread, mode = 0):
    Y_train = np.loadtxt(f"{DATA_PATH}/TR-Anch2Ten.dat")
    Y_test = np.loadtxt(f"{DATA_PATH}/TS-Anch2Ten.dat")
    if mode == 0:
        model = config(X_TRAIN, Y_train)
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        
    elif mode == 1:
        model = config(X_TRAIN, Y_train)
    else:
        modelfile = open(f"results/{N_TRAIN}/config{config.index}/models/{config.name}.fdmlmodel", 'rb')
        model = load(modelfile)
        modelfile.close()        
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        


def Anch3Ten(config : Config, n_thread, mode = 0):
    Y_train = np.loadtxt(f"{DATA_PATH}/TR-Anch3Ten.dat")
    Y_test = np.loadtxt(f"{DATA_PATH}/TS-Anch3Ten.dat")
    if mode == 0:
        model = config(X_TRAIN, Y_train)
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        
    elif mode == 1:
        model = config(X_TRAIN, Y_train)
    else:
        modelfile = open(f"results/{N_TRAIN}/config{config.index}/models/{config.name}.fdmlmodel", 'rb')
        model = load(modelfile)
        modelfile.close()        
        mean, var = model.predict(X_TEST, n_impute=100, n_thread=n_thread)
        mean = mean.reshape(-1, 1)
        var = var.reshape(-1, 1)
        np.savetxt(f"results/{N_TRAIN}/config{config.index}/pred/{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
        modelrmse = rmse(Y_test.ravel(), mean.ravel())
        file = open(f"results/{N_TRAIN}/config{config.index}/rmse.dat", "a")
        file.write(f"{config.name},{modelrmse}\n")
        file.close()        



if __name__ == "__main__":
    # RootMxc1(Config1(name="RootMxc1"), n_thread=300, mode=0)    
    # RootMyc1(Config1(name="RootMyc1"), n_thread=300, mode=0)
    # TwrBsMxt(Config1(name="TwrBsMxt"), n_thread=300, mode=0)
    # TwrBsMyt(Config1(name="TwrBsMyt"), n_thread=300, mode=0)
    # Anch1Ten(Config1(name="Anch1Ten"), n_thread=300, mode=0)
    # Anch2Ten(Config1(name="Anch2Ten"), n_thread=300, mode=0)
    # Anch3Ten(Config1(name="Anch3Ten"), n_thread=300, mode=0)   

    Anch1Ten(Config1(name="Anch1Ten-1"), n_thread=300, mode=0)
    Anch1Ten(Config2(name="Anch1Ten-1"), n_thread=300, mode=0)

