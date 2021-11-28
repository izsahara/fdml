import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pickle import dump, load
sys.path.insert(0, "../../")
from fdml.kernels import Matern52, SquaredExponential
from fdml.base_models2 import LBFGSB
from fdml.deep_models2 import GPNode, GPLayer, SIDGP

rmse = lambda yt, yp : mean_squared_error(yt, yp, squared=False)

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

        node11.solver.solver_iterations = 30
        node12.solver.solver_iterations = 30
        node11.likelihood_variance.fix()
        node12.likelihood_variance.fix()

        node11.kernel.length_scale.bounds = (1e-2 * np.ones(nftr), 10.0 * np.ones(nftr))
        node12.kernel.length_scale.bounds = (1e-2 * np.ones(nftr), 10.0 * np.ones(nftr))

        # Layer 2
        node21 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node22 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node21.solver.solver_iterations = 30
        node22.solver.solver_iterations = 30
        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()
        node21.kernel.length_scale.bounds = (1e-2 * np.ones(nftr), 10.0 * np.ones(nftr))
        node22.kernel.length_scale.bounds = (1e-2 * np.ones(nftr), 10.0 * np.ones(nftr))


        node31 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-2 * np.ones(nftr), 10.0 * np.ones(nftr))

        layer1 = GPLayer(nodes=[node11, node12])
        layer2 = GPLayer(nodes=[node21, node22])
        layer3 = GPLayer(nodes=[node31])

        layer1.set_inputs(X_train)
        layer3.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2, layer3])
        print("Train Model")
        model.train(n_iter=500, ess_burn=100)
        model.estimate()
        return model

def plot(model, X_train, Y_train, idx):
    pass

def run_experiment(config : Config, n = 25, n_thread = 5):
    X_train = np.loadtxt(f"{n}/{n}-X_train.dat")
    Y_train = np.loadtxt(f"{n}/{n}-Y_train.dat")
    scaler1 = MinMaxScaler()
    X_train = scaler1.fit_transform(X_train)
    model = config(X_train, Y_train.reshape(-1, 1))
    with open(f"{n}/MODELS/{config.name}.fdmlmodel", 'wb') as f:
        dump(model, f)

    X_test = np.loadtxt(f"{n}/{n}-X_test.dat")
    Y_test = np.loadtxt(f"{n}/{n}-Y_test.dat")
    # modelfile = open(f"{config.name}.fdmlmodel", 'rb')
    # model = load(modelfile)
    # modelfile.close()
    scaler2 = MinMaxScaler()
    X_test = scaler2.fit_transform(X_test)
    mean, var = model.predict(X_test, n_impute=100, n_thread=n_thread)
    mean = mean.reshape(-1, 1)
    var = var.reshape(-1, 1)
    np.savetxt(f"{n}/PRED/Z{config.name}.dat", np.hstack([mean, var]), delimiter='\t')
    modelrmse = rmse(Y_test.ravel(), mean.ravel())
    print(f"RMSE = {modelrmse}")
    file = open(f"{n}/rmse.dat", "a+")
    file.write(f"{modelrmse}")
    file.close()

if __name__ == "__main__":
    for ii in range(1, 26):
        run_experiment(Config1(f"{ii}"), n = 50, n_thread=5)





