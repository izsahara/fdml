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
import plotly.graph_objects as go
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
        node11 = GPNode(kernel=Matern52(length_scale=np.ones(2), variance=1.0), solver=LBFGSB(verbosity=0))
        node12 = GPNode(kernel=Matern52(length_scale=np.ones(2), variance=1.0), solver=LBFGSB(verbosity=0))

        node11.likelihood_variance.value = 1e-10
        node12.likelihood_variance.value = 1e-10

        node11.likelihood_variance.fix()
        node12.likelihood_variance.fix()

        node11.kernel.length_scale.bounds = (1e-6 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node12.kernel.length_scale.bounds = (1e-6 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))

        # ====================== Layer 2 ======================= #

        node21 = GPNode(kernel=Matern52(length_scale=np.ones(2), variance=1.0), solver=LBFGSB(verbosity=0))
        node22 = GPNode(kernel=Matern52(length_scale=np.ones(2), variance=1.0), solver=LBFGSB(verbosity=0))

        node21.likelihood_variance.value = 1e-10
        node22.likelihood_variance.value = 1e-10

        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()

        node21.kernel.length_scale.bounds = (1e-6 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        node22.kernel.length_scale.bounds = (1e-6 * np.ones(n_ftr), 3.0 * np.ones(n_ftr))
        # ====================== Layer 3 ======================= #
        node31 = GPNode(kernel=Matern52(length_scale=np.ones(2), variance=1.0), solver=LBFGSB(verbosity=0))
        node31.likelihood_variance.value = 1e-10
        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-6 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))
        # ====================== Model ======================= #

        layer1 = GPLayer(nodes=[node11, node12])
        layer2 = GPLayer(nodes=[node21, node22])
        layer3 = GPLayer(nodes=[node31])

        layer1.set_inputs(X_train)
        layer3.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2, layer3])
        model.train(n_iter=250, ess_burn=100)
        model.estimate()
        return model

def run_experiment(config: Config, n_thread : int):
    print(f"==================== {config.name} ====================")
    for exp in range(1, 3):
        print(f"EXPERIMENT {exp}")
        X_train = np.loadtxt("Xsc_train.dat", delimiter="\t")
        Y_train = np.loadtxt("Y_train.dat", delimiter="\t")
        X_test = np.loadtxt("Xsc_test.dat", delimiter="\t")
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

        # Plot
        print("Plot")
        Xo = np.loadtxt("X_train.dat", delimiter="\t")
        Yo = np.loadtxt("Y_train.dat", delimiter="\t")        

        XX, YY = np.meshgrid(np.linspace(0.7, 0.75, 100), np.linspace(1.6, 3.0, 100))
        X_plot = np.c_[XX.ravel(), YY.ravel()]
        pmean, pvar = model.predict(X_plot, n_impute=100, n_thread=n_thread)
        fig_surf = go.Figure()
        fig_surf.add_trace(go.Surface(x=XX, y=YY, z=pmean.reshape(XX.shape), opacity=0.9, colorscale='magma'))
        fig_surf.add_trace(go.Scatter3d(name='Training Samples', x=Xo[:, 0], y=Xo[:, 1], z=Yo.ravel(), mode='markers', marker=dict(size=3, symbol="square", color="darkblue")))
        fig_surf.update_layout(autosize=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig_surf.write_html(f"{pred_path}/{exp}-S.html", auto_open=False)
        

if __name__ == "__main__":
    run_experiment(Config1(name="CFG1"), n_thread=192)



