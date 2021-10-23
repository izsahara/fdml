"""
* ERF55.dat
* ERFMCS.dat
MODELS.hdf
    - CFG_1

ERFXY.hdf
    ATTR(CONFIG) = X
    ATTR(N_IMPUTE) = Y
    - MEAN
    - VARIANCE
"""
import numpy as np
import h5py as hdf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import sys
sys.path.insert(0, "../../../")
from fdml.kernels import SquaredExponential, Matern52
from fdml.base_models2 import LBFGSB
from fdml.deep_models2 import GPNode, GPLayer, SIDGP
from pickle import dump

MCS = np.loadtxt("../MCS.dat")
TRAIN_DATA = np.loadtxt("../TRAIN.dat")
Xmcs, Zmcs = MCS[:, :-1], MCS[:, -1][:, None]

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

        node11.likelihood_variance.fix()
        node12.likelihood_variance.fix()
        node11.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 2.0 * np.ones(nftr))
        node12.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 2.0 * np.ones(nftr))

        # Layer 2
        node21 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))
        node22 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node21.likelihood_variance.fix()
        node22.likelihood_variance.fix()
        node21.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 2.0 * np.ones(nftr))
        node22.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 2.0 * np.ones(nftr))

        #Layer 3
        node31 = GPNode(kernel=Matern52(length_scale=np.ones(nftr), variance=1.0), solver=LBFGSB(verbosity=0))

        node31.likelihood_variance.fix()
        node31.kernel.length_scale.bounds = (1e-6 * np.ones(nftr), 2.0 * np.ones(nftr))

        node11.solver.solver_iterations = 15
        node12.solver.solver_iterations = 15
        node21.solver.solver_iterations = 15
        node22.solver.solver_iterations = 15
        node31.solver.solver_iterations = 15


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

def plot(model, X, Y, name : int, index : int ):
    # Contour
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    XX, YY = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    WW = np.c_[XX.ravel(), YY.ravel()]
    ZZ, _ = model.predict(WW, n_impute = 100, n_thread = 300)
    ZZ = ZZ.reshape(XX.shape)
    contour = ax.contourf(XX, YY, ZZ, cmap=plt.get_cmap('jet'), alpha=1.0)
    cbar = fig.colorbar(contour)
    plt.xlabel(r'$\xi_1$')
    plt.ylabel(r'$\xi_2$')
    plt.savefig(f"PLOTS/{name}-{index}-C.png")
    plt.close('all')
    # Surface
    fig_surf = go.Figure()
    fig_surf.add_trace(go.Surface(x=XX, y=YY, z=ZZ, opacity=0.9, colorscale='Jet'))
    fig_surf.add_trace(go.Scatter3d(name='Samples', x=X[:, 0], y=X[:, 1], z=Y.ravel(), mode='markers', marker=dict(size=3, symbol="square", color="darkblue")))
    fig_surf.update_layout(autosize=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig_surf.write_html(f"PLOTS/{name}-{index}-S.html", auto_open=False)        

def predict_mcs(model, name : int, index : int):
    MM, VV = model.predict(Xmcs, n_impute = 100, n_thread = 300)
    np.savetxt(f"MCS/{name}_Z{index}.dat", np.hstack([MM.reshape(-1, 1), VV.reshape(-1, 1)]), delimiter='\t')
    df = pd.DataFrame({'DGP': MM.ravel(), 'MCS': Zmcs.ravel()})
    for d in df:
        sns.distplot(df[d], kde=True, hist=False, kde_kws = {'linewidth': 1.5}, label=d)

    plt.xlim([-2, 2])
    plt.legend()
    plt.ylabel('PDF')
    plt.xlabel(r'$f(\xi_1, \xi_2)$')
    plt.grid()
    plt.savefig(f'PLOTS/{name}-{index}-PDF.png', dpi=100)
    plt.close('all')

def run_experiment(config: Config, start : int, stop : int):
    X_train, Y_train = TRAIN_DATA[:, :-1], TRAIN_DATA[:, -1][:, None]

    for ii in range(start, stop+1):
        model = config(X_train, Y_train)
        modelfile = open(f"MODELS/{config.name}_{ii}.fdmlmodel", 'wb')
        dump(model, modelfile)
        modelfile.close()
        plot(model, X_train, Y_train, config.name, ii)
        predict_mcs(model, config.name, ii)

if __name__ == "__main__":
    run_experiment(Config1("CFG1"), start=1, stop=10)






