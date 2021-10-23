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
sys.path.insert(0, "../../")
from fdsm.utilities import save_model, load_model
from fdsm.kernels import SquaredExponential, Matern52
from fdsm.deep_models import GPNode, GPLayer, SIDGP

MCS = np.loadtxt("MCS.dat")
Xmcs, Zmcs = MCS[:, :-1], MCS[:, -1][:, None]

def plot(model, X, Y, config : int, n_imp : int ):
    # Contour
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    XX, YY = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    WW = np.c_[XX.ravel(), YY.ravel()]
    ZZ, _ = model.predict(WW, n_impute = n_imp, n_thread = 350)
    ZZ = ZZ.reshape(XX.shape)
    contour = ax.contourf(XX, YY, ZZ, cmap=plt.get_cmap('jet'), alpha=1.0)
    cbar = fig.colorbar(contour)
    plt.xlabel(r'$\xi_1$')
    plt.ylabel(r'$\xi_2$')
    plt.savefig(f"PLOTS/{config}-{n_imp}-C.png")
    plt.close('all')
    # Surface
    fig_surf = go.Figure()
    fig_surf.add_trace(go.Surface(x=XX, y=YY, z=ZZ, opacity=0.9, colorscale='Jet'))
    fig_surf.add_trace(go.Scatter3d(name='Samples', x=X[:, 0], y=X[:, 1], z=Y.ravel(), mode='markers', marker=dict(size=3, symbol="square", color="darkblue")))
    fig_surf.update_layout(autosize=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig_surf.write_html(f"PLOTS/{config}-{n_imp}-S.html", auto_open=False)        

def predict_mcs(model, config : int, n_imp : int):
    exp = hdf.File(f"ERF-{config}-{n_imp}.hdf", "a")
    exp.attrs.create("CONFIG", config)
    exp.attrs.create("N_IMPUTE", n_imp)
    MM, VV = model.predict(Xmcs, n_impute = n_imp, n_thread = 350)
    exp.create_dataset(name="MEAN", shape=MM.shape, dtype=MM.dtype, data=MM)
    exp.create_dataset(name="VARIANCE", shape=VV.shape, dtype=VV.dtype, data=VV)

    df = pd.DataFrame({'DGP': MM.ravel(), 'MCS': Zmcs.ravel()})
    for d in df:
        sns.distplot(df[d], kde=True, hist=False, kde_kws = {'linewidth': 1.5}, label=d)

    plt.xlim([-2, 2])
    plt.legend()
    plt.ylabel('PDF')
    plt.xlabel(r'$f(\xi_1, \xi_2)$')
    plt.grid()
    plt.savefig(f'PLOTS/{config}-{n_imp}-PDF.png', dpi=100)
    plt.close('all')
    exp.close()
    
def config1(trial: int, n_imp : int):
    train_data = np.loadtxt("TRAIN.dat")
    X, Z = train_data[:, :-1], train_data[:, -1][:, None]
    if trial == 0:
        node1 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node2 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node3 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node4 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node1.likelihood_variance.fix()
        node2.likelihood_variance.fix()
        node3.likelihood_variance.fix()
        node1.kernel.length_scale.bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
        node2.kernel.length_scale.bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
        node3.kernel.length_scale.bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
        layer1 = GPLayer(nodes=[node1, node2])
        layer2 = GPLayer(nodes=[node3])
        layer1.set_inputs(X)
        layer2.set_outputs(Z)
        model = SIDGP(layers=[layer1, layer2])
        model.train(n_iter=500, ess_burn=50)
        model.estimate()
        save_model(model, "MODELS.hdf", "CFG_1")
        plot(model, X, Z, config = 1, n_imp = n_imp)
        predict_mcs(model, config = 1, n_imp = n_imp)
    else:
        model = load_model("MODELS.hdf", "CFG_1")
        plot(model, X, Z, config = 1, n_imp = n_imp)
        predict_mcs(model, config = 1, n_imp = n_imp)

def config2(trial: int, n_imp : int):
    train_data = np.loadtxt("TRAIN.dat")
    X, Z = train_data[:, :-1], train_data[:, -1][:, None]
    if trial == 0:
        node1 = GPNode(kernel=SquaredExponential(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node2 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node3 = GPNode(kernel=SquaredExponential(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node4 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node5 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node1.likelihood_variance.fix()
        node2.likelihood_variance.fix()
        node3.likelihood_variance.fix()
        node4.likelihood_variance.fix()
        node5.likelihood_variance.fix()
        node1.kernel.length_scale.bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
        node2.kernel.length_scale.bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
        node3.kernel.length_scale.bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
        node4.kernel.length_scale.bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
        node5.kernel.length_scale.bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
        layer1 = GPLayer(nodes=[node1, node2])
        layer2 = GPLayer(nodes=[node3, node4])
        layer3 = GPLayer(nodes=[node5])
        layer1.set_inputs(X)
        layer3.set_outputs(Z)
        model = SIDGP(layers=[layer1, layer2, layer3])
        model.train(n_iter=500, ess_burn=50)
        model.estimate()
        save_model(model, "MODELS.hdf", "CFG_2")
        plot(model, X, Z, config = 2, n_imp = n_imp)
        predict_mcs(model, config = 2, n_imp = n_imp) 
    else:    
        model = load_model("MODELS.hdf", "CFG_2")
        plot(model, X, Z, config = 2, n_imp = n_imp)
        predict_mcs(model, config = 2, n_imp = n_imp)        


EXPEIMENTS = [5, 25, 50, 100]
for idx, ee in enumerate(EXPEIMENTS):
    config1(trial = idx, n_imp = ee)






