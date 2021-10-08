"""
EXPERIMENT 1-1 : TEST THE EFFECTS OF INCREASING N_IMPUTE IN PREDICTION WITH N_MCS = 1E6 WITH CONFIG 1 ; ERF 2D
EXPERIMENT 1-2 : CONFIG 2; NPRED = 100;  MCS = 1E6 ; ERF 2D
EXPERIMENT_1.hdf
    - SKLEARN
    - 1
        - MODEL
        - DATASET(SAMPLES)
        - 5 ; 10 ; 15; 25; 50
            - DATASET(MEAN) ; DATASET(VARIANCE)
    - 2
        - MODEL
        - DATASET(SAMPLES)
        - 100
"""

import numpy as np
import h5py as hdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Optional, List
from dataclasses import dataclass
import sys
sys.path.insert(0, "../")
#
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor as SKGPR
from sklearn.gaussian_process.kernels import Matern
#
from fdsm.utilities import save_model, load_model
from fdsm.parameters import FloatParameter, VectorParameter
from fdsm.kernels import SquaredExponential, Matern52
from fdsm.base_models import GPNode
from fdsm.deep_models import GPLayer, SIDGP

PLOT_PATH = "./1/plots"

class Experiment11:
    def __init__(self) -> None:

        # if not exists
        self.model_path = "EXPERIMENT-1-1.hdf"
        file = hdf.File(self.model_path, "a")
        # G = file["1"] if "1" in file else file.create_group("1")
        file.close()

        data = np.loadtxt("../data/erf_55.dat")
        self.X, self.Z = data[:, :-1], data[:, -1][:, None]   
        self.model = None
        # if train
        self.train_model()
        self.plot()
        # else load model
        n_pred = [10, 25, 50, 100]
        self.mcs_predict(n_pred)

    def train_model(self):
        node1 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node2 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node3 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node4 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node1.likelihood_variance.fix()
        node2.likelihood_variance.fix()
        node3.likelihood_variance.fix()
        node4.likelihood_variance.fix()
        node1.kernel.length_scale.bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
        node2.kernel.length_scale.bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
        node3.kernel.length_scale.bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
        node4.kernel.length_scale.bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
        layer1 = GPLayer(nodes=[node1, node2])
        layer2 = GPLayer(nodes=[node3])
        layer1.set_inputs(self.X)
        layer2.set_outputs(self.Z)
        self.model = SIDGP(layers=[layer1, layer2])
        self.model.train(n_iter=500, ess_burn=50)
        self.model.estimate()
        save_model(self.model, self.model_path, label=None)

    def plot(self):
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])

        XX, YY = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
        WW = np.c_[XX.ravel(), YY.ravel()]
        mean, var = self.model.predict(X = WW, n_impute = 50, n_thread = 350)
        ZZ = mean.reshape(XX.shape)
        contour = ax.contourf(XX, YY, ZZ, cmap=plt.get_cmap('jet'), alpha=1.0)
        cbar = fig.colorbar(contour)
        plt.xlabel(r'$\xi_1$')
        plt.ylabel(r'$\xi_2$')
        plt.savefig("EXPERIMENT-1-1-C.png")
        fig_surf = go.Figure()
        fig_surf.add_trace(go.Surface(x=XX, y=YY, z=ZZ, opacity=0.9, colorscale='Jet'))
        fig_surf.add_trace(go.Scatter3d(name='Samples',
                                        x=self.X[:, 0], y=self.X[:, 1], z=self.Z.ravel(),
                                        mode='markers', marker=dict(size=3, symbol="square", color="darkblue")))
        fig_surf.update_layout(autosize=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig_surf.write_html("EXPERIMENT-1-1-S.html", auto_open=False)        

    def plot_pdf(self, nn, mean, true):

        df = pd.DataFrame({'DGP': mean.ravel(), 'MCS': true.ravel()})
        for d in df:
            sns.distplot(df[d], kde=True, hist=False, kde_kws = {'linewidth': 1.5}, label=d)

        plt.xlim([-2, 2])
        plt.legend()
        plt.ylabel('PDF')
        plt.xlabel(r'$f(\xi_1, \xi_2)$')
        plt.grid()
        plt.savefig(f'EXPERIMENT-1-1-{nn}-PDF.png', dpi=100)
        plt.close('all')        

    def mcs_predict(self, n_pred):
        mcs_data = np.loadtxt("../data/erf_mcs.dat")
        experiment = hdf.File(self.model_path, "a")
        # experiment = file["1"]
        # if samples is empty
        experiment.create_dataset(name="SAMPLES", shape=mcs_data.shape, dtype=mcs_data.dtype, data=mcs_data)
        #
        XX, YY = mcs_data[:, :-1], mcs_data[:, -1][:, None]
        for nn in n_pred:
            MM, VV = self.model.predict(XX, n_impute = nn, n_thread = 350)
            # Save into hdf
            # if f""{nn}" in experiment: overwrite MEAN VARIANCE
            # else
            state = experiment.create_group(f"{nn}")
            state.create_dataset(name="MEAN", shape=MM.shape, dtype=MM.dtype, data=MM)
            state.create_dataset(name="VARIANCE", shape=VV.shape, dtype=VV.dtype, data=VV)
            self.plot_pdf(nn, MM, YY)

class Experiment12:
    def __init__(self) -> None:
        self.model = None
        self.model_path = "EXPERIMENT-1-2.hdf"
        # if not exists
        file = hdf.File(self.model_path, "a")
        # G = file["2"] if "2" in file else file.create_group("2")
        file.close()

        data = np.loadtxt("../data/erf_55.dat")
        self.X, self.Z = data[:, :-1], data[:, -1][:, None]   
        self.model = None
        # if train
        self.train_model()
        self.plot()
        n_pred = [100]
        self.mcs_predict(n_pred)        

    def plot(self):
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])

        XX, YY = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
        WW = np.c_[XX.ravel(), YY.ravel()]
        mean, var = self.model.predict(X = WW, n_impute = 50, n_thread = 350)
        ZZ = mean.reshape(XX.shape)
        contour = ax.contourf(XX, YY, ZZ, cmap=plt.get_cmap('jet'), alpha=1.0)
        cbar = fig.colorbar(contour)
        plt.xlabel(r'$\xi_1$')
        plt.ylabel(r'$\xi_2$')
        plt.savefig("EXPERIMENT-1-2-C.png")
        plt.close('all')
        fig_surf = go.Figure()
        fig_surf.add_trace(go.Surface(x=XX, y=YY, z=ZZ, opacity=0.9, colorscale='Jet'))
        fig_surf.add_trace(go.Scatter3d(name='Samples',
                                        x=self.X[:, 0], y=self.X[:, 1], z=self.Z.ravel(),
                                        mode='markers', marker=dict(size=3, symbol="square", color="darkblue")))
        fig_surf.update_layout(autosize=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig_surf.write_html("EXPERIMENT-1-2-S.html", auto_open=False)        

    def plot_pdf(self, nn, mean, true):

        df = pd.DataFrame({'DGP': mean.ravel(), 'MCS': true.ravel()})
        for d in df:
            sns.distplot(df[d], kde=True, hist=False, kde_kws = {'linewidth': 1.5}, label=d)

        plt.xlim([-2, 2])
        plt.legend()
        plt.ylabel('PDF')
        plt.xlabel(r'$f(\xi_1, \xi_2)$')
        plt.grid()
        plt.savefig(f'EXPERIMENT-1-2-{nn}-PDF.png', dpi=100)
        plt.close('all')        

    def mcs_predict(self, n_pred):
        mcs_data = np.loadtxt("../data/erf_mcs.dat")
        experiment = hdf.File(self.model_path, "a")
        # experiment = file["2"]
        # if samples is empty
        experiment.create_dataset(name="SAMPLES", shape=mcs_data.shape, dtype=mcs_data.dtype, data=mcs_data)
        #
        XX, YY = mcs_data[:, :-1], mcs_data[:, -1][:, None]
        for nn in n_pred:
            MM, VV = self.model.predict(XX, n_impute = nn, n_thread = 350)
            # Save into hdf
            # if f""{nn}" in experiment: overwrite MEAN VARIANCE
            # else
            state = experiment.create_group(f"{nn}")
            state.create_dataset(name="MEAN", shape=MM.shape, dtype=MM.dtype, data=MM)
            state.create_dataset(name="VARIANCE", shape=VV.shape, dtype=VV.dtype, data=VV)
            self.plot_pdf(nn, MM, YY)

        experiment.close()

    def train_model(self):
        """ 3 Layers Mixture of SE and M52 """
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
        layer1.set_inputs(self.X)
        layer3.set_outputs(self.Z)
        self.model = SIDGP(layers=[layer1, layer2, layer3])
        self.model.train(n_iter=500, ess_burn=50)
        self.model.estimate()
        save_model(self.model, self.model_path, label=None)        

    
if __name__ == "__main__":
    # exp = Experiment11()
    exp = Experiment12()
