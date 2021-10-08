from __future__ import annotations
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

global_path = "."

class ERF2D:

    def __init__(self, config : int, test_name : str, n_mcs : int, pred_imp : int, mode : int = 0,
                 plot_contour : bool = True,
                 plot_surface : bool = True,
                 only_sklearn : bool = False):
        _config = {1: self.config1, 2: self.config2, 3: self.config3}
        self.paths = {'model'       : f"{global_path}/model/2D_ERF.hdf",
                      'mcs_data'    : f"{global_path}/data/erf_mcs.dat",
                      'train_data'  : f"{global_path}/data/erf_55.dat",
                      'plot'        : f"{global_path}/plots/2D",
                      'plot_contour': f"{global_path}/plots/2D/ERF_CONFIG_{config}C_{test_name}.png",
                      'plot_surface': f"{global_path}/plots/2D/ERF_CONFIG_{config}S_{test_name}.html",
                      }
        self.config = config
        self.test_name = test_name
        self.n_mcs = n_mcs
        self.pred_imp = pred_imp
        self.plot_contour = plot_contour
        self.plot_surface = plot_surface
        self.only_sklearn = only_sklearn
        X, Y = self.data()

        if mode == 0:
            # New Experiment
            if not only_sklearn:
                _config[config](X, Y)
                self.plot(X, Y)
                self.compute_mcs()
                self.sklearn_model(X, Y)
            else:
                self.sklearn_model(X, Y)
            self.plot_pdf()
        else:
            # Predict Only
            model = load_model(self.paths["model"], label=f"CONFIG_{config}/{test_name}")
            XX = np.loadtxt(self.paths["mcs_data"])[:n_mcs, :-1]
            MM, VV = model.predict(XX, n_impute=pred_imp, n_thread=4)

            file = hdf.File(self.paths["model"], "a")
            state = file[f'CONFIG_{config}/{test_name}']
            mcs_g = state["MCS"]
            mcs_sub1 = mcs_g[f"{n_mcs}"]
            mcs_sub2 = mcs_sub1[f"PRED_IMP_{pred_imp}"] if f"PRED_IMP_{pred_imp}" in mcs_sub1 else mcs_sub1.create_group(f"PRED_IMP_{pred_imp}")
            if len(mcs_sub2) == 0:
                mcs_sub2.create_dataset(name="SAMPLES", shape=XX.shape, dtype=XX.dtype, data=XX)
                mcs_sub2.create_dataset(name="MEAN", shape=MM.shape, dtype=MM.dtype, data=MM)
                mcs_sub2.create_dataset(name="VARIANCE", shape=VV.shape, dtype=VV.dtype, data=VV)
            else:
                mcs_sub2["SAMPLES"][:] = XX
                mcs_sub2["MEAN"][:] = MM
                mcs_sub2["VARIANCE"][:] = VV
            self.plot_pdf()


    @staticmethod
    def compute():
        from scipy.special import erf
        X = np.random.standard_normal((55, 2))
        Z = erf(-4 - (3 * X[:, 0]) + (2 * X[:, 1]))[:, None]

    def sklearn_model(self, X, Y):
        kernel = Matern(nu=2.5)
        model = SKGPR(kernel=kernel)
        model.fit(X, Y)
        skmodel_path = f"{global_path}/model/SKL_2D_ERF_CONFIG_{self.config}_{self.test_name}.skmodel"
        dump(model, skmodel_path)
        # Plot
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        XX, YY = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
        WW = np.c_[XX.ravel(), YY.ravel()]
        ZZ = model.predict(WW, return_std=False).reshape(XX.shape)
        contour = ax.contourf(XX, YY, ZZ, cmap=plt.get_cmap('jet'), alpha=1.0)
        cbar = fig.colorbar(contour)
        plt.xlabel(r'$\xi_1$')
        plt.ylabel(r'$\xi_2$')
        plt.savefig(f"{self.paths['plot']}/ERF_SKLEARNC_CONFIG_{self.config}_{self.test_name}.png")
        fig_surf = go.Figure()
        fig_surf.add_trace(go.Surface(x=XX, y=YY, z=ZZ, opacity=0.9, colorscale='Jet'))
        fig_surf.add_trace(go.Scatter3d(name='Samples',
                                        x=X[:, 0], y=X[:, 1], z=Y.ravel(),
                                        mode='markers', marker=dict(size=3, symbol="square", color="darkblue")))
        fig_surf.update_layout(autosize=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig_surf.write_html(f"{self.paths['plot']}/ERF_SKLEARNS_CONFIG_{self.config}_{self.test_name}.html", auto_open=False)
        # MCS Samples
        XX = np.loadtxt(self.paths['mcs_data'])[:self.n_mcs, :-1]
        MM, VV = model.predict(XX, return_std=True)
        VV = VV**2
        # Save into hdf
        # if not self.only_sklearn:
        file = hdf.File(self.paths['model'], "a")
        state = file[f'CONFIG_{self.config}/{self.test_name}']
        skg = state["SKLEARN"] if "SKLEARN" in state else state.create_group("SKLEARN")
        if "MODEL_PATH" not in skg.attrs:
            skg.attrs.create("MODEL_PATH", skmodel_path)
        skg_sub = skg["MCS"] if "MCS" in skg else skg.create_group("MCS")
        skg_mcs = skg_sub.create_group(f"{self.n_mcs}")
        skg_mcs.create_dataset(name="SAMPLES", shape=XX.shape, dtype=XX.dtype, data=XX)
        skg_mcs.create_dataset(name="MEAN", shape=MM.shape, dtype=MM.dtype, data=MM)
        skg_mcs.create_dataset(name="VARIANCE", shape=VV.shape, dtype=VV.dtype, data=VV)
        # else:
        #     file = hdf.File(self.paths['model'], "r+")
        #     skg_mcs = file[f'CONFIG_{self.config}/{self.test_name}']["SKLEARN"]["MCS"][f"{self.n_mcs}"]
        #     samples = skg_mcs["SAMPLES"]
        #     mean = skg_mcs["MEAN"]
        #     variance = skg_mcs["VARIANCE"]
        #     samples[...] = XX
        #     mean[...] = MM
        #     variance[...] = VV
        file.close()

    def compute_mcs(self):
        XX = np.loadtxt(self.paths['mcs_data'])[:self.n_mcs, :-1]
        MM, VV = self.model.predict(XX, n_impute = self.pred_imp, n_thread = 6)
        # Save into hdf
        file = hdf.File(self.paths['model'], "a")
        state = file[f'CONFIG_{self.config}/{self.test_name}']
        mcs_g = state["MCS"] if "MCS" in state else state.create_group("MCS")
        mcs_sub1 = mcs_g[f"{self.n_mcs}"] if f"{self.n_mcs}" in mcs_g else mcs_g.create_group(f"{self.n_mcs}")
        mcs_sub2 = mcs_sub1[f"PRED_IMP_{self.pred_imp}"] if f"PRED_IMP_{self.pred_imp}" in mcs_sub1 else mcs_sub1.create_group(f"PRED_IMP_{self.pred_imp}")
        if len(mcs_sub2) == 0:
            mcs_sub2.create_dataset(name="SAMPLES", shape=XX.shape, dtype=XX.dtype, data=XX)
            mcs_sub2.create_dataset(name="MEAN", shape=MM.shape, dtype=MM.dtype, data=MM)
            mcs_sub2.create_dataset(name="VARIANCE", shape=VV.shape, dtype=VV.dtype, data=VV)
        else:
            mcs_sub2["SAMPLES"][:] = XX
            mcs_sub2["MEAN"][:] = MM
            mcs_sub2["VARIANCE"][:] = VV

        file.close()
        self.plot_pdf()

    def data(self):
        _data = np.loadtxt(self.paths['train_data'])
        X, Z = _data[:, :-1], _data[:, -1][:, None]
        return X, Z

    def plot(self, X, Y):
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])

        XX, YY = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
        WW = np.c_[XX.ravel(), YY.ravel()]
        mean, var = self.model.predict(X = WW, n_impute = 50, n_thread = 4)
        ZZ = mean.reshape(XX.shape)
        if self.plot_contour:
            contour = ax.contourf(XX, YY, ZZ, cmap=plt.get_cmap('jet'), alpha=1.0)
            cbar = fig.colorbar(contour)
            plt.xlabel(r'$\xi_1$')
            plt.ylabel(r'$\xi_2$')
            plt.savefig(self.paths['plot_contour'])
        if self.plot_surface:
            fig_surf = go.Figure()
            fig_surf.add_trace(go.Surface(x=XX, y=YY, z=ZZ, opacity=0.9, colorscale='Jet'))
            fig_surf.add_trace(go.Scatter3d(name='Samples',
                                            x=X[:, 0], y=X[:, 1], z=Y.ravel(),
                                            mode='markers', marker=dict(size=3, symbol="square", color="darkblue")))
            fig_surf.update_layout(autosize=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            fig_surf.write_html(self.paths['plot_surface'], auto_open=False)

    def plot_pdf(self):
        file = hdf.File(self.paths["model"], "r")
        exp = file[f"CONFIG_{self.config}"][self.test_name]
        model_mcs = exp["MCS"][f"{self.n_mcs}"][f"PRED_IMP_{self.pred_imp}"]
        sk_mcs = exp["SKLEARN"]["MCS"][f"{self.n_mcs}"]

        mean, samples, variance = model_mcs.get("MEAN")[:], model_mcs.get("SAMPLES")[:], model_mcs.get("VARIANCE")[:]
        sk_mean, sk_samples, sk_variance = sk_mcs.get("MEAN")[:], sk_mcs.get("SAMPLES")[:], sk_mcs.get("VARIANCE")[:]
        true_mcs = np.loadtxt(self.paths["mcs_data"])[:, -1][:, None]

        df = pd.DataFrame({'DGP': mean.ravel(), 'GPR': sk_mean.ravel(), 'True': true_mcs.ravel()})
        for d in df:
            sns.distplot(df[d], kde=True, hist=False, kde_kws = {'linewidth': 1.5}, label=d)

        plt.xlim([-2, 2])
        plt.legend()
        plt.ylabel('PDF')
        plt.xlabel(r'$f(\xi_1, \xi_2)$')
        plt.grid()
        plt.savefig(f'E:/23620029-Faiz/C/SMpy/tests/plots/2D/PDF/2D_ERF_CONFIG_{self.config}_{self.test_name}.png', dpi=100)
        plt.close('all')
        file.close()

    def config1(self, X, Y):
        """ 2 Layers All Matern 52 """
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
        layer1.set_inputs(X)
        layer2.set_outputs(Y)
        self.model = SIDGP(layers=[layer1, layer2])
        self.model.train(n_iter=500, ess_burn=50)
        self.model.estimate()
        save_model(self.model, self.paths['model'], f"CONFIG_1/{self.test_name}")

    def config2(self, X, Y):
        """ 3 Layers All Matern 52 """
        node1 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node2 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node3 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
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
        layer3.set_outputs(Y)
        self.model = SIDGP(layers=[layer1, layer2, layer3])
        self.model.train(n_iter=500, ess_burn=50)
        self.model.estimate()
        save_model(self.model, self.paths['model'], f"CONFIG_2/{self.test_name}")

    def config3(self, X, Y):
        """ 3 Layers Mixture of SE and M52 """
        node1 = GPNode(kernel=SquaredExponential(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node2 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node3 = GPNode(kernel=SquaredExponential(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node4 = GPNode(kernel=Matern52(length_scale=np.array([1.0, 1.0]), variance=1.0))
        node5 = GPNode(kernel=SquaredExponential(length_scale=np.array([1.0, 1.0]), variance=1.0))
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
        layer3.set_outputs(Y)
        self.model = SIDGP(layers=[layer1, layer2, layer3])
        self.model.train(n_iter=500, ess_burn=50)
        self.model.estimate()
        save_model(self.model, self.paths['model'], f"CONFIG_3/{self.test_name}")


test_erf_2d = ERF2D(config=1, test_name="55_1", n_mcs=int(1E6), pred_imp=5, mode=0)

