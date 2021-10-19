import numpy as np
import h5py as hdf
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import compress
from pickle import dump, load
from joblib import dump as jdump
from os import mkdir
from os.path import exists
from os import getcwd as current_dir
from typing import Callable, Optional, Tuple, Union, List
from scipy.stats import norm
from .base_models2 import GPR
from .deep_models2 import SIDGP

from sklearn.gaussian_process import GaussianProcessRegressor as SKGPR

AVAILABLE_MODELS = {"RELIABILITY": [GPR, SIDGP, SKGPR]}
AVAILABLE_LEARN  = {"RELIABILITY": ["U_FXN", "EF_FXN", "ER_FXN"]}
TNone  = type(None)
TGPR   = type(GPR)
TSIDGP = type(SIDGP)
TSKGPR = type(SKGPR)


# ========================== ANALYTICAL REFRENCE MODELS ========================== #

class Analytic:
    def __init__(self, method : str):
        methods = {"FOUR_BRANCH": self.four_branch}
        if method.upper() not in list(methods.keys()):
            raise NotImplementedError
        self.method = methods[method]

    def __call__(self, XX, **kwargs) -> np.ndarray:
        return self.method(XX)

    @staticmethod
    def four_branch(XX: np.ndarray, k: int = 7) -> np.ndarray:
        assert XX.ndim == 2, 'X.ndim != 2'
        X1, X2 = XX[:, 0], XX[:, 1]
        tmp1 = 3 + (0.1 * (X1 - X2) ** 2) - ((X1 + X2) / np.sqrt(2))
        tmp2 = 3 + (0.1 * (X1 - X2) ** 2) + ((X1 + X2) / np.sqrt(2))
        tmp3 = (X1 - X2) + k / np.sqrt(2)
        tmp4 = -(X1 - X2) + k / np.sqrt(2)
        return np.min((tmp1, tmp2, tmp3, tmp4), axis=0).reshape(-1, 1)


# ============================== PROBLEM DEFINITIONS ============================= #

class Problem:

    def __init__(self,
                 name            : str,
                 X_initial       : np.ndarray,
                 Z_initial       : np.ndarray,
                 model           : Optional[Union[GPR, SIDGP, SKGPR]],
                 reference_model : Callable[..., np.ndarray],
                 log_path        : Optional[str] = None) -> None:
        self.name = name
        self.X_initial = X_initial if X_initial.ndim == 2 else X_initial.reshape(-1, 1)
        self.Z_initial = Z_initial if Z_initial.ndim == 2 else Z_initial.reshape(-1, 1)
        self.model = model
        self.reference_model = reference_model
        self.log_path = current_dir() if isinstance(log_path, TNone) else log_path
        if not exists(self.log_path):
            mkdir(self.log_path)
        self.problem_hdf = None


class Reliability(Problem):

    def __init__(self,
                 name              : str,
                 X_initial         : np.ndarray,
                 Z_initial         : np.ndarray,
                 model             : Optional[Union[GPR, SIDGP, SKGPR]],
                 reference_model   : Callable[..., np.ndarray],
                 log_path          : Optional[str] = None,
                 n_mcs             : int = int(1E6),
                 learning_function : Optional[Union[Callable[[np.ndarray, np.ndarray], int], str]] = "U_FXN",
                 experiment_label  : Optional[str] = None, **kwargs) -> None:
        super().__init__(name, X_initial, Z_initial, model, reference_model, log_path)
        self.problem_hdf = f"{log_path}/{name}.fdmlreliability"
        self.n_mcs = n_mcs

        check = [isinstance(self.model, m) for m in AVAILABLE_MODELS["RELIABILITY"]]
        if not any(check):
            raise TypeError("Unrecognized Model")

        self.model_type = list(compress(AVAILABLE_MODELS["RELIABILITY"], check))[0]

        if isinstance(learning_function, str):
            check = learning_function.upper() in AVAILABLE_LEARN["RELIABILITY"]
            if not check:
                raise TypeError("Unrecognized Learning Function")

        self.learning_function = learning_function

        file = hdf.File(self.problem_hdf, "a")
        if isinstance(experiment_label, str):
            self.experiment_label = experiment_label
            file.create_group(experiment_label)
        else:
            self.experiment_label = f"{id(model)}"
            file.create_group(self.experiment_label)
        file.close()


# ================================= ADAPTIVE MODEL =============================== #

class AdaptiveModel:

    def __init__(self, problem: Problem, **kwargs) -> None:
        self.model = problem.model
        self.reference_model = problem.reference_model
        self.model_type = problem.model_type
        self.problem = problem
        self.plot = kwargs.get("plot")
        self.X = problem.X_initial
        self.Z = problem.Z_initial

    @staticmethod
    def generate_mcs(shape: tuple) -> np.ndarray:
        samples = np.random.standard_normal(shape)
        return samples

    @staticmethod
    def u_fxn(mu: np.ndarray, std: np.ndarray) -> int:
        val = np.abs(mu.ravel()) / std.ravel()
        idx = np.argmin(val)
        return idx

    @staticmethod
    def ef_fxn(mu: np.ndarray, std: np.ndarray, a=0) -> int:
        eps = 2 * std ** 2
        t1 = (a - mu) / std
        t2 = (a - eps - mu) / std
        t3 = (a + eps - mu) / std
        eff = (mu - a) * (2 * norm.cdf(t1) - norm.cdf(t2) - norm.cdf(t3)) \
              - std * (2 * norm.pdf(t1) - norm.pdf(t2) - norm.pdf(t3)) \
              + (norm.cdf(t3) - norm.cdf(t2))
        idx = np.argmax(eff)
        return idx

    @staticmethod
    def er_fxn(mu: np.ndarray, std: np.ndarray) -> int:
        mu, std = mu.ravel(), std.ravel()
        sign = np.sign(mu).ravel()
        term1 = (-sign * mu) * norm.cdf(-sign * (mu / std))
        term2 = std * norm.pdf(mu / std)
        erf = term1 + term2
        idx = np.argmax(erf)
        return idx

    def _train(self, n_iter, ess_burn) -> None:
        if isinstance(self.model_type, TGPR):
            self.model.train()
        elif isinstance(self.model_type, TSIDGP):
            self.model.train(n_iter=n_iter, ess_burn=ess_burn)
        elif isinstance(self.model_type, TSKGPR):
            self.model.fit(self.X, self.Z)
        else:
            raise NotImplementedError

    def _predict(self, XX, n_impute: int, n_thread: int) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.model, GPR):
            MM, VV = self.model.predict(XX, return_var=True)
        elif isinstance(self.model, SIDGP):
            MM, VV = self.model.predict(XX, n_impute=n_impute, n_thread=n_thread)
        elif isinstance(self.model, SKGPR):
            MM, VV = self.model.predict(XX, return_std=True)
            VV = (VV**2)[:, None]
        else:
            raise NotImplementedError
        return MM, VV

    def _update_data(self, new_sample : np.ndarray):
        self.X = np.vstack([self.X, new_sample[:, :-1]])
        self.Z = np.vstack([self.Z, np.atleast_2d(new_sample[:, -1])])
        if isinstance(self.model_type, TGPR):
            self.model.inputs = self.X.copy()
            self.model.outputs = self.Z.copy()
        elif isinstance(self.model_type, TSIDGP):
            self.model.set_observed(self.X.copy(), self.Z.copy())
        elif isinstance(self.model_type, TSKGPR):
            pass
        else:
            raise NotImplementedError

    def _save_model(self, name : str) -> None:
        if not exists(f"{self.problem.log_path}/MODELS"):
            mkdir(f"{self.problem.log_path}/MODELS")

        if isinstance(self.model_type, TGPR):
            model_file = open(f"{self.problem.log_path}/MODELS/{self.problem.name}_{self.problem.experiment_label}_{name}.fdmlgpr", "wb")
            dump(self.model, model_file)
        elif isinstance(self.model_type, TSIDGP):
            model_file = open(f"{self.problem.log_path}/MODELS/{self.problem.name}_{self.problem.experiment_label}_{name}.fdmlsidgp", "wb")
            dump(self.model, model_file)
        elif isinstance(self.model_type, TSKGPR):
            model_file = open(f"{self.problem.log_path}/MODELS/{self.problem.name}_{self.problem.experiment_label}_{name}.sklearn", "wb")
            jdump(self.model, model_file)
        else:
            raise NotImplementedError

        model_file.close()

    def _reliability(self, n_update: int, **kwargs):
        print("Inside Reliability")

        analytical = kwargs.get("analytical")
        rfunctions = {"U_FXN": self.u_fxn, "EF_FXN": self.ef_fxn, "ER_FXN": self.er_fxn}
        call_learn = rfunctions[self.problem.learning_function] if isinstance(self.problem.learning_function, str) else self.problem.learning_function
        n_iter = kwargs["n_iter"] if "n_iter" in kwargs else 500
        ess_burn = kwargs["ess_burn"] if "ess_burn" in kwargs else 50
        n_impute = kwargs["n_impute"] if "n_impute" in kwargs else 50
        n_thread = kwargs["n_thread"] if "n_thread" in kwargs else 4

        def _learn(mu_: np.ndarray, var_: np.ndarray, k: int = 2):
            mu_ = mu_.ravel()
            var_ = var_.ravel()
            std_ = np.sqrt(var_)
            idx_ = call_learn(mu_, std_)
            p_plus = np.sum((mu_ - k * std_) <= 0) / len(mu_)
            p_minus = np.sum((mu_ + k * std_) <= 0) / len(mu_)
            pof = np.sum(mu_ <= 0) / len(mu_)
            pf_ = (p_plus - p_minus) / pof
            cov_ = np.sqrt((1 - pof) / (pof * len(mu_)))
            return idx_, pf_, cov_

        def _plot_contour(name, **contour):
            points, point_labels = [], []
            lines, line_labels = [], []
            plt.ioff()
            fig = plt.figure(1, figsize=(8, 6))
            ax = fig.add_subplot(111)
            # Plot Options
            xlim = (-7, 7)
            ylim = (-7, 7)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            Xplot, Yplot = np.meshgrid(np.linspace(*xlim, 100), np.linspace(*ylim, 100))
            Wplot = np.c_[Xplot.ravel(), Yplot.ravel()]
            Zplot = self.reference_model(Wplot)
            true_cs = ax.contour(Xplot, Yplot, Zplot.reshape(Xplot.shape), [0.0], colors='k', linewidths=1, zorder=1.0)
            lines.append(true_cs.collections[0])
            line_labels.append("True Model")

            Xsurr, Ysurr = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))
            Wsurr = np.c_[Xsurr.ravel(), Ysurr.ravel()]
            Zmodel, _ = self._predict(Wsurr, n_impute, n_thread)
            model_cs = ax.contour(Xsurr, Ysurr, Zmodel.reshape(Xsurr.shape), [0.0], colors='r', linewidths=1, zorder=1.0)
            lines.append(model_cs.collections[0])
            line_labels.append(f"{self.problem.name} Model")

            train_p = ax.scatter(self.X[:, 0], self.X[:, 1], c='blue', marker='x', s=15, zorder=1)
            points.append(train_p)
            point_labels.append("Training Samples")

            if "safe_pred" in contour:
                safe_p = ax.scatter(contour["safe_pred"][:, 0], contour["safe_pred"][:, 1],
                                     c='springgreen', marker='o', edgecolor='seagreen', s=10, alpha=0.5, zorder=0.1)
                points.append(safe_p)
                point_labels.append("Predicted Safe")

            if "fail_pred" in contour:
                fail_p = ax.scatter(contour["fail_pred"][:, 0], contour["fail_pred"][:, 1],
                                     c='lightgray', marker='o', edgecolor='dimgray', s=10, alpha=0.5, zorder=0.1)
                points.append(fail_p)
                point_labels.append("Predicted Fail")

            points_legend = plt.legend(points, point_labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), shadow=False,
                                       ncol=len(point_labels), fontsize='small', markerscale=2)
            lines_legend = plt.legend(lines, line_labels, loc='upper left', ncol=1, shadow=False,
                                      fontsize='small', markerscale=2)

            plt.gca().add_artist(points_legend)
            plt.gca().add_artist(lines_legend)
            if not exists(f"{self.problem.log_path}/PLOTS"):
                mkdir(f"{self.problem.log_path}/PLOTS")
            plt.savefig(f"{self.problem.log_path}/PLOTS/{self.problem.name}_{self.problem.experiment_label}_{name}")
            plt.close("all")


        log_file = hdf.File("DGP.fdmlreliability", "a")
        self._train(n_iter, ess_burn)
        self._save_model(name="initial")
        mcs_shape = (self.problem.n_mcs, self.X.shape[1])
        samples = self.generate_mcs(mcs_shape)
        experiment = log_file[self.problem.experiment_label]
        new_samples = experiment["new_samples"] if "new_samples" in experiment else experiment.create_group(name="new_samples")
        stats = experiment["stats"] if "stats" in experiment else experiment.create_group(name="stats")

        for up in range(n_update):
            MM, VV = self._predict(samples, n_impute, n_thread)
            idx, pf, cov = _learn(MM, VV)
            Xnew = samples[idx, :][None, :]
            Znew = self.reference_model(Xnew)
            new_samples.create_dataset(name=f"{up}", dtype=Xnew.dtype, data=np.hstack([Xnew, Znew]))
            stats.create_dataset(name=f"{up}", dtype=Xnew.dtype, data=np.array([pf, cov]))
            print(f'{self.problem.name}: Update {up}| Sample = {np.hstack([Xnew, Znew]).ravel()} | CoV = {cov:.6f} | Pf = {pf:.6f}')
            if self.plot:
                safe_pred = samples[MM.ravel() > 0, :]
                fail_pred = samples[MM.ravel() < 0, :]
                _plot_contour(up, safe_pred=safe_pred, fail_pred=fail_pred)
            if analytical:
                self._update_data(np.hstack([Xnew, Znew]))
                self._train(n_iter, ess_burn)
                self._save_model(f"{up}")

    def update(self, n_update: int, **kwargs):
        if isinstance(self.problem, Reliability):
            self._reliability(n_update, **kwargs)
        else:
            raise NotImplementedError




