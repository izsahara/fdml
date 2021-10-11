import numpy as np
import h5py as hdf
import sys
sys.path.insert(0, "../")
from fdsm.utilities import save_model, load_model
from fdsm.kernels import SquaredExponential, Matern52
from fdsm.deep_models import GPNode, GPLayer, SIDGP


def config1(Xtr, Ytr, Xts):
    n_samp, n_ftr = Xtr.shape
    node1 = GPNode(kernel=SquaredExponential(length_scale=1.0, variance=1.0))
    node2 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
    node3 = GPNode(kernel=SquaredExponential(length_scale=1.0, variance=1.0))
    node4 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
    node5 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))

    node1.likelihood_variance.fix()
    node2.likelihood_variance.fix()
    node3.likelihood_variance.fix()
    node4.likelihood_variance.fix()
    node5.likelihood_variance.fix()

    node1.kernel.length_scale.bounds = (1e-6 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))
    node2.kernel.length_scale.bounds = (1e-6 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))
    node3.kernel.length_scale.bounds = (1e-6 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))
    node4.kernel.length_scale.bounds = (1e-6 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))
    node5.kernel.length_scale.bounds = (1e-6 * np.ones(n_ftr), 10.0 * np.ones(n_ftr))

    layer1 = GPLayer(nodes=[node1, node2, node3])
    layer2 = GPLayer(nodes=[node4, node5])

    layer1.set_inputs(Xtr)
    layer2.set_outputs(Ytr)

    model = SIDGP(layers=[layer1, layer2])
    model.train(n_iter=500, ess_burn=50)
    model.estimate()
    save_model(model, "./ENGINE/AIRCRAFT_ENGINE.hdf", "CFG1")
    mean, var = model.predict(Xts, n_impute=50, n_thread=250)
    return mean, var

file = hdf.File("./ENGINE/AIRCRAFT_ENGINE.hdf", "a")
dataset = file["100"]
X_train, Y_train = dataset["X_train"][:], dataset["Y_train"][:]
X_test = dataset["X_test"][:]
MEAN, VARIANCE = config1(X_train, Y_train, X_test)
results = file["RESULTS"]["CFG1"]["100"]
results.create_dataset(name="MEAN", shape=MEAN.shape, dtype=MEAN.dtype, data=MEAN)
results.create_dataset(name="VARIANCE", shape=VARIANCE.shape, dtype=VARIANCE.dtype, data=VARIANCE)
file.close()

