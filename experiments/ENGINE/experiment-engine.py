"""
AIRCRAFT-ENGINE EXPERIMENT
ENGINE.hdf
    - DATASETS
        - 100
            - TRIAL_1 - TRIAL_2 - ..... TRIAL_25
    - CFG1
        - MODEL_100 - ..... - MODEL_{N_SAMPLES}
        - 100
            - TRIAL_1 - TRIAL_2 - ..... TRIAL_25
    - CFG2 
        - MODEL_100 - ..... - MODEL_{N_SAMPLES}
        - 100
            - TRIAL_1 - TRIAL_2 - ..... TRIAL_25

CFG1 / CFG2 .. CREATED DURING TRAINING
"""

import numpy as np
import h5py as hdf
import sys
sys.path.insert(0, "../../")
from fdsm.utilities import save_model, load_model
from fdsm.kernels import SquaredExponential, Matern52
from fdsm.deep_models import GPNode, GPLayer, SIDGP

N_SAMPLES = 100

def config1(trial : int):
    print(f"TRIAL {trial}")
    file = hdf.File("ENGINE.hdf", "a")
    dataset = file["DATASETS"][f"{N_SAMPLES}"][f"TRIAL_{trial}"]
    X_train, Y_train = dataset["X_train"][:], dataset["Y_train"][:]
    X_test = dataset["X_test"][:]    
    if trial == 1:
        n_samp, n_ftr = X_train.shape
        node1 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
        node2 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
        node3 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
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

        layer1.set_inputs(X_train)
        layer2.set_outputs(Y_train)

        model = SIDGP(layers=[layer1, layer2])
        model.train(n_iter=500, ess_burn=50)
        model.estimate()
        save_model(model, "ENGINE.hdf", f"CFG1/MODEL_{N_SAMPLES}")
    else:
        model = load_model("ENGINE.hdf", f"CFG1/MODEL_{N_SAMPLES}") 
    mean, var = model.predict(X_test, n_impute=100, n_thread=150)

    cfg = file["CFG1"][f"{N_SAMPLES}"] if trial > 1 else file["CFG1"].create_group(f"{N_SAMPLES}")
    results = cfg.create_group(f"TRIAL_{trial}")
    results.create_dataset(name="MEAN", shape=mean.shape, dtype=mean.dtype, data=mean)
    results.create_dataset(name="VARIANCE", shape=var.shape, dtype=var.dtype, data=var)
    file.close()


if __name__ == "__main__":
    for tt in range(2, 26):
        config1(trial = tt)

