"""
[ X_train, X_test are scaled (StandardScaler) ]
* DATA100.hdf
    - X_train - Y_train - X_test - Y_test 
    - ATTR(scaler_mean) -ATTR(scaler_std) 
    
* NREL-1-100.hdf
    - MODEL
    - MEAN
    - VARIANCE

"""

import numpy as np
import h5py as hdf
import sys
sys.path.insert(0, "../../")
from fdsm.utilities import save_model, load_model
from fdsm.kernels import SquaredExponential, Matern52
from fdsm.deep_models import GPNode, GPLayer, SIDGP


def nrel_1_100():
    """ 
    CONFIG-1 : 1 HIDDEN LAYER
    """
    dataset = hdf.File("DATA100.hdf", "r")
    X_train, Y_train = dataset["X_train"][:], dataset["Y_train"][:]
    X_test = dataset["X_test"][:]
    train_samp, train_ftr = X_train.shape
    test_samp, test_ftr = X_test.shape

    # INPUTS  : ['Vhub', 'Hs', 'WaveDir', 'Tp', 'PlatformDir']
    # OUTPUTS : ['rf4_RootMxc1', 'rf4_RootMyc1',
    #            'rf4_TwrBsMxt', 'rf4_TwrBsMyt',
    #            'rf4_Anch1Ten', 'rf4_Anch2Ten', 'rf4_Anch3Ten'] 

    # LAYER 1
    node11 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
    node12 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
    node13 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
    node14 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
    node15 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))

    node11.likelihood_variance.fix()
    node12.likelihood_variance.fix()
    node13.likelihood_variance.fix()
    node14.likelihood_variance.fix()
    node15.likelihood_variance.fix()

    node11.kernel.length_scale.bounds = (1e-6 * np.ones(train_ftr), 10.0 * np.ones(train_ftr))
    node12.kernel.length_scale.bounds = (1e-6 * np.ones(train_ftr), 10.0 * np.ones(train_ftr))
    node13.kernel.length_scale.bounds = (1e-6 * np.ones(train_ftr), 10.0 * np.ones(train_ftr))
    node14.kernel.length_scale.bounds = (1e-6 * np.ones(train_ftr), 10.0 * np.ones(train_ftr))
    node15.kernel.length_scale.bounds = (1e-6 * np.ones(train_ftr), 10.0 * np.ones(train_ftr))


    # LAYER 2
    node21 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
    node22 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
    node23 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
    node24 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))
    node25 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))              
    node26 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))              
    node27 = GPNode(kernel=Matern52(length_scale=1.0, variance=1.0))          

    node21.likelihood_variance.fix()
    node22.likelihood_variance.fix()
    node23.likelihood_variance.fix()
    node24.likelihood_variance.fix()
    node25.likelihood_variance.fix()
    node26.likelihood_variance.fix()
    node27.likelihood_variance.fix()

    node21.kernel.length_scale.bounds = (1e-6 * np.ones(test_ftr), 10.0 * np.ones(test_ftr))
    node22.kernel.length_scale.bounds = (1e-6 * np.ones(test_ftr), 10.0 * np.ones(test_ftr))
    node23.kernel.length_scale.bounds = (1e-6 * np.ones(test_ftr), 10.0 * np.ones(test_ftr))
    node24.kernel.length_scale.bounds = (1e-6 * np.ones(test_ftr), 10.0 * np.ones(test_ftr))
    node25.kernel.length_scale.bounds = (1e-6 * np.ones(test_ftr), 10.0 * np.ones(test_ftr))
    node26.kernel.length_scale.bounds = (1e-6 * np.ones(test_ftr), 10.0 * np.ones(test_ftr))
    node27.kernel.length_scale.bounds = (1e-6 * np.ones(test_ftr), 10.0 * np.ones(test_ftr))


    layer1 = GPLayer(nodes=[node11, node12, node13, node14, node15])
    layer2 = GPLayer(nodes=[node21, node22, node23, node24, node25, node26, node27])

    layer1.set_inputs(X_train)
    layer2.set_outputs(Y_train)

    model = SIDGP(layers=[layer1, layer2])
    model.train(n_iter=500, ess_burn=50)
    model.estimate()
    save_model(model, "NREL-1-100.hdf", f"MODEL")
    mean, var = model.predict(X_test, n_impute=100, n_thread=250)

    results = hdf.File("NREL-1-100.hdf", "a")
    results.create_dataset(name="MEAN", shape=mean.shape, dtype=mean.dtype, data=mean)
    results.create_dataset(name="VARIANCE", shape=var.shape, dtype=var.dtype, data=var)
    results.close()


if __name__ == "__main__":
    nrel_1_100()
