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
from fdsm.utilities import save_model, load_model
from fdsm.parameters import FloatParameter, VectorParameter
from fdsm.kernels import SquaredExponential, Matern52
from fdsm.base_models import GPNode
from fdsm.deep_models import GPLayer, SIDGP

def experiment1(X_train, Y_train):
    pass

def main():
    file = hdf.File("./NREL/NREL50.hdf", "r")
    data = file["TRAIN_50"]
    X_train = data["X_train"]
    Y_train = data["Y_train"]


if __name__ == "__main__":
    main()