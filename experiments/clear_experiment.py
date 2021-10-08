import h5py as hdf

file = hdf.File("./1/EXPERIMENT_1.hdf", "a")
for k in file.keys():
    del file[k]
