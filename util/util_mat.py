import os
import h5py
import numpy as np


def load_mat(path: os.path):
    result = {}
    with h5py.File(path, mode='r') as file:
        for key in file.keys():
            result[key] = np.array(file[key])
    return result


def save_mat(path: os.path, data):
    with h5py.File(path, mode='w') as file:
        for key, value in data.items():
            file[key] = value
