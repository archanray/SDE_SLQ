import numpy as np
from random import sample
import os

def get_data(name):
    if name == "random":
        """
        symmetric matrix with random entries
        """
        dataset_size = 5000
        data = np.random.random((dataset_size, dataset_size))
        data = (data.T + data) / 2 # set all elements to [-1, 1]
        return data, dataset_size
    return None