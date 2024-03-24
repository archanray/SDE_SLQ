import numpy as np
from random import sample
import os

def get_data(name):
    file_path = os.path.join("matrices", name+".npy")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            dataset = np.load(f)
        dataset_size = len(dataset)
    else:
        if name == "random":
            """
            symmetric matrix with random entries
            """
            dataset_size = 5000
            data = np.random.random((dataset_size, dataset_size))
            data = (data.T + data) / 2 # set all elements to [-1, 1]
            with open(file_path, "w") as f:
                np.save(f, data)
            return data, dataset_size
    return dataset, dataset_size