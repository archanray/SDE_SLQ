import numpy as np
from random import sample
import os
from scipy.stats import ortho_group

def normalize_adj_sym(adj_mat):
    """
    code from Aditya
    """
    degs = np.linalg.norm(adj_mat, ord=1, axis=0)
    adj_mat = adj_mat*(1./np.sqrt(degs))	
    adj_mat = np.nan_to_num(adj_mat).T
    adj_mat = adj_mat*(1./np.sqrt(degs))	
    return np.nan_to_num(adj_mat)

def get_data(name, seed=1):
    # set seed for repeatale experiments
    np.random.seed(seed)
    # set up file path
    file_path = os.path.join("matrices", name+".npy")
    
    # if file exists just load it, else generate or copy from data
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            dataset = np.load(f)
        dataset_size = len(dataset)
    else:
        if name == "random":
            """
            symmetric matrix with random entries
            """
            dataset_size = 500
            data = np.random.random((dataset_size, dataset_size))
            data = (data.T + data) / 2 # set all elements to [-1, 1]
            data = data / np.linalg.norm(data, 2) # set A's spectral norm to 1
            with open(file_path, "wb") as f:
                np.save(f, data)
            return data, dataset_size
        
        if name == "hypercube":
            """
            code adapted from Aditya
            """
            d = 14
            n = pow(2, d)
            adj_mat = np.zeros((n, n))
            for i in range(n): 
                for b in range(d): 
                    adj_mat[i][i ^ pow(2, b)] = 1.
            data = normalize_adj_sym(adj_mat)
            with open(file_path, "wb") as f:
                np.save(f, data)
            return data, n
        
        if name == "gaussian":
            """
            as described in https://arxiv.org/pdf/2104.03461.pdf
            """
            n = 1000
            Lambda = np.random.normal(size=n)
            Lambda = Lambda / max(Lambda)
            V = ortho_group.rvs(n)
            data = V @ np.diag(Lambda) @ V.T
            
            with open(file_path, "wb") as f:
                np.save(f, data)
            return data, n
        
        if name == "uniform":
            """
            as described in https://arxiv.org/pdf/2104.03461.pdf
            """
            n = 1000
            Lambda = np.random.uniform(low=-1.0, high=1.0, size=n)
            Lambda = Lambda / max(Lambda)
            V = ortho_group.rvs(n)
            data = V @ np.diag(Lambda) @ V.T
            
            with open(file_path, "wb") as f:
                np.save(f, data)
            return data, n
            
    return dataset, dataset_size