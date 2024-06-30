import numpy as np
from random import sample
import os
from scipy.stats import ortho_group
import scipy.io
import urllib.request
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def normalize_adj_sym(adj_mat):
    """
    code from Aditya
    """
    degs = np.linalg.norm(adj_mat, ord=1, axis=0)
    adj_mat = adj_mat*(1./np.sqrt(degs))	
    adj_mat = np.nan_to_num(adj_mat).T
    adj_mat = adj_mat*(1./np.sqrt(degs))	
    return np.nan_to_num(adj_mat)

def get_data(name, load=True):
    # set seed for repeatale experiments
    # np.random.seed(seed)
    # set up file path
    file_path = os.path.join("matrices", name+".npy")
    
    # if file exists just load it, else generate or copy from data
    if os.path.exists(file_path) and load==True:
        with open(file_path, "rb") as f:
            dataset = np.load(f)
        dataset_size = len(dataset)
        dataset /= np.linalg.norm(dataset)
    else:
        if name == "random":
            """
            symmetric matrix with random entries
            """
            dataset_size = 500
            data = np.random.random((dataset_size, dataset_size))
            data = (data.T + data) / 2 # set all elements to [-1, 1]
            data /= np.linalg.norm(data, 2) # set A's spectral norm to 1
            with open(file_path, "wb") as f:
                np.save(f, data)
            return data, dataset_size
        
        if name == "hypercube":
            """
            code adapted from Aditya
            """
            d = 10
            n = pow(2, d)
            adj_mat = np.zeros((n, n))
            for i in range(n): 
                for b in range(d): 
                    adj_mat[i][i ^ pow(2, b)] = 1.
            data = (adj_mat + adj_mat.T) / 2 
            data /= np.linalg.norm(data, ord=2)
            print("generated data!")
            # data = normalize_adj_sym(adj_mat)
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
            data /= np.linalg.norm(data, ord=2)
            
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
            data /= np.linalg.norm(data, ord=2)
            
            with open(file_path, "wb") as f:
                np.save(f, data)
            return data, n
        
        if name == "erdos992":
            """
            as described in https://arxiv.org/pdf/2104.03461.pdf
            file download instructions:
            `wget https://suitesparse-collection-website.herokuapp.com/mat/Pajek/Erdos992.mat ./`
            """
            fname = "matrices/Erdos992.mat"
            if os.path.isfile(fname):
                mat = scipy.io.loadmat('matrices/Erdos992.mat')
            else:
                urllib.request.urlretrieve("https://suitesparse-collection-website.herokuapp.com/mat/Pajek/Erdos992.mat", fname)
                mat = scipy.io.loadmat('matrices/Erdos992.mat')
                pass
            data = mat["Problem"][0][0][2].todense()
            data /= np.linalg.norm(data, ord=2)
            
            with open(file_path, "wb") as f:
                np.save(f, data)
            return data, len(data)
        
        if name == "small_large_diagonal" or name == "low_rank_matrix":
            n = 1000
            p = 10
            if name == "small_large_diagonal":
                small_vals = np.sort(np.random.randn(n-p) / 1e+10)
            if name == "low_rank_matrix":
                small_vals = np.zeros(n-p)
            large_vals = np.sort(100*np.random.randn(p))
            diagonal = np.concatenate((large_vals, small_vals))
            data = np.diag(diagonal)
            data /= np.linalg.norm(data, ord=2)
            # plt.imshow(data, cmap="gray")
            # plt.colorbar()
            # plt.show()
            with open(file_path, "wb") as f:
                np.save(f, data)
            return data, len(data)
        
        if name == "power_law_spectrum" or name == "inverse_spectrum" or name == "square_inverse_spectrum":
            n = 1000
            if name == "power_law_spectrum":
                divisors = np.geomspace(1.0, np.power(2,n-1, dtype=float), num=n)
            if name == "inverse_spectrum":
                divisors = np.arange(1, n+1,1)
            if name == "square_inverse_spectrum":
                divisors = np.arange(1, n+1,1)**2
            diagonal = np.divide(np.ones(n), divisors)
            data = np.diag(diagonal)
            data /= np.linalg.norm(data, ord=2)
            with open(file_path, "wb") as f:
                np.save(f, data)
            return data, len(data)
            
    return dataset, dataset_size

# get_data(name="erdos992")