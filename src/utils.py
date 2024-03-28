import numpy as np
import pickle
import os

def sort_abs_descending(x, type="values"):
    """
    Sort array by absolute value in descending order.
    """
    abs_x = np.abs(x)
    idx = np.argsort(-abs_x)
    if type == "values":
        return x[idx]
    else:
        return idx

def sort_descending(x, type="values"):
    """
    Sort array in descending order.
    """
    idx = np.argsort(-x)
    if type == "values":
        return x[idx]
    else:
        return idx

def get_spectrum(input):
    """
    get spectrum of the input matrix
    """
    spectrum, _ = np.linalg.eig(input)
    spectrum = sort_abs_descending(np.real(spectrum))
    return spectrum

def saver(save_dict, name_adder="single_eval_method_"):
    dir_path = os.path.join("outputs", save_dict["dataset"])
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    if "single_eval_method_" in name_adder:
        file_path = os.path.join(dir_path, \
                                name_adder+
                                save_dict["method"]+\
                                ".pkl")
    with open(file_path, "wb") as f:
        pickle.dump(save_dict, f)
    return None

def ChebyshevPolynomial(x, n):
    """
    chebyshev polynomial using recursive calls
    """
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    return 2*np.multiply(x, ChebyshevPolynomial(x, n-1)) - ChebyshevPolynomial(x, n-2)

def ChebyshevWrapper(x, n=0, weight=np.pi):
    return np.array(ChebyshevPolynomial(x, n)) / weight

def normalizedChebyPolyFixedPoint(x, k):
    """
    chebyshev polynomial using dynamic programming
    """
    res = np.zeros(k+1)
    for i in range(k+1):
        if i == 0:
            res[i] = 1
        elif i == 1:
            res[i] = x
        else:
            res[i] = 2*x*res[i-1] - res[i-2]
    return 2*res[1:] / np.pi

def Wasserstein(D1, D2):
    from scipy.stats import wasserstein_distance as distFun
    keys1 = list(D1.support.keys())
    values1 = list(D1.support.values())
    keys2 = list(D2.support.keys())
    values2 = list(D2.support.values())
    distance = distFun(keys1, keys2, values1, values2)
    return distance

def jacksonDampingCoefficients(N):
    """
    requires N to be a multiple of N for the output to be of size N
    """
    z = int(np.ceil(N/4))
    g = np.ones(2*z+1)
    c = np.convolve(np.convolve(g, g), np.convolve(g, g))
    b = c[N:2*N+2]
    return b

# print(ChebyshevPolynomial(list(range(4)), 2).shape)