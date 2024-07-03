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
    keys1 = np.array(list(D1.support.keys()))
    values1 = np.array(list(D1.support.values()))
    keys2 = np.array(list(D2.support.keys()))
    values2 = np.array(list(D2.support.values()))
    values1[values1 <= 0] = 0
    values1 = values1 / np.sum(values1)
    values2[values2 <= 0] = 0
    values2 = values2 / np.sum(values2)
    distance = distFun(keys1, keys2, values1, values2)
    return distance

def jacksonDampingCoefficientsConv(N):
    """
    compute jackson damping polynomial as in https://arxiv.org/pdf/2104.03461 using convolutions
    """
    z = int(np.ceil(N/4))
    g = np.ones(2*z+1)
    c = np.convolve(np.convolve(g, g), np.convolve(g, g))
    b = c[N:2*N+2]
    return b

def jacksonDampingCoefficients(N):
    """
    compute jackson damping polynomial as in https://arxiv.org/pdf/2104.03461
    """
    bkN = np.zeros(N+1)
    for k in range(N+1):
        js = N/2 + 1 - np.abs(np.arange(-N/2-1, N/2+1-k+1))
        jplusk = N/2 + 1 - np.abs(np.arange(-N/2-1, N/2+1-k+1) + k)
        bkN[k] = np.dot(js, jplusk)
    return bkN

def altJackson(N):
    """
    https://arxiv.org/pdf/1308.5467 appendix A for jackson polynomials
    """
    ks = np.arange(0, N+1)
    alphaM = np.pi / (N+2)
    sin = np.sin(alphaM)
    cos = np.cos(alphaM)
    coses = np.cos(alphaM*ks)
    sines = np.sin(alphaM*ks)
    gkN = ((1-(ks/(N+2)))* sin * coses) + ((1/(N+2))*cos*sines)
    gkN /= sin
    return gkN

def jackson_poly_coeffs(deg):
    """
    code from Aditya
    """
    norm = (1./3)*(2*(deg/2. + 1)**3 + deg/2. + 1)
    coeffs = np.zeros(deg + 1)
    for k in range(deg + 1): 
        a = (deg/2. + 1) - np.abs(np.arange(-deg/2. - 1, deg/2. + 1 - k + 1))
        b = (deg/2. + 1) - np.abs(np.arange(-deg/2. - 1 + k, deg/2. + 1 - k + 1 + k))
        coeffs[k] = (1./norm)*np.sum((a*b))
    return coeffs

def sortEigValues(eigvals, eigvecs):
    sorted_indices = np.argsort(eigvals)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    return eigvals, eigvecs

def sortTwoArray(A, B):
    sorted_indices = np.argsort(A)
    A = A[sorted_indices]
    B = B[sorted_indices]
    return A, B
    
def aggregator(A, B):
    A, B = sortTwoArray(A, B)
    # print(A, B)
    outputA, outputB = [], []
    start, stop = 0, 0
    for i in range(1, len(A)):
        if A[i-1] == A[i]:
            stop += 1
        else:
            outputA.append(A[start])
            outputB.append(np.mean(B[start:stop+1]))
            start, stop = i, i
    
    outputA.append(A[start])
    outputB.append(np.mean(B[start:stop+1]))
    return np.array(outputA), np.array(outputB)      

# print(ChebyshevPolynomial(list(range(4)), 2).shape)
# print(aggregator(np.array([1,3,4,5,1, 5]), np.array([2,3,2,3,1,5])))