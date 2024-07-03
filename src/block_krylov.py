import numpy as np
from copy import deepcopy

def bki(A, k=1, q=10):
    """
    implements block krylov iterative
    
    Inputs:
    A -- n times d matrix
    k -- number of iterations
    c -- multiplier

    Outputs:
    Q -- n times k matrix
    matvecs -- number of matrix vector products on A, the input matrix
    """
    n, d = A.shape[0], A.shape[1]
    q = int(q)

    k = int(k)
    Pi = np.random.normal(0, 1/np.sqrt(k), (d, k))
    Pi = Pi / np.linalg.norm(Pi, axis=0)

    APi = A @ Pi
    APi, R = np.linalg.qr(APi)
    K = deepcopy(APi)

    # generating the Krylov Subspace
    for i in range(1,q+1):
        APi = A @ (A.T @ APi)
        APi, R = np.linalg.qr(APi)
        K = np.concatenate((K, APi), axis = 1)

    # orthonormalizing columns of K
    Q, R = np.linalg.qr(K)

    # return first k columns of Q
    return Q