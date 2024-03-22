import numpy as np
from copy import deepcopy

def bki(A, eps=1, k=1, c=1, return_var="Q", q=1, q_given=False):
    """
    implements block krylov iterative
    
    Inputs:
    A -- n times d matrix
    k -- number of iterations
    c -- multiplier

    Outputs:
    Z/Q -- n times k matrix
    matvecs -- number of matrix vector products on A, the input matrix
    """
    n, d = A.shape[0], A.shape[1]
    if q_given:
        q = int(q)
    else:
        q = c * np.log(d) / np.sqrt(eps)
        q = int(np.ceil(q))

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

    # orthonormalizing columns of K to obtain K
    Q, R = np.linalg.qr(K)

    if return_var == "Q":
        return Q

    # compute M
    M = (Q.T @ A) @ (A.T @ Q)

    # compute SVD of M
    U, S, V = np.linalg.svd(M, full_matrices=True, compute_uv=True)
    Uk = U[:, :k]

    if return_var != "Q":
        return Q @ Uk