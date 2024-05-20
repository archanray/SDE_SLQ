import numpy as np
from copy import deepcopy
from scipy.sparse import diags
from numpy.linalg import qr

def naive_lanczos(A, v, k, return_type="T", reorth=False):
    """
    implements lanczos algorithm to return T and not the Q vectors
    
    check page 41 of:
    Golub, G.H. and Meurant, G., 2009. Matrices, moments and quadrature with applications (Vol. 30). Princeton University Press.
    
    - modification by Paige as per the book for local orthogonality when reorth is True
    
    + for ortho doing: T = Q.T AQ
    
    - need to make this algorithm memory efficient
    """
    # init variables
    n = len(A)
    Q = np.zeros((n,k))
    Qtilde = np.zeros((n,k+1))
    alpha = np.zeros(k)
    eta = np.zeros(k-1)
    
    # init steps
    Q[:,0] = v / np.linalg.norm(v)
    alpha[0] = Q[:,0].T @ A @ Q[:,0]
    Qtilde[:,1] = (A @ Q[:,0]) - (alpha[0] * Q[:,0])
    
    for t in range(1,k):
        eta[t-1] = np.linalg.norm(Qtilde[:,t])
        Q[:,t] = Qtilde[:,t] / eta[t-1]
        if not reorth:
            alpha[t] = Q[:,t].T @ A @ Q[:,t]
            Qtilde[:,t+1] = (A @ Q[:,t]) - (alpha[t]*Q[:,t]) - (eta[t-1] * Q[:, t-1])
            pass
        else:
            alpha[t] = Q[:,t].T @ ((A @ Q[:,t]) - (eta[t-1]*Q[:,t-1]))
            Qtilde[:,t+1] = (A @ Q[:,t] - (eta[t-1]*Q[:,t-1])) - (alpha[t]*Q[:,t])
            pass
    
    if not reorth:
        T = diags([eta, alpha, eta],[-1,0,1]).toarray()
    else:
        # T = diags([eta, alpha, eta],[-1,0,1]).toarray()
        T = Q.T @ A @ Q
        pass
    
    if return_type == "T":
        return T
    elif return_type == "Q":
        return Q
    else:
        return Q, T
    
def CTU_lanczos(A, v, k, return_type="T", reorth=False):
    """
    Algo 2 of https://arxiv.org/pdf/2105.06595
    """
    # set up and init variables
    n = len(A)
    Q = np.zeros((n,k))
    alpha = np.zeros(k)
    beta = np.zeros(k-1)
    
    Q[:,0] = v / np.linalg.norm(v)
    
    for i in range(1, k+1):
        # construct the Krylov space
        if i > 1:
            qtilde = A @ Q[:, i-1] - beta[i-2]*Q[:, i-2]
        else:
            qtilde = A @ Q[:, i-1]
        
        alpha[i-1] = qtilde.T @ Q[:, i-1]
        qtilde = qtilde - alpha[i-1]*Q[:, i-1]
        
        if reorth:
            # Gram-Schmidt orthogonalization for qtilde
            qtilde = qtilde - Q[:,:i-2] @ (Q[:,:i-2].T @ qtilde)
        
        if i < k:
            beta[i-1] = np.linalg.norm(qtilde)
            Q[:, i] = qtilde / beta[i-1]
    
    
    if reorth:
        # T = Q.T @ A @ Q
        T = diags([beta, alpha, beta],[-1,0,1]).toarray()
    else:
        T = diags([beta, alpha, beta],[-1,0,1]).toarray()
            
    if return_type == "T":
        return T
    elif return_type == "Q":
        return Q
    else:
        return Q, T

