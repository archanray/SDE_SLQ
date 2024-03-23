import numpy as np
from copy import deepcopy

def hutchMomentEstimator(A, N, l):
    """
    implements algorithm 2 of https://arxiv.org/pdf/2104.03461.pdf
    """
    assert A == A.T
    n = len(A)
    G = np.random.randn((n,l))
    tau = np.zeros(n)
    TAG0 = deepcopy(G)
    for k in range(N):
        tau[k] = np.sum(np.multiply(G, TAG0))
        if k == 0:
            TAG1 = np.dot(A, G)
        else:
            TAG2 = 2*np.dot(A, TAG1) - TAG0
            TAG0 = deepcopy(TAG1)
            TAG1 = deepcopy(TAG2)
        pass
    tau = tau * (np.sqrt(2/np.pi) / l*n)
    return tau

def approxChebMomentMatching(A, N, tau):
    """
    implements algorithm 1 of https://arxiv.org/pdf/2104.03461.pdf
    """
    assert A == A.T
    n = len(A)
    z = np.divide(tau, list(range(1,n+1)))
    d = np.ceil(N**3 / 2)
    
    return None