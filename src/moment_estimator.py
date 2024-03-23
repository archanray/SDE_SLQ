import numpy as np
from copy import deepcopy
from src.utils import ChebyshevWrapper
from src.optimizers import L1Solver

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

def approxChebMomentMatching(tau, N=40):
    """
    implements algorithm 1 of https://arxiv.org/pdf/2104.03461.pdf
    """
    n = len(tau)
    nIntegers = np.array(list(range(1,n+1)))
    z = np.divide(tau, nIntegers)
    d = np.ceil(N**3 / 2)
    xs = -1 + (np.array(list(range(1,d+1))) / d) # d values
    Tkbar = np.zeros((N, d))
    for i in range(1,N+1):
        Tkbar[i-1,:] = ChebyshevWrapper(xs, i, weight=np.pi/2)
    TNd = np.divide(Tkbar, nIntegers)
    solver = L1Solver(TNd, z)
    return solver.res.x