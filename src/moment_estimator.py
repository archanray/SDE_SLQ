import numpy as np
from copy import deepcopy
from src.utils import normalizedChebyPolyFixedPoint
from src.optimizers import L1Solver, cvxpyL1Solver
from tqdm import tqdm

def hutchMomentEstimator(A, N, l):
    """
    implements algorithm 2 of https://arxiv.org/pdf/2104.03461.pdf
    """
    np.testing.assert_allclose(A, A.T)
    n = len(A)
    G = np.random.normal(0, 1/np.sqrt(l), (n,l))
    tau = np.zeros(N)
    TAG0 = deepcopy(G)
    # run the chebyshev series below
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

def approxChebMomentMatching(tau):
    """
    implements algorithm 1 of https://arxiv.org/pdf/2104.03461.pdf
    """
    N = len(tau)
    nIntegers = np.array(list(range(1,N+1)))
    z = np.divide(tau, nIntegers)
    d = 10000#int(np.ceil(N**3 / 2))
    xs = -1.0 + (2*np.array(list(range(1,d+1)), dtype=tau.dtype) / d)
    Tkbar = np.zeros((N, d))
    for i in range(d):
        Tkbar[:, i] = normalizedChebyPolyFixedPoint(xs[i], N)
    TNd = np.divide(Tkbar, nIntegers.reshape(-1,1))
    solver = cvxpyL1Solver(TNd, z)
    solver.minimizer()
    return xs, solver.res.x