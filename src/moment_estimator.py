import numpy as np
from copy import deepcopy
from src.utils import normalizedChebyPolyFixedPoint, jacksonDampingCoefficients
from src.optimizers import L1Solver
from src.optimizers import cvxpyL1Solver
from src.optimizers import pgdSolver
from tqdm import tqdm

def hutchMomentEstimator(A, N, l):
    """
    implements algorithm 2 of https://arxiv.org/pdf/2104.03461.pdf
    """
    np.testing.assert_allclose(A, A.T)
    assert (N % 4 == 0)
    n = len(A)
    G = 2*np.random.binomial(1, 0.5, size=(n, l)) - 1.0
    tau = np.zeros(N+1)
    TAG0 = deepcopy(G)
    # run the chebyshev series below
    for k in range(N+1):
        tau[k] = np.sum(np.multiply(G, TAG0))
        if k == 0:
            TAG1 = np.dot(A, G)
        else:
            TAG2 = 2*np.dot(A, TAG1) - TAG0
            TAG0 = deepcopy(TAG1)
            TAG1 = deepcopy(TAG2)
        pass
    tau = tau[1:]
    tau = tau * (np.sqrt(2/np.pi) / l*n)
    return tau

def approxChebMomentMatching(tau, method="pgd"):
    """
    implements algorithm 1 of https://arxiv.org/pdf/2104.03461.pdf
    """
    N = len(tau)
    nIntegers = np.array(list(range(1,N+1)))
    z = np.divide(tau, nIntegers)
    d = 100#int(np.ceil(N**3 / 2))
    xs = -1.0 + (2*np.array(list(range(1,d+1)), dtype=tau.dtype) / d)
    Tkbar = np.zeros((N, d))
    for i in range(d):
        Tkbar[:, i] = normalizedChebyPolyFixedPoint(xs[i], N)
    TNd = np.divide(Tkbar, nIntegers.reshape(-1,1))
    if method == "cvxpy":
        solver = cvxpyL1Solver(TNd, z)
    if method == "pgd":
        solver = pgdSolver(TNd, z)
    if method == "optimize":
        solver = L1Solver(TNd, z)
    solver.minimizer()
    return xs, solver.res.x

def discretizedJacksonDampedKPM(tau):
    """
    implements a discretization of algorithm 6 of https://arxiv.org/pdf/2104.03461.pdf
    outputs a density function supported on [-1,1] in range \R^{>=0}
    """
    N = len(tau)
    tau = np.insert(tau, 0, 1/np.sqrt(np.pi))
    b = jacksonDampingCoefficients(N)
    b = b / b[0]
    d = 10000 # set this for discretization
    xs = -1.0 + (2*np.array(list(range(1,d+1)), dtype=tau.dtype) / d)
    # remove any 1s and -1s for stability issues of w
    xs = xs[np.where(abs(xs) != 1)]
    d = len(xs)
    Tkbar = np.zeros((d, N))
    for i in range(d):
        Tkbar[i,:] = normalizedChebyPolyFixedPoint(xs[i], N)
    Tkbar = np.insert(Tkbar, 0, np.ones(d)/np.sqrt(np.pi), axis=1) # insert as the first column
    # sum_{k=0}^N (bk / b0) . tau . Tkbar(x)
    coeffs = np.dot(b * tau, Tkbar.T)
    ws = np.ones(len(xs)) / np.sqrt(1 - xs**2)
    # q = (stilde + (w*sqrt(2)/(N*sqrt(pi)))) / (1+ sqrt(2pi)/N)
    q = (ws * coeffs + ws*np.sqrt(2)/(N*np.sqrt(np.pi))) / (1 + np.sqrt(2*np.pi) / N)
    return xs, q