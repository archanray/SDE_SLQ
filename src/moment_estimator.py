import numpy as np
from copy import deepcopy
from src.utils import normalizedChebyPolyFixedPoint, jacksonDampingCoefficients, jackson_poly_coeffs, sortEigValues, aggregator
from src.optimizers import L1Solver
from src.optimizers import cvxpyL1Solver
from src.optimizers import pgdSolver
from tqdm import tqdm
import math
import numpy.polynomial as poly
import src.pgd as pgd
from src.lanczos import CTU_lanczos
from src.distribution import Distribution, mergeDistributions

def adder(l):
    def valCal(v1, v2):
        return v1+(v2/l)
    return valCal

def hutchMomentEstimator(A, N, l=1000, G=None):
    """
    implements algorithm 2 of https://arxiv.org/pdf/2104.03461.pdf
    """
    np.testing.assert_allclose(A, A.T)
    assert (N % 4 == 0)
    n = len(A)
    if G is None:
        # G = 2*np.random.binomial(1, 0.5, size=(n, l)) - 1.0
        G = np.random.normal(loc=0.,scale=1., size=(n, l))
    tau = np.zeros(N+1)
    Tkm2 = deepcopy(G)
    Tkm1 = np.dot(A, G)
    
    tau[0] = np.sum(np.multiply(G, Tkm2))
    tau[1] = np.sum(np.multiply(G, Tkm1))
    
    for k in range(2, N+1):
        # necessary computations
        Tk = 2*np.dot(A, Tkm1) - Tkm2
        tau[k] = np.sum(np.multiply(G, Tk))
        # set up for future iterations
        Tkm2 = Tkm1
        Tkm1 = Tk

    tau = tau[1:]
    tau = tau * (np.sqrt(2/np.pi) / (l*n))
    return tau

def approxChebMomentMatching(tau, method="cvxpy",cheb_vals=None):
    """
    implements algorithm 1 of https://arxiv.org/pdf/2104.03461.pdf
    """
    N = len(tau)
    nIntegers = np.array(list(range(1,N+1)))
    z = np.divide(tau, nIntegers)
    if cheb_vals is None:
        d = int(np.ceil(N**3 / 2))
    else:
        d = cheb_vals
    xs = np.linspace(-1,1,num=d+1,endpoint=True)
    TNd = np.zeros((N+1, d+1))
    for k in range(1,N+1):
        a = np.zeros(N+1)
        a[k] = 1
        TNd[k, :] = poly.chebyshev.chebval(xs, a)/(max(1,k)*np.sqrt(np.pi))
    TNd = np.sqrt(2)*TNd[1:,:]
            
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
    outputs a density function supported on [-1,1] in range \R^{0+}
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
    q = (ws * coeffs + (ws*np.sqrt(2)/(N*np.sqrt(np.pi)))) / (1 + (np.sqrt(2*np.pi) / N))
    return xs, q

def SLQMM(data, nv, k):
    n = len(data)
    V = np.random.randn(n,k)
    V /= np.linalg.norm(V, axis=0)
    LambdaStore = np.zeros((k, nv))
    WeightStore = np.zeros_like(LambdaStore)
    for i in range(k):
        T = CTU_lanczos(data, V[:, i], nv, reorth=True)
        Lambda, Vectors = np.linalg.eig(T)
        # Lambda, Vectors = sortEigValues(Lambda, Vectors)
        # print("inside:", Lambda)
        weights = np.square(Vectors[0,:])
        LambdaStore[i,:] = Lambda
        WeightStore[i,:] = weights
    # \sum_{i=1}^k \sum_{j=1}^{nv} (w_{ij}/k)*delta(x-\lambda_{ij})
    WeightStore /= k
    LambdaStore = LambdaStore.ravel()
    WeightStore = WeightStore.ravel()
    LambdaStore, WeightStore = aggregator(LambdaStore, WeightStore)
    # print("outside:", np.mean(LambdaStore, axis=0))
    return LambdaStore, WeightStore

def VRSLQMM(data, m, k):
    # assumption:
    # 1. l = nv/2
    l = int(m/2)
    n = len(data)
    V = np.random.randn(n,k)
    V /= np.linalg.norm(V, axis=0)
    LambdaStore = []
    WeightStore = []
    for i in range(k):
        Q, T = CTU_lanczos(data, V[:, i], m, reorth=True, return_type="QT")
        Lambda, Vectors = np.linalg.eig(T)
        # Lambda, Vectors = sortEigValues(Lambda, Vectors)
        # print("inside:", Lambda)
        S = []
        S_dash = []
        # assumptions (we can play around this): 
        # 1. the bound on the first constraint is 1/(n**2)
        # 2. bound on the second constraint is 2/n
        for j in range(l):
            QV = Q @ Vectors[:,j]
            constraint1 = np.linalg.norm(data @ QV - Lambda[j]*QV) <= 1/(n**2) # True
            constraint2 = Vectors[0,j]**2 <= 2/n
            if constraint1 and constraint2:
                S.append(j)
            else:
                S_dash.append(j)
        S_dash = S_dash + list(range(l,m))
        # L2 = Lambda[S_dash]
        # V2 = Vectors[:, S_dash]
        weights = np.square(Vectors[0,S_dash])
        weights = ((1-(len(S)/n)) / np.sum(weights))*weights
        mask = np.ones_like(Lambda)
        mask[S_dash] = weights
        mask[S] = mask[S] / n
        LambdaStore.append(Lambda)
        WeightStore.append(mask)
    # \sum_{i=1}^k \sum_{j=1}^{nv} (w_{ij}/k)*delta(x-\lambda_{ij})
    LambdaStore = np.array(LambdaStore)
    WeightStore = np.array(WeightStore)
    WeightStore /= k
    LambdaStore = LambdaStore.ravel()
    WeightStore = WeightStore.ravel()
    LambdaStore, WeightStore = aggregator(LambdaStore, WeightStore)
    # print("outside:", np.mean(LambdaStore, axis=0))
    return LambdaStore, WeightStore
