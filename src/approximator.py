import numpy as np
from copy import deepcopy
from src.block_krylov import bki
from src.lanczos import modified_lanczos as lanczos
from src.utils import normalizedChebyPolyFixedPoint
from src.moment_estimator import hutchMomentEstimator, approxChebMomentMatching
from src.distribution import Distribution, mergeDistributions
from tqdm import tqdm

def aggregator(k, n):
    def valCalc(q1, q2):
        return (k*q1 + (n-k)*q2) / n
    return valCalc

def adder(l):
    def valCal(v1, v2):
        return v1+(v2/l)
    return valCal

def slq(A, k, l, seed=0): 
    """
    implements sde using stochastic Lanczos quadrature
    """
    # set up
    np.random.seed(seed)
    n = len(A)
    outputDistro = Distribution()
    
    # main iteration
    for _ in range(l):
        g = np.random.randn(n)
        g = g / np.linalg.norm(g)
        T = lanczos(A, g, k) # T is a kxk matrix
        
        L, V = np.linalg.eig(T)
        VVT = np.dot(V, V.T)
        weights = np.sum(VVT, axis=1)
        localDistro = Distribution(L, weights)
        
        outputDistro = mergeDistributions(outputDistro, localDistro, func=adder(l))

    # returns a distribution
    return outputDistro

def bkde(A, k, l, seed=0):
    """
    implements sde using block krylov deflation and SDE of BKM22
    """
    np.random.seed(seed)
    n = len(A)
    Z, Lambda = bki(A, k, 10)
    
    # filter the Zs and lambdas here

    P = np.eye(n) - np.dot(Z, Z.T)
    L = n
    # approximate moments
    fx = hutchMomentEstimator(np.dot(P, np.dot(A, P))/L, k, l)
    fx = (n / (n-k)) * fx - (k / (n-k)) * normalizedChebyPolyFixedPoint(0, len(fx))
    supports, gx = approxChebMomentMatching(fx)
    
    # assuming gx is a distribution
    new_supports = []
    new_ps = []
    for i in range(len(supports)):
        key = supports[i]
        if -1 <= key <= 1:
            new_supports.append(key*L)
            new_ps.append(gx[i])
    D2 = Distribution(np.array(new_supports), np.array(new_ps))
    
    D1 = Distribution(Lambda, np.ones_like(Lambda)/k)
    outputDistro = mergeDistributions(D1, D2, aggregator(k, n))
    return outputDistro


def KPM(A, k, l, seed=0):
    outputDistro = Distribution()
    return outputDistro

