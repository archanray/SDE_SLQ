import numpy as np
from copy import deepcopy
from src.block_krylov import bki
from src.stochastic_lanczos import lanczos
from src.utils import sort_abs_descending, ChebyshevWrapper
from src.moment_estimator import hutchMomentEstimator, approxChebMomentMatching
from src.distribution import Distribution, mergeDistributions

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
        localDistro = Distribution()
        
        g = np.random.randn(n)
        g = g / np.linalg.norm(g)
        T = lanczos(A, g, k) # T is a kxk matrix
        
        L, V = np.linalg.eig(T)
        weights = np.square(V[0,:]) / k # dividing by k to have the values normalized
        localDistro.set_weights(L, weights)
        
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

    P = np.eye(n) - np.dot(Z, Z.T)
    L = n
    # approximate moments
    fx = hutchMomentEstimator(np.dot(P, np.dot(A, P))/L, k, l)
    gx = deepcopy(fx)
    for i in range(len(gx)):
        gx[i] = (n*gx[i] - k*ChebyshevWrapper([0], i+1)) / (n-k)
    gx = approxChebMomentMatching(gx, N=k)
    
    # INCOMPLETE -- gx needs to be a discrete distribution
    # we have information of the values at each point. but we dont know the supports
    
    # assuming gx is a distribution
    D2 = Distribution()
    for key in gx.support.keys():
        if -1 <= key <= 1:
            D2.support[key*L] = gx.support[key]
    
    D1 = Distribution()
    D1.set_weights(Lambda, np.ones_like(Lambda)/k)
    outputDistro = mergeDistributions(D1, D2, aggregator(k, n))
    return outputDistro

