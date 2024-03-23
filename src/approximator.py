import numpy as np
from copy import deepcopy
from src.block_krylov import block_krylov_iter_oth as bki
from src.stochastic_lanczos import lanczos
from src.utils import sort_abs_descending, ChebyshevWrapper
from moment_estimator import hutchMomentEstimator, approxChebMomentMatching
from distribution import Distribution, mergeDistributions

def aggregator(k, n):
    def valCalc(q1, q2):
        return (k*q1 + (n-k)*q2) / n
    return valCalc

def adder(l):
    def valCal(v1, v2):
        return v1+(v2/l)
    return adder

def slq(A, k, l, seed=0):
    # THERE IS A BUG AS IN THERE SHOULDN'T BE AN X HERE. 
    # need to define supports correctly
    # set up
    np.random.seed(seed)
    n = len(A)
    fx = np.zeros(n)
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

def kd(A, k, l, seed=0):
    """
    sde using block krylov deflation and SDE of BKM22
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
    
    D1, D2 = Distribution(), Distribution()
    D1.set_weights(fx, np.ones_like(fx)/len(fx))
    D2.set_weights(gx, np.ones_like(fx)/len(gx))
    outputDistro = mergeDistributions(D1, D2, aggregator(k, n))
    return outputDistro

