import numpy as np

def recurrance(A, g, idx):
    if idx == 0:
        return g
    if idx == 1:
        return np.dot(A, g)
    return 2*A*np.dot(recurrance(A, g, idx-1), g) - np.dot(recurrance(A, g, idx-2), g)

def preComputer(A, G):
    n = len(A)
    l = A.shape[1]
    dict_of_matvecs = {}
    for i in range(n):
        for j in range(l):
            
        
    

def hutchMomentEstimator(A, N, l):
    """
    implements algorithm 2 of https://arxiv.org/pdf/2104.03461.pdf
    """
    assert A == A.T
    n = len(A)
    G = np.random.randn((n,l))
    tau = np.zeros(n)
    for k in range(N):
        for idx in range(l):
            TkG = 
        
    tau = tau * (np.sqrt(2/np.pi) / l*n)
    