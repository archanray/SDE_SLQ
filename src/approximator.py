import numpy as np
from src.block_krylov import block_krylov_iter_oth as bki
from src.stochastic_lanczos import lanczos
from src.utils import sort_abs_descending

def findMatches(nums, support, indicesTopKMag):
    indices = []
    for j in indicesTopKMag:
        if nums[j] == support:
            indices.append(j)
    return indices

def slq(A, x, k, l, seed=0):
    # set up
    np.random.seed(seed)
    n = len(A)
    fx = np.zeros(n)
    
    # main iteration
    for _ in range(l):
        g = np.random.randn(n)
        g = g / np.linalg.norm(g)
        T = lanczos(A, g, k)
        
        L, V = np.linalg.eig(T)
        indicesTopKMag = sort_abs_descending(L, type="indices")[:k]
        for j in range(n):
            indices = findMatches(L, x[j], indicesTopKMag)
            Vk = V[:, indices]
            VkVkT = np.dot(Vk, Vk.T)
            fx[j] += VkVkT[0,0] / n
    # returns a values taken at the support points
    return fx / l

def sde(A, k):
    n = len(A)
    fx = np.zeros(n)
    Z, L = bki(A, k, 10)
    return fx

