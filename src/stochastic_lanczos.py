import numpy as np

def slq(A):
    """
    implements stochastic lanczos quadrature
    inputs:
    A -- n times d matrix
    """
    n, d = A.shape[0], A.shape[1]
    return A