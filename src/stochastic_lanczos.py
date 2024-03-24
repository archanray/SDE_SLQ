import numpy as np
from copy import deepcopy

def lanczos(A, v, k):
    """
    implements stochastic lanczos algorithm to return T and not the Q vectors
    
    check page 41 of:
    Golub, G.H. and Meurant, G., 2009. Matrices, moments and quadrature with applications (Vol. 30). Princeton University Press.
    
    + modification by Paige as per the book for local orthogonality
    """
    # init variables
    n = len(A)
    T = np.zeros((k,k))
    
    # set up
    Q = v / np.linalg.norm(v)
    alpha = np.dot(Q.T, np.dot(A, Q))
    Q_tilde = np.dot(A, Q) - alpha*Q
    T[0,0] = alpha
    
    # iteration
    for i in range(1,k):
        eta = np.linalg.norm(Q_tilde)
        # note there is a modification in the following line for local orthogonalization 
        alpha = np.dot(Q_tilde.T, np.dot(A, Q_tilde)) / (eta ** 2) - np.dot(Q_tilde.T, Q)
        Q_tilde = (np.dot(A, Q_tilde / eta)) - (alpha / eta)*Q_tilde - eta*Q
        Q = Q_tilde / eta
        
        # Set variable
        T[i,i] = alpha
        T[i, i-1] = T[i-1, i] = eta
        
    #     print("alpha, eta:", alpha, eta)
    # print("*******************************EOL**************************************************")
    return T