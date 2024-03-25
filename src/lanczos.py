import numpy as np
from copy import deepcopy

def naive_lanczos(A, v, k, return_type="T"):
    """
    implements lanczos algorithm to return T and not the Q vectors
    
    check page 41 of:
    Golub, G.H. and Meurant, G., 2009. Matrices, moments and quadrature with applications (Vol. 30). Princeton University Press.
    
    + modification by Paige as per the book for local orthogonality
    """
    # init variables
    n = len(A)
    T = np.zeros((k,k))
    Q = np.zeros((n,k))
    
    # set up
    Q[:,0] = v / np.linalg.norm(v)
    alpha = np.dot(Q[:,0].T, np.dot(A, Q[:,0]))
    Q_tilde = np.dot(A, Q[:,0]) - alpha*Q[:,0]
    # orthogonalize Q_tilde
    Q_tilde = Q_tilde - np.dot(np.dot(Q, Q.T), Q_tilde)
    T[0,0] = alpha
    
    # iteration
    for i in range(1,k):
        eta = np.linalg.norm(Q_tilde)
        Q[:,i] = Q_tilde / eta
        AQi = np.dot(A, Q[:,i])
        # modification for finite precision stability
        alpha = np.dot(Q[:,i].T, AQi) - eta*np.dot(Q[:,i].T, Q[:,i-1])
        Q_tilde = AQi - alpha*Q[:,i] - eta**Q[:,i-1]
        # adding the following line for local orthogonalization 
        Q_tilde = Q_tilde - np.dot(np.dot(Q, Q.T), Q_tilde)
        
        # Set variable
        T[i,i] = alpha
        T[i, i-1] = T[i-1, i] = eta
        
    if return_type == "T":
        return T
    elif return_type == "Q":
        return Q
    else:
        return Q, T

def modified_lanczos(A, v, k, return_type="T"):
    """
    implements lanczos algorithm from https://arxiv.org/pdf/2105.06595.pdf
    """
    # init variables
    n = len(A)
    T = np.zeros((k,k))
    Q = np.zeros((n,k))
    
    # set up
    beta = 0
    Q[:,0] = v / np.linalg.norm(v)
    
    for i in range(k):
        if i == 0:
            Qtilde = np.dot(A, Q[:,i])
        else:
            Qtilde = np.dot(A, Q[:,i]) - beta*Q[:,i-1]
        alpha = np.dot(Qtilde.T, Q[:,i])
        T[i,i] = alpha
        Qtilde = Qtilde - alpha*Q[:,i]
        # reorthogonalization
        Qtilde = Qtilde - np.dot(np.dot(Q,Q.T), Qtilde)
        beta = np.linalg.norm(Qtilde)
        
        if i >= 1:
            T[i,i-1] = T[i-1,i] = beta
        
        if i < k-1:
            Q[:,i+1] = Qtilde /beta
    
    if return_type == "T":
        return T
    elif return_type == "Q":
        return Q
    else:
        return Q, T