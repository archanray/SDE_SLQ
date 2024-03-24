import numpy as np
from copy import deepcopy

def lanczos(A, v, k, return_type="T"):
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
    alpha = np.dot(Q.T, np.dot(A, Q))
    Q_tilde = np.dot(A, Q) - alpha*Q
    # orthogonalize Q_tilde
    Q_tilde = Q_tilde - np.dot(np.dot(Q, Q.T), Q_tilde)
    T[0,0] = alpha
    
    # iteration
    for i in range(1,k):
        eta = np.linalg.norm(Q_tilde)
        # note there is a modification in the following line for local orthogonalization 
        Q[:,i] = Q_tilde / eta
        alpha = 
        Q_tilde = (np.dot(A, Q_tilde / eta)) - (alpha / eta)*Q_tilde - eta*Q
        Q = Q_tilde / eta
        
        # Set variable
        T[i,i] = alpha
        T[i, i-1] = T[i-1, i] = eta

        if return_type == "Q" or return_type == "QT":
            approx_eigvecs[:, i] = Q
        
    if return_type == "T":
        return T
    elif return_type == "Q":
        return Q
    else:
        return Q, T
    
def exact_lanczos(A,q0,k,reorth=True, return_type="QT"):
    """
    run Lanczos with reorthogonalization
    
    Input
    -----
    A : entries of diagonal matrix A
    q0 : starting vector
    k : number of iterations
    B : entries of diagonal weights for orthogonalization
    """
    
    n = len(q0)
    
    Q = np.zeros((n,k),dtype=A.dtype)
    a = np.zeros(k,dtype=A.dtype)
    b = np.zeros(k-1,dtype=A.dtype)
    
    if return_type == "QT":
        T = np.zeros((k,k))
    
    Q[:,0] = q0 / np.sqrt(q0.T@q0)
    
    for i in range(1,k+1):
        # expand Krylov space
        qi = A@Q[:,i-1] - b[i-2]*Q[:,i-2] if i>1 else A@Q[:,i-1]
        
        a[i-1] = qi.T@Q[:,i-1]
        qi -= a[i-1]*Q[:,i-1]
        
        if return_type == "QT":
            T[i-1,i-1] = a[i-1]
        
        if reorth:
            qi -= Q@(Q.T@qi) # regular GS
            #for j in range(i-1): # modified GS (a bit too slow)
            #    qi -= (qi.T@Q[:,j])*Q[:,j]
            
        if i < k:
            b[i-1] = np.sqrt(qi.T@qi)
            Q[:,i] = qi / b[i-1]
            
            if return_type == "QT":
                T[i-1,i] = T[i,i-1] = b[i-1]
    if return_type == "QT":            
        return Q, T
    else:
        return Q