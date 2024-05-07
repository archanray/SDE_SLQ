import numpy as np
from copy import deepcopy
from scipy.sparse import diags

def naive_lanczos(A, v, k, return_type="T", reorth=False):
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
        if reorth:
            Q_tilde = Q_tilde - np.dot(np.dot(Q, Q.T), Q_tilde)
        
        # Set variable
        T[i,i] = alpha
        T[i, i-1] = T[i-1, i] = eta
    
    if reorth:
        T = Q.T @ A @ Q
    
    if return_type == "T":
        return T
    elif return_type == "Q":
        return Q
    else:
        return Q, T

def modified_lanczos(A, v, k, return_type="T", reorth=True):
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
        if reorth:
            Qtilde = Qtilde - np.dot(np.dot(Q,Q.T), Qtilde)
        beta = np.linalg.norm(Qtilde)
        
        if i >= 1:
            T[i,i-1] = T[i-1,i] = beta
        
        if i < k-1:
            Q[:,i+1] = Qtilde /beta

        if reorth:
            T = Q.T @ A @ Q
        
    if return_type == "T":
        return T
    elif return_type == "Q":
        return Q
    else:
        return Q, T
    
def exact_lanczos(A,q0,k,reorth=True, return_type="T"):
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
    
    Q[:,0] = q0 / np.sqrt(q0.T@q0)
    
    for i in range(1,k+1):
        # expand Krylov space
        qi = A@Q[:,i-1] - b[i-2]*Q[:,i-2] if i>1 else A@Q[:,i-1]
        
        a[i-1] = qi.T@Q[:,i-1]
        qi -= a[i-1]*Q[:,i-1]
        
        if reorth:
            qi -= Q@(Q.T@qi) # regular GS
            #for j in range(i-1): # modified GS (a bit too slow)
            #    qi -= (qi.T@Q[:,j])*Q[:,j]
            
        if i < k:
            b[i-1] = np.sqrt(qi.T@qi)
            Q[:,i] = qi / b[i-1]
     
    offset = [-1,0,1]        
    T = diags([b, a, b],offset).toarray()
    
    if return_type == "T":
        return T
    if return_type == "Q":
        return Q
    if return_type == "QT":
        return Q,T
    
def wiki_lanczos(A,v1,k,return_type="T"):
    # init values
    n = len(A)
    alpha = np.zeros(k)
    beta = np.zeros(k)
    V = np.zeros((n, k))
    w_dash = np.zeros_like(V)
    w = np.zeros_like(V)
    
    # init iteration step
    V[:,0] = v1 / np.linalg.norm(v1)
    w_dash[:,0] = A @ V[:,0]
    alpha[0] = w_dash[:,0].T @ V[:,0]
    w[:,0] = w_dash[:,0] - alpha[0] * V[:,0]
    
    for j in range(1, k):
        beta[j] = np.linalg.norm(w[:,j-1])
        if beta[j] == 0:
            v_temp = np.random.randn(n)
            v_temp /= np.linalg.norm(v_temp)
            V[:,j] = v_temp - (V @ V.T @ v_temp)
        else:
            V[:,j] = w[:,j-1] / beta[j]
        w_dash[:,j] = A @ V[:,j]
        alpha[j] = w_dash[:,j].T @ V[:,j]
        w[:,j] = w_dash[:,j] - (alpha[j] * V[:,j]) - (beta[j]*V[:,j-1])
        pass
    offset = [-1,0,1]        
    T = diags([beta[1:], alpha, beta[1:]],offset).toarray()
    if return_type == "T":
        return T
    if return_type == "Q":
        return V
    if return_type == "QT":
        return V, T