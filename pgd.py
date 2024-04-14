import numpy as np

def func_l1(x, A, b): 
    return  np.linalg.norm(A.dot(x) - b, ord=1)

def func_l2(x, A, b):
    return 0.5 * np.linalg.norm(A.dot(x) - b)**2

def grad_l2(x, A, b):
    grad = -(A.T).dot(b)
    grad = grad + (A.T).dot(A.dot(x))
    return grad

l1_tol = 1e-12
def grad_l1(x, A, b): 
    v = A.dot(x) - b
    grad = (A.T).dot((np.abs(v) > l1_tol)*np.sign(v))
    return grad

def projection_simplex(y):
    x = y.copy()
    if np.all(x >= 0) and np.sum(x) <= 1:
        return x
    x = np.clip(x, 0, np.max(x))
    if np.sum(x) <= 1:
        return x
    n = x.shape[0]
    bget = False
    x.sort()
    x = x[::-1]
    temp_sum = 0
    t_hat = 0
    for i in range(n - 1):
        temp_sum += x[i]
        t_hat = (temp_sum - 1.0) / (i + 1)
        if t_hat >= x[i + 1]:
            bget = True
            break
    if not bget:
        t_hat = (temp_sum + x[n - 1] - 1.0) / n
    return np.maximum(y - t_hat, 0)


def projection_simplex_l2(y): #Algorithm from Duchi 2008, Figure 1.
    u = np.flip(np.sort(y))
    rho = len(u) - 1 - np.argmax(((u + (1. - np.cumsum(u))/(np.arange(len(u)) + 1)) > 0)[::-1])
    lam = (1/(rho+1))*(1 - np.cumsum(u)[np.intc(rho)])
    return np.maximum(y+lam, 0)


def simplex_FW_linsolver(grad): 
    x = np.zeros(len(grad))
    x[np.argmin(grad)] = 1. 
    return x