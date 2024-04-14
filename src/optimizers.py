from scipy.optimize import minimize, fmin_tnc
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import pulp
import torch
from tqdm import tqdm
from torch import nn

class resultObject:
    def __init__(self, x=None, fun=None):
        self.x = x
        self.fun = fun

class L1Solver:
    def __init__(self, T=None, z=None, res=None):
        self.res = res
        self.T = T
        self.z = z

    def MinizeFunc(self, q):
        """
        minimizing function for L1:
        \|Tq - z\|_1
        """
        return np.linalg.norm(np.dot(self.T, q) - self.z, ord=1)
    
    def minimizer(self):
        """
        Inputs: 
            T: n \times d matrix
            z: n sized vector
        Outputs:
            q: n sized vector
        Solves:
            min_q \|Tq - z\|_1 s.t. \|q\|_1 and q_i >= 0
        this is slow too!
        """
        q = np.ones(self.T.shape[-1])
        cons = ({"type": "eq", "fun": lambda x: np.sum(x)-1})
        bnds = [(0, None) for _ in range(q.shape[0])]
        self.res = minimize(self.MinizeFunc, q, constraints=cons, bounds=bnds)
        return None

class cvxpyL1Solver:
    def __init__(self, T=None, z=None, res=None):
        self.res = res
        self.T = T
        self.z = z

    def minimizer(self):
        """
        Inputs: 
            T: n \times d matrix
            z: n sized vector
        Outputs:
            q: n sized vector
        Solves:
            min_q \|Tq - z\|_1 s.t. \|q\|_1 and q_i >= 0
        this is the fastest I have until now. We need linprog
        """
        N,d = self.T.shape
        # create variable
        q = cp.Variable(shape=d)
        # create constraint
        constraints = [0 <= q, cp.sum(q) == 1]
        objective = cp.Minimize(cp.norm(self.T @ q - self.z, 1))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
        q = q.value
        # q[q<=0] = 0
        # q = q / np.sum(q)
        self.res = resultObject(q, prob.value)
        return None
    
class pgdSolver:
    """
    code adapted from Aditya
    """
    def __init__(self, T=None, z=None, res=None):
        self.T = T
        self.z = z
        self.res = res
        
    def l1forward(x, A, b):
        return np.linalg.norm(np.dot(A, x)-b, ord=1)
    
    def l1backward(x, A, b):
        l1_tol = 1e-12
        v = np.dot(A, x) - b
        return np.dot(A.T, (np.abs(v) > l1_tol)*np.sign(v))
    
    def minimizer(self):