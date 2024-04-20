from scipy.optimize import minimize, fmin_tnc
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import pulp
import torch
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import sys

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
        
    def proj_simplex_array(self, y):
        """
        algorithm 1 of
        L. Condat, "Fast Projection onto the Simplex and the l1 Ball," Mathematical Programming Series A., 2015
        for projection onto the probability simplex
        """
        sorted_y = np.sort(y)[::-1]
        cumsum_sorted_y = np.cumsum(sorted_y)
        thresholds = (cumsum_sorted_y - 1) / np.arange(1, len(y) + 1)
        return np.maximum(y - np.max(thresholds), 0)

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
        constraints = [0 <= q, cp.sum(q) == 1, q<=1]
        objective = cp.Minimize(cp.norm(self.T @ q - self.z, 1))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
        q = q.value
        # q[q<=0] = 0
        # q = q / np.sum(q)
        q = self.proj_simplex_array(q)
        self.res = resultObject(q, prob.value)
        # print(q)
        return None
    
class pgdSolver:
    """
    code adapted from Aditya
    """
    def __init__(self, T=None, z=None, res=None):
        self.T = T
        self.z = z
        self.res = res
        
    def l1forward(self, x):
        return np.linalg.norm(np.dot(self.T, x)-self.z, ord=1)
    
    def l1backward(self, x):
        l1_tol = 1e-12
        v = np.dot(self.T, x) - self.z
        return np.dot(self.T.T, (np.abs(v) > l1_tol)*np.sign(v))
    
    def simplex_FW_linsolver(self, grad): 
        x = np.zeros(len(grad))
        x[np.argmin(grad)] = 1. 
        return x
    
    def proj_simplex_array(self, y):
        """
        algorithm 1 of
        L. Condat, "Fast Projection onto the Simplex and the l1 Ball," Mathematical Programming Series A., 2015
        for projection onto the probability simplex
        """
        sorted_y = np.sort(y)[::-1]
        cumsum_sorted_y = np.cumsum(sorted_y)
        thresholds = (cumsum_sorted_y - 1) / np.arange(1, len(y) + 1)
        return np.maximum(y - np.max(thresholds), 0)
    
    def projection_simplex(self, y):
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

    
    def minimizer(self, max_iter=100000, tol=1e-1, plot=False):
        x = np.zeros(self.T.shape[1])
        lr = 1e-6
        x[-1] = 1.
        it = 1
        f_vals = [sys.float_info.max, self.l1forward(x)]
        #x_vals = [x]
        #grad_vals = []
        
        for i in range(it+1, max_iter):
            # print("printing this:", f_vals[-1] - f_vals[-2])
            if f_vals[-1] - f_vals[-2] < tol:
                # gradient descent
                gradient = self.l1backward(x)
                y = x - lr * gradient
                # project onto the probability simplex
                x = self.proj_simplex_array(y)
                # forward pass
                f_vals += [self.l1forward(x)]
            else:
                break
        
        # while (f_vals[-2] - f_vals[-1]) > tol:
        #     if it >= max_iter: 
        #         print('Max iter reached!')
        #         break
        #     g = self.l1backward(x)
        #     x = x + (2/(2 + it))*(self.simplex_FW_linsolver(g) - x)
        #     f_vals += [self.l1forward(x)]
        #     x_vals += [x]
        #     grad_vals += [g]
        #     it += 1
        # print(f_vals)
        
        if plot:
            plt.plot(list(range(len(f_vals[1:]))), f_vals[1:])
            plt.savefig("figures/unittests/PGD_minimization.pdf", bbox_inches='tight', dpi=200)
            plt.close("all")
        
        self.res = resultObject(x, f_vals[-1])
        return None