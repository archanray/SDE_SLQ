from scipy.optimize import minimize, fmin_tnc
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import pulp

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
        self.res = minimize(self.MinizeFunc, q, constraints=cons, bounds=bnds, tol=1e-10)
        return None
    
class resultObject:
    def __init__(self, x=None, func=None):
        self.x = x
        self.func = func

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
        # convert q to numpy array
        #(prob.__dict__.keys())
        print(prob._status)
        q = q.value
        q[q<=0] = 0
        q = q / np.sum(q)
        self.res = resultObject(q, prob.value)
        return None
        
class pulpL1solver:
    def __init__(self, T=None, z=None, res=None):
        self.T = T
        self.z = z
        self.res = res
    
    def abs_expr(self, expr):
        """
        Custom function to handle absolute value of an LpAffineExpression.
        """
        return expr if expr >= 0 else -expr
    
    def minimizer(self):
        """
        Inputs: 
            T: n \times d matrix
            z: n sized vector
        Outputs:
            q: n sized vector
        Solves:
            min_q \|Tq - z\|_1 s.t. \|q\|_1 and q_i >= 0
            
        Alternately:
        Solves:
            c^T 1
            subject to 
             Aq - b <= c
            -Aq + b <= c
            q^T1 = 1
            q >= 0

        Pretty, pretty, pretty slow! :(
        """
        prob = pulp.LpProblem('problem',pulp.LpMinimize)
        q = pulp.LpVariable.dicts("q", range(self.T.shape[1]), lowBound=0)
        prob += pulp.lpSum(self.abs_expr(self.T[i][j] * q[j] - self.z[i]) for i in range(self.T.shape[0]) for j in range(self.T.shape[1]))
        prob += pulp.lpSum(q[i] for i in range(self.T.shape[1])) == 1
        status = prob.solve()
        x = np.zeros(self.T.shape[-1])
        for i in range(self.T.shape[-1]):
            x[i] = q[i].value()
        self.res = resultObject(x, prob.objective.value())
        return None
    
class pgdL1Solver:
    def __init__(self, T=None, z=None, res=None):
        self.res = res
        self.T = T
        self.z = z

    def MinizeFunc(self, q):
        """
        minimizing function for L1:
        \|Tq - z\|_1
        """
        return np.linalg.norm(np.dot(self.T, q) - self.z, ord=1) + np.sum(q)-1
    
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
        # cons = ({"type": "eq", "fun": lambda x: np.sum(x)-1})
        bnds = [(0, None) for _ in range(q.shape[0])]
        self.res = fmin_tnc(self.MinizeFunc, q, bounds=bnds)
        return None