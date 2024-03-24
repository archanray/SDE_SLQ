from scipy.optimize import minimize
import numpy as np

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
    
    def unitNormConstraint(self, x):
        """
        return x^Tx - 1
        """
        return np.linalg.norm(x) - 1
        
    def minimizer(self):
        """
        Inputs: 
            T: n \times d matrix
            z: n sized vector
        Outputs:
            q: n sized vector
        Solves:
            min_q \|Tq - z\|_1 s.t. \|q\|_1 and q_i >= 0
        """
        q = np.ones(self.T.shape[-1])
        cons = ({"type": "eq", "fun": lambda x: np.linalg.norm(x)-1})
        bnds = [(0, None) for _ in range(q.shape[0])]
        self.res = minimize(self.MinizeFunc, q, constraints=cons, bounds=bnds)
        return None
    
    
# # Example usage
# T = np.array([[2, 1], [2, 1]])
# z = np.array([3, 3])

# solver = L1Solver(T=T, z=z)
# solver.minimizer()

# print("Optimal q:", solver.res.x)
# print("Optimal value:", solver.MinizeFunc(solver.res.x))
# v = [0.90561836, 0.32527857]#solver.res.x
# v = v / np.linalg.norm(v)
# print("Optimal value:", solver.MinizeFunc(v))
