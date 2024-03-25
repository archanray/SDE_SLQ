from scipy.optimize import minimize
from scipy.optimize import linprog
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

class linprogL1Solver:
    def __init__(self, T=None, z=None, res=None):
        self.res = res
        self.T = T
        self.z = z
        
    def minimizer(self):
        """
        Solves the L1 minimization problem with unit norm and non-negativity constraint for q.

        Args:
            T: A Nxd matrix.
            z: A N-sized vector.

        Returns:
            The optimal q vector and the minimum L1 norm value.
        """
        print(self.T.shape, self.z.shape)
        N, d = self.T.shape

        # Create objective function (minimize sum of slack variables)
        c = np.ones(N + d)

        # Define upper and lower bounds (q non-negative, slacks >= 0)
        bounds = tuple((0, None) for _ in range(N + d))

        # Construct A matrix and b vector for constraints
        A = np.zeros((2*N, N + d))
        b = np.zeros(2*N)

        # Non-negativity enforcement constraints
        for i in range(N):
            A[i, i] = 1
            A[i, N + i] = 1
            b[i] = self.z[i]

        # Relationship with slack variables
        for i in range(N):
            A[N + i, i] = -1
            A[N + i, N + i] = 1
            b[N + i] = -self.z[i]

        # Unit norm constraint
        A[-1, :] = self.T.flatten()
        b[-1] = 1

        # Solve the LP problem
        self.res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
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
