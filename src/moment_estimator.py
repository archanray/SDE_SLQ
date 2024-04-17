import numpy as np
from copy import deepcopy
from src.utils import normalizedChebyPolyFixedPoint, jacksonDampingCoefficients, jackson_poly_coeffs
from src.optimizers import L1Solver
from src.optimizers import cvxpyL1Solver
from src.optimizers import pgdSolver
from tqdm import tqdm
import math
import numpy.polynomial as poly
import src.pgd as pgd

def hutchMomentEstimator(A, N, l=1000, G=None):
    """
    implements algorithm 2 of https://arxiv.org/pdf/2104.03461.pdf
    """
    np.testing.assert_allclose(A, A.T)
    assert (N % 4 == 0)
    n = len(A)
    if G is None:
        G = 2*np.random.binomial(1, 0.5, size=(n, l)) - 1.0
    tau = np.zeros(N+1)
    Tkm2 = deepcopy(G)
    Tkm1 = np.dot(A, G)
    
    tau[0] = np.sum(np.multiply(G, Tkm2))
    tau[1] = np.sum(np.multiply(G, Tkm1))
    
    for k in range(2, N+1):
        # necessary computations
        Tk = 2*np.dot(A, Tkm1) - Tkm2
        tau[k] = np.sum(np.multiply(G, Tk))
        # set up for future iterations
        Tkm2 = Tkm1
        Tkm1 = Tk

    tau = tau[1:]
    tau = tau * (np.sqrt(2/np.pi) / (l*n))
    return tau

def approxChebMomentMatching(tau, method="cvxpy"):
    """
    implements algorithm 1 of https://arxiv.org/pdf/2104.03461.pdf
    """
    N = len(tau)
    nIntegers = np.array(list(range(1,N+1)))
    z = np.divide(tau, nIntegers)
    d = 100#int(np.ceil(N**3 / 2))
    xs = -1.0 + (2*np.array(list(range(1,d+1)), dtype=tau.dtype) / d)
    Tkbar = np.zeros((N, d))
    for i in range(d):
        Tkbar[:, i] = normalizedChebyPolyFixedPoint(xs[i], N)
    TNd = np.divide(Tkbar, nIntegers.reshape(-1,1))
    if method == "cvxpy":
        solver = cvxpyL1Solver(TNd, z)
    if method == "pgd":
        solver = pgdSolver(TNd, z)
    if method == "optimize":
        solver = L1Solver(TNd, z)
    solver.minimizer()
    return xs, solver.res.x

def discretizedJacksonDampedKPM(tau):
    """
    implements a discretization of algorithm 6 of https://arxiv.org/pdf/2104.03461.pdf
    outputs a density function supported on [-1,1] in range \R^{0+}
    """
    N = len(tau)
    tau = np.insert(tau, 0, 1/np.sqrt(np.pi))
    b = jacksonDampingCoefficients(N)
    b = b / b[0]
    d = 10000 # set this for discretization
    xs = -1.0 + (2*np.array(list(range(1,d+1)), dtype=tau.dtype) / d)
    # remove any 1s and -1s for stability issues of w
    xs = xs[np.where(abs(xs) != 1)]
    d = len(xs)
    Tkbar = np.zeros((d, N))
    for i in range(d):
        Tkbar[i,:] = normalizedChebyPolyFixedPoint(xs[i], N)
    Tkbar = np.insert(Tkbar, 0, np.ones(d)/np.sqrt(np.pi), axis=1) # insert as the first column
    # sum_{k=0}^N (bk / b0) . tau . Tkbar(x)
    coeffs = np.dot(b * tau, Tkbar.T)
    ws = np.ones(len(xs)) / np.sqrt(1 - xs**2)
    # q = (stilde + (w*sqrt(2)/(N*sqrt(pi)))) / (1+ sqrt(2pi)/N)
    q = (ws * coeffs + (ws*np.sqrt(2)/(N*np.sqrt(np.pi)))) / (1 + (np.sqrt(2*np.pi) / N))
    return xs, q

def baselineHutch(data, deg, rand_vecs=None, l=1000):
    """
    code adapted from Aditya
    """
    num_rand_vecs = l
    n = len(data)
    moments = np.zeros(deg + 1)
    if rand_vecs is None:
        rand_vecs = 2*np.random.binomial(1, 0.5, size=(n, num_rand_vecs)) - 1
    v_iminus1 = rand_vecs
    v_i = np.dot(data, rand_vecs)
    
    moments[0] = np.trace(np.matmul(rand_vecs.T, v_iminus1))/(n*num_rand_vecs)
    moments[1] = np.trace(np.matmul(rand_vecs.T, v_i))/(n*num_rand_vecs)
    for i in range(2, deg + 1):
        temp = v_i
        matmul_vec = np.dot(data, v_i)
        v_i = 2*matmul_vec - v_iminus1
        v_iminus1 = temp
        moments[i] = np.trace(np.matmul(rand_vecs.T, v_i))/(n*num_rand_vecs)
    return moments * np.sqrt(2/np.pi)

def baselineKPM(data, target_deg, num_rand_vecs = 5):
    add = math.sqrt(2/math.pi)*(1/target_deg)
    scaling = (1 + math.sqrt(2*math.pi)*(1/target_deg))
    hutch_moments = baselineHutch(data, target_deg, l=num_rand_vecs)
    
    norms = np.ones(target_deg + 1)*(2/math.pi)
    norms[0] = (1/math.pi)
    kpm_coeffs = (jackson_poly_coeffs(target_deg)*norms)*hutch_moments
    kpm_coeffs[0] += add
    kpm_coeffs = kpm_coeffs/scaling
    
    grid_size = int(1e+03)
    grid = np.linspace(-1, 1, num=grid_size, endpoint=True)
    y = np.zeros(len(grid))
    
    c_grid = grid[1:-1]
    y[1:-1] = poly.chebyshev.chebweight(c_grid)*poly.chebyshev.chebval(c_grid, kpm_coeffs)
    
    y[0] = y[1] + (y[1] - y[2])*abs(grid[2] - grid[1])
    y[-1] = y[-2] + (y[-2] - y[-3])*abs(grid[-2] - grid[-3])
    
    return grid, y

def baselineCMM(data, target_deg, num_rand_vecs = 5):
    cheb_mesh = np.arange(-0.99, 0.99, 1e-3)
    scaled_moment_matrix = np.zeros((target_deg + 1, len(cheb_mesh)))
    cheb_moments = hutchMomentEstimator(data, target_deg, l=num_rand_vecs)
    for d in range(target_deg + 1): 
        a = np.zeros(target_deg + 1)
        a[d] = 1 
        if d != 0: 
            scaled_moment_matrix[d, :] = poly.chebyshev.chebval(cheb_mesh, a)/d
        else: 
            scaled_moment_matrix[d, :] = poly.chebyshev.chebval(cheb_mesh, a)

    scaled_moments = cheb_moments[:]
    scaled_moments[1:] = cheb_moments[1:]/np.arange(1, target_deg+1)
    
    solver = cvxpyL1Solver(scaled_moment_matrix, scaled_moments)
    solver.minimizer()
    res = solver.res.x
    
    grid_size = len(res)
    grid = np.linspace(-1, 1, num=grid_size, endpoint=True)
    
    return grid, res