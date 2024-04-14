import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial as poly
from scipy import sparse
from scipy import integrate
from scipy import stats
import scipy.sparse.linalg
import math 
from collections import deque
import pandas as pd
import pickle
import time
import csv
import sbm as sbm
import pgd as pgd


import liboptpy.base_optimizer as base
import liboptpy.constr_solvers as cs
import liboptpy.step_size as ss


def load_pickle(pickle_file):
        try:
                with open(pickle_file, 'rb') as f:
                        pickle_data = pickle.load(f)
        except UnicodeDecodeError as e:
                with open(pickle_file, 'rb') as f:
                        pickle_data = pickle.load(f, encoding='latin1')
        except Exception as e:
                print('Unable to load data ', pickle_file, ':', e)
                raise
        return pickle_data


def write_pickle(data, pickle_file):
    with open(pickle_file, 'wb') as handle:
        pickle.dump(data, handle)













def cheby_moments_hutch(deg, mat_obj, matMult_fn, num_rand_vecs): 
    n = mat_obj.n
    moments = np.zeros(deg + 1)
    rand_vecs = 2*np.random.binomial(1, 0.5, size=(n, num_rand_vecs)) - 1
    # rand_vecs = np.random.randn(n, num_rand_vecs)
    v_iminus1 = rand_vecs 
    avg_sparsity = 0.0
    v_i, sp = matMult_fn(rand_vecs)
    avg_sparsity += sp
    
    moments[0] = np.trace(np.matmul(rand_vecs.T, v_iminus1))/(n*num_rand_vecs)
    moments[1] = np.trace(np.matmul(rand_vecs.T, v_i))/(n*num_rand_vecs)
    for i in range(2, deg + 1): 
        temp = v_i
        matmul_vec, sp = matMult_fn(v_i)
        v_i = 2*matmul_vec - v_iminus1
        avg_sparsity += sp
        v_iminus1 = temp
        moments[i] = np.trace(np.matmul(rand_vecs.T, v_i))/(n*num_rand_vecs)
    return (moments, avg_sparsity/(deg*mat_obj.nnz)) 

def cheby_moments_exact(deg, mat_obj, matMult_fn): 
    n = mat_obj.n
    moments = np.zeros(deg + 1)
    v_iminus1 = np.identity(n) 
    v_i = matMult_fn(v_iminus1)
    moments[0] = np.trace(v_iminus1)/(n)
    moments[1] = np.trace(v_i)/(n)
    for i in range(2, deg + 1): 
        temp = v_i
        v_i = 2*matMult_fn(v_i) - v_iminus1
        v_iminus1 = temp
        moments[i] = np.trace(v_i)/(n)
    return (moments, 0.0) 

def cheby_moments_exact_fromEigs(deg, mat_obj): 
    n = mat_obj.n
    moments = np.zeros(deg + 1)
    eigs = np.real(mat_obj.compute_eigs())
    v_iminus1 = np.ones(n) 
    v_i = eigs
    moments[0] = np.sum(v_iminus1)/(n)
    moments[1] = np.sum(v_i)/(n)
    for i in range(2, deg + 1): 
        temp = v_i
        v_i = 2*(eigs*v_i) - v_iminus1
        v_iminus1 = temp
        moments[i] = np.sum(v_i)/(n)
    return (moments, 0.0)




def create_hutch_moment_fn(mat_obj, num_rand_vecs):
    def fn(target_deg):
        return cheby_moments_hutch(target_deg, mat_obj, mat_obj.create_simple_matmul_fn(), 
                                    num_rand_vecs)
    return fn

def create_approxHutch_moment_fn(mat_obj, num_rand_vecs, col_budget):
    def fn(target_deg):
        return cheby_moments_hutch(target_deg, mat_obj, 
                                mat_obj.create_column_matmul_fn(col_budget), num_rand_vecs)
    return fn

# def create_exact_moment_fn(mat_obj):
# 	def fn(target_deg):
# 		return cheby_moments_exact(target_deg, mat_obj, mat_obj.create_simple_matmul_fn())
# 	return fn
def create_exact_moment_fn(mat_obj):
    def fn(target_deg):
        return cheby_moments_exact_fromEigs(target_deg, mat_obj)
    return fn







def jackson_poly_coeffs(deg): 
    norm = (1./3)*(2*(deg/2. + 1)**3 + deg/2. + 1)
    coeffs = np.zeros(deg + 1)
    for k in range(deg + 1): 
        a = (deg/2. + 1) - np.abs(np.arange(-deg/2. - 1, deg/2. + 1 - k + 1))
        b = (deg/2. + 1) - np.abs(np.arange(-deg/2. - 1 + k, deg/2. + 1 - k + 1 + k))
        coeffs[k] = (1./norm)*np.sum((a*b))
    return coeffs

def fejer_poly_coeffs(deg): 
    a = np.arange(0, deg + 1, 1)/deg
    return (1 - a) #Fejer
    
def cheb_poly_mesh(mesh, poly_coeffs): 
    deg = len(poly_coeffs)-1
    prev_cheb_vals = np.ones(len(mesh))
    curr_cheb_vals = mesh
    y_vals = poly_coeffs[0]*prev_cheb_vals + poly_coeffs[1]*curr_cheb_vals 
    for i in range(2, deg + 1): 
        temp = curr_cheb_vals
        curr_cheb_vals = 2*mesh*(curr_cheb_vals) - prev_cheb_vals
        prev_cheb_vals = temp 
        y_vals += poly_coeffs[i]*curr_cheb_vals
    return (mesh, y_vals)

def evaluate_sde_mesh(mesh, cheb_coeffs): 
    y = poly.chebyshev.chebweight(mesh)*poly.chebyshev.chebval(mesh, cheb_coeffs)
    return (mesh, y)









# cheb_mesh = np.arange(-0.99, 0.99, 1e-2)



# n = 100
# # A = np.random.randn(n, n)
# M = np.diag(np.random.chisquare(1., size=n))
# eigs = np.real(scipy.linalg.eigvals(M))
# eigs = np.sort(eigs/np.max(np.abs(eigs)))
# # print(eigs)
# deg = 10

# gfn = sbm.return_sbmgraph_object_filename(1000, label='cliquePlusMatching')
# moments, _ = cheby_moments_exact_fromEigs(4, load_pickle(gfn))
# moments = np.zeros(5)
# moments[0] = 1.

# moments = np.zeros(deg + 1)
# regular_moments = np.zeros(deg + 1)
# # eigs = np.real(mat_obj.compute_eigs())
# v_iminus1 = np.ones(n) 
# v_i = eigs
# moments[0] = np.sum(v_iminus1)/(n)
# moments[1] = np.sum(v_i)/(n)

# regular_moments[0] = 1 
# regular_moments[1] = np.sum(eigs)/n 

# I_d = np.eye(deg+1)
# for i in range(2, deg + 1): 
# 	temp = v_i
# 	v_i = 2*(eigs*v_i) - v_iminus1
# 	v_iminus1 = temp
# 	moments[i] = np.sum(v_i)/(n)

# 	regular_moments[i] = np.sum(np.polyval(I_d[deg - i], eigs))/n



def sde_rep_kpm(target_deg, moment_fn, add, scaling): 
    cheb_moments, sparsity = moment_fn(target_deg)
    # cheb_moments, sparsity = moments, 0.

    norms = np.ones(target_deg + 1)*(2/math.pi)
    norms[0] = (1/math.pi)
    coeffs = (jackson_poly_coeffs(target_deg)*norms)*cheb_moments
    coeffs[0] += add
    return (coeffs/scaling, sparsity, cheb_moments)


def sde_momentMatching_lp(target_deg, moment_fn, cheb_mesh): 
    cheb_moments, sparsity = moment_fn(target_deg)
    # cheb_moments, sparsity = moments, 0. 

    moment_matrix = np.zeros((target_deg + 1, len(cheb_mesh)))
    for d in range(target_deg + 1): 
        a = np.zeros(target_deg + 1)
        a[d] = 1 
        moment_matrix[d, :] = poly.chebyshev.chebval(cheb_mesh, a)

    num_vars = target_deg + 1 + len(cheb_mesh)
    num_constraints = 2*(target_deg + 1)
    
    const_matrix_ub = np.zeros((num_constraints, num_vars))
    const_matrix_eq = np.zeros((1, num_vars))
    b_ub = np.zeros(num_constraints)
    b_eq = np.ones(1)
    #CReating constraints -- for each k in [d]: |sum_i T_k(x_i)q_i - z_i| = alpha_k
    for k in range(target_deg + 1): 
        const_matrix_ub[2*k][k] = -1.
        # const_matrix_ub[2*k][target_deg + 1:] = moment_matrix[k][:]
        const_matrix_ub[2*k][target_deg + 1:] = moment_matrix[k][:]/max(1, k)

        const_matrix_ub[2*k+1][k] = -1.
        # const_matrix_ub[2*k+1][target_deg + 1:] = -moment_matrix[k][:]
        const_matrix_ub[2*k+1][target_deg + 1:] = -moment_matrix[k][:]/max(1, k)

        # b_ub[2*k], b_ub[2*k+1] = cheb_moments[k], -cheb_moments[k]
        b_ub[2*k], b_ub[2*k+1] = cheb_moments[k]/max(1, k), -cheb_moments[k]/max(1, k)

    const_matrix_eq[0][target_deg + 1:] = np.ones(len(cheb_mesh))

    c = np.zeros(num_vars)
    c[:target_deg + 1] = np.ones(target_deg + 1)
    # res = scipy.optimize.linprog(c, A_ub=const_matrix_ub, b_ub=b_ub, A_eq=const_matrix_eq,
    # 	b_eq=b_eq, bounds = (0, np.inf))
    # interior-point

    res = scipy.optimize.linprog(c, A_ub=const_matrix_ub, b_ub=b_ub, bounds = (0, None), method='interior-point',
        options={'cholesky':False, 'sym_pos':False, 'lstsq':True, 'presolve':True})
    return (np.clip(res.x[target_deg + 1:],0,1), sparsity, moment_matrix@(res.x[target_deg + 1:]))


# def sde_momentMatching_pgd(target_deg, moment_fn, cheb_mesh): 
# 	cheb_moments, sparsity = moment_fn(target_deg)
# 	moment_matrix = np.zeros((target_deg + 1, len(cheb_mesh)))
# 	for d in range(target_deg + 1): 
# 		a = np.zeros(target_deg + 1)
# 		a[d] = 1 
# 		moment_matrix[d, :] = poly.chebyshev.chebval(cheb_mesh, a)	

# 	pgd_prob = cs.ProjectedGD(lambda x : pgd.func_l2(x, moment_matrix, cheb_moments),
# 		lambda x: pgd.grad_l2(x, moment_matrix, cheb_moments),
# 		lambda x: pgd.projection_simplex_l2(x), ss.ScaledInvIterStepSize())

# 	x0 = np.ones(len(cheb_mesh))/len(cheb_mesh)
# 	res = pgd_prob.solve(x0=x0, max_iter=10000, tol=1e-12, disp=0)
# 	return (res, sparsity, moment_matrix@res)






def solve_fw(f, grad, linsolver, x0, max_iter=1000, tol=1e-5): 
    x = x0
    it = 2
    f_vals = [np.inf, f(x)] 
    x_vals = [x]
    grad_vals = []
    while (f_vals[-2] - f_vals[-1]) > tol:
        if it >= max_iter: 
            print('Max iter reached!')
            break
        g = grad(x)
        x = x + (2/(2 + it))*(linsolver(g) - x)
        f_vals += [f(x)]  
        x_vals += [x]
        grad_vals += [g]
        it += 1
    return x


def sde_momentMatching_pgd(target_deg, moment_fn, cheb_mesh): 
    cheb_moments, sparsity = moment_fn(target_deg)
    scaled_moment_matrix = np.zeros((target_deg + 1, len(cheb_mesh)))
    for d in range(target_deg + 1): 
        a = np.zeros(target_deg + 1)
        a[d] = 1 
        if d != 0: 
            scaled_moment_matrix[d, :] = poly.chebyshev.chebval(cheb_mesh, a)/d
        else: 
            scaled_moment_matrix[d, :] = poly.chebyshev.chebval(cheb_mesh, a)
    
    x0 = np.zeros(len(cheb_mesh))
    x0[-1] = 1.

    scaled_moments = cheb_moments[:]
    scaled_moments[1:] = cheb_moments[1:]/np.arange(1, target_deg+1)

    res = solve_fw(lambda x : pgd.func_l1(x, scaled_moment_matrix, scaled_moments), 
                lambda x: pgd.grad_l1(x, scaled_moment_matrix, scaled_moments), 
                pgd.simplex_FW_linsolver, x0, max_iter=100000)

    return (res, sparsity, scaled_moment_matrix@res)




















    

def approximate_eigs_fromPDF(n, net, pdf_arr): 
    evals = np.zeros(len(net)) 
    y = pdf_arr
    it = np.trapz(y, net)
    y = y/it # normalize the pdf for numerical issues.
    #Compute the definite integral in the intervals defined by net.
    for i in range(len(evals)-1):
        evals[i] = np.trapz([y[i], y[i+1]], dx=net[i+1] - net[i])
    

    curr_eval_index = 0
    eig_vals = np.ones(n)
    next_mass = np.zeros(len(evals))
    next_mass[0] = evals[0]
    for i in range(n): 
        while 1./n - np.sum(next_mass) >= 0.1/n: 
            curr_eval_index+=1 #Keeping track of ub of window in curr_mass
            if curr_eval_index >= len(net): 
                return eig_vals[:i] #Need to trim eigs here.
            next_mass[curr_eval_index] = evals[curr_eval_index]
        prev_mass = np.sum(next_mass[:curr_eval_index])
        if prev_mass > 1e-30:
            prev_avg = np.average(net[:curr_eval_index], weights = next_mass[:curr_eval_index])
        else: prev_avg = 0.

        eig_vals[i] = (n*np.sum(net[:curr_eval_index]*next_mass[:curr_eval_index]) + 
            (1. - n*prev_mass)*net[curr_eval_index])
        next_mass[:curr_eval_index] = 0.0
        next_mass[curr_eval_index] -= (1./n - prev_mass)
    return eig_vals

def compute_approx_eigs_hist(n, num_bins, mesh, sde_rep_coeffs):
    eigs = approximate_eigs(n, mesh, sde_rep_coeffs)
    return np.histogram(eigs, bins=num_bins)




#Assumes val_arr is evenly spaced grid of [-1, 1]. Assumes val_arr is shorter than net
def expand_vals_to_net(val_arr, len_net): 
    new_vals = np.zeros(len_net)
    rel_fact = len_net/len(val_arr)
    for i in range(len(val_arr)): new_vals[np.intc(np.round((i+1)*rel_fact-1))] += val_arr[i]
    return new_vals

# def pdf_fromRep_kpm(kpm_coeffs, mesh): 
# 	return poly.chebyshev.chebweight(mesh)*poly.chebyshev.chebval(mesh, kpm_coeffs)

# def pdf_fromRep_mm(dist_mm, mesh):
# 	return expand_vals_to_net(dist_mm, len(mesh))

def create_grid(grid_size): return np.linspace(-1, 1, num=grid_size, endpoint=True)

def pdf_fromRep_kpm(kpm_coeffs, grid_size): 
    grid = create_grid(grid_size)
    y = np.zeros(len(grid))
    
    c_grid = grid[1:-1]
    y[1:-1] = poly.chebyshev.chebweight(c_grid)*poly.chebyshev.chebval(c_grid, kpm_coeffs)
    
    y[0] = y[1] + (y[1] - y[2])*abs(grid[2] - grid[1])
    y[-1] = y[-2] + (y[-2] - y[-3])*abs(grid[-2] - grid[-3])

    return (grid, y)

def pdf_fromRep_mm(dist_mm, grid_size):
    return (create_grid(grid_size), dist_mm)
    # return expand_vals_to_net(dist_mm, len(mesh))

#Rep is a an array of Chebyshev coefficients of the distribution. Outputs
#cdf evaluated on the mesh. 
def cdf_fromRep_kpm(kpm_coeffs, mesh):
    dist_kpm = evaluate_sde_mesh(mesh, kpm_coeffs)[1]
    return scipy.integrate.cumtrapz(dist_kpm, x=mesh, initial=0.)

def cdf_fromRep_mm(dist_mm, mesh): 
    pdf = expand_vals_to_net(dist_mm, len(mesh))
    return np.cumsum(pdf)






def compute_sde_exact_kpm(mat_obj, args):
    target_deg = args['deg']
    moment_fn_exact = create_exact_moment_fn(mat_obj)
    return sde_rep_kpm(target_deg, moment_fn_exact, 0, 1)

def compute_sde_hutch_kpm(mat_obj, args): 
    target_deg, num_rand_vecs = args['deg'], args['num_random_vecs']
    moment_fn_hutch = create_hutch_moment_fn(mat_obj, num_rand_vecs)
    add = math.sqrt(2/math.pi)*(1/target_deg)
    scaling = (1 + math.sqrt(2*math.pi)*(1/target_deg))	
    return sde_rep_kpm(target_deg, moment_fn_hutch, add, scaling)

def compute_sde_hutchApprox_kpm(mat_obj, args): 
    target_deg, num_rand_vecs, col_budget = args['deg'], args['num_random_vecs'], args['col_budget']
    moment_fn_hutchApprox = create_approxHutch_moment_fn(mat_obj, num_rand_vecs, col_budget)
    add = math.sqrt(2/math.pi)*(1/target_deg)
    scaling = (1 + math.sqrt(2*math.pi)*(1/target_deg))	
    rep, sparsity, moments = sde_rep_kpm(target_deg, moment_fn_hutchApprox, add, scaling)

    mesh_delta = 1e-5
    mesh = np.arange(-1+mesh_delta, 1-mesh_delta, mesh_delta)
    (x, y) = evaluate_sde_mesh(mesh, rep)
    min_val = np.abs(min(np.min(y), 0.))
    area = np.trapz(y, x)

    rep[0] += min_val 
    return (rep/(area + min_val*2), sparsity, moments)


def compute_sde_exact_mm_lp(mat_obj, args): 
    moment_fn_exact = create_exact_moment_fn(mat_obj)
    return sde_momentMatching_lp(args['deg'], moment_fn_exact, args['mesh'])

def compute_sde_hutch_mm_lp(mat_obj, args): 
    moment_fn_hutch = create_hutch_moment_fn(mat_obj, args['num_random_vecs'])
    return sde_momentMatching_lp(args['deg'], moment_fn_hutch, args['mesh'])

def compute_sde_hutchApprox_mm_lp(mat_obj, args): 
    moment_fn_hutchApprox = create_approxHutch_moment_fn(mat_obj, args['num_random_vecs'], 
        args['col_budget'])
    return sde_momentMatching_lp(args['deg'], moment_fn_hutchApprox, args['mesh'])

def compute_sde_exact_mm_pgd(mat_obj, args): 
    moment_fn_exact = create_exact_moment_fn(mat_obj)
    return sde_momentMatching_pgd(args['deg'], moment_fn_exact, args['mesh'])

def compute_sde_hutch_mm_pgd(mat_obj, args): 
    moment_fn_hutch = create_hutch_moment_fn(mat_obj, args['num_random_vecs'])
    return sde_momentMatching_pgd(args['deg'], moment_fn_hutch, args['mesh'])

def compute_sde_hutchApprox_mm_pgd(mat_obj, args): 
    moment_fn_hutchApprox = create_approxHutch_moment_fn(mat_obj, args['num_random_vecs'], 
        args['col_budget'])
    return sde_momentMatching_pgd(args['deg'], moment_fn_hutchApprox, args['mesh'])






class Result:
    def __init__(self, filename, sde_type, args): 
        self.mat_filename = filename
        self.sde_type = sde_type
        self.args = args 
        self.pdf = None 
        self.grid = None
        self.cdf = None 
        self.rep = None
        self.cheb_moments = None
        self.avg_matmul_sp = 0.0
        self.approx_eigs = None
        self.rep_time = None

    def get_result_filename(mat_filename, sde_type, args): 
        degree = "sde_type=%s_degree=%d" % (sde_type, args['deg']) 
        if 'exact' in sde_type: return mat_filename + degree
        num_rand_vecs = "_num_rand_vecs=%d" %(args['num_random_vecs'])
        if 'hutch' in sde_type: return mat_filename + degree + num_rand_vecs
        num_cols = "_num_cols=%d" % (args['col_budget'])
        return mat_filename + degree + num_rand_vecs + num_cols



type_list = ['exact_kpm', 'hutch_kpm', 'approx_kpm', 
            'exact_mm_lp', 'hutch_mm_lp', 'approx_mm_lp',
            'exact_mm_pgd', 'hutch_mm_pgd', 'approx_mm_pgd']	
sde_func_list = [compute_sde_exact_kpm, compute_sde_hutch_kpm, compute_sde_hutchApprox_kpm,
                    compute_sde_exact_mm_lp, compute_sde_hutch_mm_lp, compute_sde_hutchApprox_mm_lp,
                    compute_sde_exact_mm_pgd, compute_sde_hutch_mm_pgd, compute_sde_hutchApprox_mm_pgd]

pdf_func_list = [pdf_fromRep_kpm, pdf_fromRep_kpm, pdf_fromRep_kpm,
                    pdf_fromRep_mm, pdf_fromRep_mm, pdf_fromRep_mm,
                    pdf_fromRep_mm, pdf_fromRep_mm, pdf_fromRep_mm]

cdf_func_list = [cdf_fromRep_kpm, cdf_fromRep_kpm, cdf_fromRep_kpm,
                    cdf_fromRep_mm, cdf_fromRep_mm, cdf_fromRep_mm,
                    cdf_fromRep_mm, cdf_fromRep_mm, cdf_fromRep_mm]

def run_sde_experiment(mat_filename, sde_type, args, mesh_size, num_trials=1):
    print("Running on %s with args="%(mat_filename), args)
    
    result = Result(mat_filename, sde_type, args)
    mat_obj = load_pickle(mat_filename)
    # mesh = np.arange(-1+mesh_delta, 1-mesh_delta, mesh_delta)
    # mesh = create_grid(mesh_size)
    
    type_i = type_list.index(sde_type)
    
    sde_rep_func = sde_func_list[type_i]
    pdf_func = pdf_func_list[type_i]

    s = time.time()
    
    result.rep, result.avg_matmul_sp, result.cheb_moments, result.pdf = [], [], [], []
    for i in range(num_trials): 
        a, b, c = sde_rep_func(mat_obj, args)
        result.rep += [a]
        result.avg_matmul_sp +=  [b]
        result.cheb_moments += [c]
        g, pdf = pdf_func(a, mesh_size)
        result.grid = g
        result.pdf += [pdf]

    t = time.time()	
    print("Computing SDE coefficients %s took %3.2fs" %(sde_type, t - s))
    # print("Average Mat Mult sparsity was %3.4f" % (np.mean(result.avg_matmul_sp)))

    result.rep_time = (t - s)

    write_pickle(result, Result.get_result_filename(mat_filename, sde_type, args))


def compute_approxEigs_W1(main_mat_fn, result_fns, print_result=False):
    eigs = np.sort(np.real(load_pickle(main_mat_fn).compute_eigs()))
    errs = []
    for r in range(len(result_fns)): 
        approx_eigs = np.sort(load_pickle(result_fns[r]).approx_eigs)
        # print (len(approx_eigs))
        eigs_trimmed = eigs[:len(approx_eigs)]
        err = np.sum(np.abs(eigs_trimmed - approx_eigs))/len(eigs)
        errs += [err]
        if print_result: print("W1 error of approximate spectrum from %s is : %3.4f" % 
                        (load_pickle(result_fns[r]).sde_type, err))
    return errs



def compute_W1_fromPDF(main_mat_fn, result_fns, print_result=False):
    eigs = np.sort(np.real(load_pickle(main_mat_fn).compute_eigs()))
    errs = []
    for r in range(len(result_fns)): 
        # approx_eigs = np.sort(load_pickle(result_fns[r]).approx_eigs)
        # # print (len(approx_eigs))
        # eigs_trimmed = eigs[:len(approx_eigs)]
        # err = []
        pdfs = load_pickle(result_fns[r]).pdf
        grid = load_pickle(result_fns[r]).grid
        err = np.zeros(len(pdfs))
        for i in range(len(pdfs)): err[i] = scipy.stats.wasserstein_distance(eigs, grid, np.ones(len(eigs))/len(eigs), 
            pdfs[i])
        # err = [ for i in range(len(pdfs))]
        errs += [err]
        # if print_result: print("W1 error of approximate spectrum from %s is : %3.4f" % 
        # 				(load_pickle(result_fns[r]).sde_type, err))
    return errs








plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=16)
plt.rc('legend', fontsize=8)    # legend fontsize


# colors = ['xkcd:leaf green', 'xkcd:red', 'xkcd:darkblue']
# colors = ['C3', 'C3', 'C4', 'C0', 'C3', 'C3', 'C5']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
# colors = ['C0', 'C3', 'C0', 'C3', 'C3', 'C3']
linestyles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid']
alphas_hist = [0.75, 0.75, .75]
alphas_plots = [0.75, 0.75, 0.75, .75, 0.75, 0.75]


def plot_sdeRep_diffDegs(main_mat_fn, deg_list, mesh_delta, mm_mesh_delta, labels, title=""):
    mm_mesh = np.arange(-0.999, 0.999, mm_mesh_delta) 
    fig = plt.figure(figsize=(6, 5))
    mesh = np.arange(-1 + mesh_delta, 1 - mesh_delta, mesh_delta)

    sde_kind_list = ['exact_mm', 'exact_kpm']
    colors = ['C3', 'C0']
    for i in range(len(sde_kind_list)): 
        result_fns = []
        for d in deg_list: 
            args = {'deg':d, 'mesh':mm_mesh}
            result_fns += [Result.get_result_filename(gfn, sde_kind_list[i], args)]
        errs = compute_W1_fromPDF(main_mat_fn, result_fns, mesh, False)
        plt.plot(deg_list, errs, label = labels[i], alpha = 0.75, color = colors[i], linewidth=1.5)
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.xticks(np.arange(0, np.max(deg_list) + 5, 5))
    # plt.title(title)
    fig.savefig("sdeRep_%s.pdf" %(title), bbox_inches='tight')
    fig.show()

def plot_sdeRep(main_mat_fn, result_fns, labels, mesh_delta, title=""):
    fig = plt.figure(figsize=(6, 5))
    mesh = np.arange(-1 + mesh_delta, 1-mesh_delta, mesh_delta)


    eigs = np.real(load_pickle(main_mat_fn).compute_eigs())
    plt.plot(np.sort(eigs), np.cumsum(np.ones(len(eigs))/len(eigs)), label='True',
        color='xkcd:black', linestyle='solid', linewidth=1.5)
    
    for r in range(len(result_fns)): 
        cdf = load_pickle(result_fns[r]).cdf 
        y = expand_vals_to_net(cdf, len(mesh))
        plt.plot(mesh, y, label=labels[r], alpha=0.75, 
            color=colors[r], linestyle=linestyles[r], linewidth=1.5)

    # plt.title('Spectral Density Estimate using Jackson Damped KPM')
    # plt.yticks(np.arange(0, 1.2*np.max(pdf), 0.5))
    plt.legend(loc='lower right')
    plt.yscale('log')
    fig.savefig("sdeRep_%s.pdf" %(title), bbox_inches='tight')
    fig.show()

def save_sdeRep_csv(main_mat_fn, result_fns, labels, mesh_delta, title=""):
    mesh = np.arange(-0.99, 0.99, mesh_delta)
    y_list = []
    for r in range(len(result_fns)): 
        rep = load_pickle(result_fns[r]).rep
        (x, y) = evaluate_sde_mesh(mesh, rep)
        y_list += [y]

    with open("sdeRep_%s.csv"%(title), 'w', newline='') as file:
        writer = csv.writer(file)
        l = list(zip(*([x] + y_list)))
        writer.writerow(['x'] + labels)
        for p in l: writer.writerow(p)


def plot_approxEigs(main_mat_fn, result_fns, labels, mesh_delta, title=""): 
    fig = plt.figure(figsize=(6, 5))
    num_bins = 12
    delt = 2./(len(result_fns)+2)
    w = 2./(num_bins*(len(result_fns)+2))
    
    for r in range(len(result_fns)): 
        eigs_hist_approx, bins = np.histogram(load_pickle(result_fns[r]).approx_eigs, bins=num_bins, range=(-1.,1.))
        bins = bins + (delt)*r*2*w
    
        plt.bar(bins[:-1], eigs_hist_approx, width=w, label=labels[r], alpha=0.8, color=colors[r])

    eigs = np.real(load_pickle(main_mat_fn).compute_eigs())

    eigs_hist_true, bins = np.histogram(eigs, bins=num_bins, range=(-1.,1.))
    bins_hist = bins + delt*(len(result_fns))*2*w
    
    plt.bar(bins_hist[:-1], eigs_hist_true, label='True Spectrum', width=w, color='xkcd:grey', alpha=1.)
    plt.yticks(np.arange(0, 1.75*np.max(eigs_hist_true), (np.max(eigs_hist_true)//100)*100/4))
    # plt.title('Histogram of Eigenvalues')
    plt.legend(loc='upper right')
    fig.savefig("approxEigs_%s.pdf" %(title), bbox_inches='tight')

def save_approxEigs_csv(main_mat_fn, result_fns, labels, mesh_delta, title=""):
    num_bins = 12
    delt = 2./(len(result_fns)+2)
    w = 2./(num_bins*(len(result_fns)+2))
    
    hists_list = []
    for r in range(len(result_fns)): 
        hists_list += [load_pickle(result_fns[r]).approx_eigs]
        print(len(hists_list[-1]))

    eigs = np.real(load_pickle(main_mat_fn).compute_eigs())	
    hists_list += [eigs]

    with open("approxEigs_%s.csv"%(title), 'w', newline='') as file:
        writer = csv.writer(file)
        l = list(zip(*(hists_list)))
        writer.writerow(labels + ['True'])
        for p in l: writer.writerow(p)
    
def plot_moments(main_mat_fn, result_fns, labels, mesh_delta, title=""): 
    fig, axs = plt.subplots(2)
    fig.set_size_inches(6, 8)

    mesh = np.arange(-0.98, 0.98, mesh_delta)
    num_bins = 25
    delt = 2./(len(result_fns)+1)
    w = 2./(num_bins*(len(result_fns)+1))
    
    for r in range(len(result_fns)): 
        cheb_moments = load_pickle(result_fns[r]).cheb_moments
        rep = load_pickle(result_fns[r]).rep

        moment_bins = np.arange(0, len(rep)) + r/(len(result_fns))
        norms = np.ones(len(rep))*(math.pi/2)
        norms[0] = (math.pi)

        axs[1].bar(moment_bins, cheb_moments*(1/np.sqrt(norms)), width=1/(len(result_fns)), 
                    alpha=alphas_hist[r], label=labels[r], color=colors[r])

        axs[0].bar(moment_bins, rep*(np.sqrt(norms)), width=1/(len(result_fns)), 
                    alpha=alphas_hist[r], label=labels[r], color=colors[r])

    plt.setp(axs[0].get_xticklabels(), visible=False)
    # plt.legend(loc='upper right')
    fig.savefig("moments_%s.pdf" %(title), bbox_inches='tight')


def save_moments_csv(main_mat_fn, result_fns, labels, mesh_delta, title=""):
    moments_list = []
    coefficients_list = []
    for r in range(len(result_fns)): 
        cheb_moments = load_pickle(result_fns[r]).cheb_moments
        rep = load_pickle(result_fns[r]).rep

        moment_bins = np.arange(0, len(rep)) + r/(len(result_fns))
        norms = np.ones(len(rep))*(math.pi/2)
        norms[0] = (math.pi)

        moments_list += [cheb_moments*(1/np.sqrt(norms))]
        coefficients_list += [rep*(np.sqrt(norms))]

    with open("coefficients_%s.csv"%(title), 'w', newline='') as file:
        writer = csv.writer(file)
        l = list(zip(*([np.arange(0, len(rep))] + coefficients_list)))
        writer.writerow(['degree'] + labels)
        for p in l: writer.writerow(p)

    with open("moments_%s.csv"%(title), 'w', newline='') as file:
        writer = csv.writer(file)
        l = list(zip(*([np.arange(0, len(rep))] + moments_list)))
        writer.writerow(['degree'] + labels)
        for p in l: writer.writerow(p)

# X = MatrixObjectCSCSparse("ca-GrQc/ca-GrQc_graphRep_normAdjMat")
# experiment_fixedDegVecsBudget(X, 0.01, 20, 20, 5000)

# n = 5000 #Dimension of matrix. 

# A = np.random.binomial(1, 0.75, size=(n, n)) + np.random.binomial(1, 0.25, size=(n, n))
# M = np.matmul(A - np.ones((n, n)), (A - np.ones((n, n))).T )
# X = MatrixObjectDense.normalize_mat(M, np.linalg.norm(M, 2))
# write_pickle(X, "test/test_matrix_binomial_5000")
# write_pickle(np.linalg.eigvals(X), "test/test_matrix_binomial_5000_eigs")

# num_communities = 2 #number of communites
# comm_size = 5000 #Size of each communiity
# comm_p = 0.5 #Probabilityof edge within community
# noncomm_p = 0.01 #Probability of edge outside community
# gfn = sbm.return_sbmgraph_object_filename(num_communities, comm_size, comm_p, noncomm_p)









# num = 1000
# gfn = sbm.return_sbmgraph_object_filename(num, label='cliquePlusMatching')
#40, 2, 85000 



# num = 1000
# gfn = sbm.return_sbmgraph_object_filename(num, label='cliquePlusStar')
#40, 2, 450 

# num = pow(2, 14)
# gfn = sbm.return_sbmgraph_object_filename(num, label='hypercube_graph')
# 40, 2, 7500
#80, 2, 11000, 0.025




# mat_obj = MatrixObjectCSCSparse("ca-GrQc/ca-GrQc_graphRep_normAdjMat")
# graph_list = ['cliquePlusMatching', 'cliquePlusStar', 'hypercube_graph']
# graph_index = 0


# curr_graph = graph_list[graph_index]
# num = [1000, 1000, pow(2, 14)][graph_index]
# gfn = sbm.return_sbmgraph_object_filename(num, label=curr_graph)

# sde_degree = 28 #Degree of chebyshev polynomial for sde
# num_rand_vecs = 2  #Number of hutchinson's vecs
# cols_oversample_factor = 8000 #Scaling factor for col probabilities  
# mesh_delta = 1e-5 #No need to change 
# mm_mesh_delta = 1e-5
# mm_mesh = np.arange(-1, 1+mm_mesh_delta, mm_mesh_delta)

# exact_args = {'deg' : sde_degree, 'mesh':mm_mesh}
# hutch_args = {'deg':sde_degree, 'num_random_vecs':num_rand_vecs,'mesh':mm_mesh}
# approx_args = {'deg':sde_degree, 'num_random_vecs':num_rand_vecs,'mesh':mm_mesh,
# 			'col_budget':cols_oversample_factor}


# run_sde_experiment(gfn, 'exact_kpm', exact_args, mesh_delta)
# run_sde_experiment(gfn, 'hutch_kpm', hutch_args, mesh_delta)
# run_sde_experiment(gfn, 'approx_kpm', approx_args, mesh_delta)
# run_sde_experiment(gfn, 'exact_mm', exact_args, mesh_delta)
# run_sde_experiment(gfn, 'hutch_mm', hutch_args, mesh_delta)
# run_sde_experiment(gfn, 'approx_mm', approx_args, mesh_delta)


# result_fns = [Result.get_result_filename(gfn, 'exact_mm', exact_args),
# Result.get_result_filename(gfn, 'hutch_mm', hutch_args),
# Result.get_result_filename(gfn, 'approx_mm', approx_args),
# Result.get_result_filename(gfn, 'exact_kpm', exact_args),
# Result.get_result_filename(gfn, 'hutch_kpm', hutch_args),
# Result.get_result_filename(gfn, 'approx_kpm', approx_args)]

# labels = ["MM", "KPM"]
# # labels = ['Idealized MM', 'Hutchinson MM', 'Approximate Hutch MM',
# # 'Idealized KPM', 'Hutchinson KPM', 'Approximate Hutch KPM']

# deg_list = np.arange(4, 64, 4, dtype=np.intc)

# for d in deg_list: 
# 	exact_args = {'deg' : d, 'mesh':mm_mesh}
# 	# run_sde_experiment(gfn, 'exact_kpm', exact_args, mesh_delta)
# 	run_sde_experiment(gfn, 'exact_mm', exact_args, mesh_delta)



# result_fns = []
# deg_list = [4,8,12,16,20,40]
# labels = ['4', '8', '12', '16', '20', '40']
# mm_mesh = np.arange(-1, 1+mm_mesh_delta, mm_mesh_delta)

# run_sde_experiment(gfn, 'exact_mm', {'deg':100, 'mesh':mm_mesh}, mesh_delta)	

# num_bins = 20
# delt = 2./(len(deg_list)+2)
# w = 2./(num_bins*(len(deg_list)+2))

# for i in range(len(deg_list)): 
# 	d = deg_list[i]
# 	exact_args = {'deg' : d, 'mesh':mm_mesh}	
# 	# run_sde_experiment(gfn, 'exact_mm', exact_args, mesh_delta)	
    # result_fns += [Result.get_result_filename(gfn, 'exact_mm', exact_args)]
# 	res = load_pickle(Result.get_result_filename(gfn, 'exact_mm', exact_args))
# 	filt = 1000*res.rep >= 1 
# 	plt.bar(mm_mesh[filt]+(delt)*i*2*w, 1000*res.rep[filt], label=labels[i], alpha=1., width=w)

# eigs = np.real(load_pickle(gfn).compute_eigs())
# eigs_hist_true, bins = np.histogram(eigs, bins=num_bins, range=(-1.,1.))
# bins_hist = bins + delt*(len(result_fns))*2*w

# plt.bar(bins_hist[:-1], eigs_hist_true, label='True Spectrum', width=w, color='xkcd:grey', alpha=1.)
# plt.legend()

# compute_W1_fromPDF(gfn, result_fns, np.arange(-1+mesh_delta, 1 - mesh_delta, mesh_delta), print_result=True)
# plot_sdeRep(gfn, result_fns, labels, mesh_delta, curr_graph)	

# plot_sdeRep_diffDegs(gfn, deg_list, mesh_delta, mm_mesh_delta, ['Moment Matching', 'KPM'], curr_graph)

# plot_sdeRep(gfn, result_fns, labels, mesh_delta, curr_graph)
# plot_approxEigs_sdeRep_moments(gfn, result_fns, labels, mesh_delta, curr_graph)
# # save_sdeRep_csv(gfn, result_fns, labels, mesh_delta, curr_graph)
# plot_approxEigs(gfn, result_fns, labels, mesh_delta, curr_graph)
# # save_approxEigs_csv(gfn, result_fns, labels, mesh_delta, curr_graph)
# plot_moments(gfn, result_fns, labels, mesh_delta, curr_graph)
# # save_moments_csv(gfn, result_fns, labels, mesh_delta, curr_graph)

# compute_approxEigs_W1(gfn, result_fns, True)



# plt.show()