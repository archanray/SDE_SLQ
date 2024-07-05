import numpy as np
from src.lanczos import naive_lanczos
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from src.moment_estimator import approxChebMomentMatching, discretizedJacksonDampedKPM, hutchMomentEstimator, SLQMM, VRSLQMM, adder, bkde
from src.utils import Wasserstein, jacksonDampingCoefficients, jackson_poly_coeffs
from src.distribution import Distribution, mergeDistributions
from src.optimizers import cvxpyL1Solver
from src.optimizers import pgdSolver
from src.get_dataset import get_data
import scipy as sp
import numpy.polynomial as poly
from src.utils import normalizedChebyPolyFixedPoint
import time
from src.optimizers import pgdSolver
import numpy.polynomial as poly
import sys
import seaborn as sns
import pickle
from src.block_krylov import bki

def findMaxIndex(L1):
    if np.abs(L1[-1]) > np.abs(L1[0]):
        return -1
    else:
        return 0

def sortValues(eigvals, eigvecs):
    sorted_indices = np.argsort(eigvals)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    return eigvals, eigvecs

def padZeros(values, n):
    zeroarray = np.zeros(n-len(values))
    values = np.concatenate((values, zeroarray))
    values = np.sort(values)
    return values

def sortTwoArrays(A, B):
    sorted_indices = np.argsort(A)
    A = A[sorted_indices]
    B = B[sorted_indices]
    return A, B
    

class TestCalculations:
    def checkDistros(self):
        D = Distribution()
        for i in range(5):
            print(D.support)
            q = np.random.randn()
            localD = Distribution([i], [q])
            print(q, localD.support)
            D = mergeDistributions(D, localD, func=adder(5))
        print(D.support)
        D.finalizeWeights()
        print(D.support)
        return None
    
    def checkCalculation(self):
        A = np.random.randn(10,10)
        A = (A+A.T) / 2
        L, V = np.linalg.eig(A)
        VVT = np.dot(V, V.T)
        support = L
        weights1 = np.sum(VVT, axis=1)
        weights2 = np.zeros_like(support)
        for i in range(len(L)):
            weights2[i] = np.outer(V[:,i], V[:,i])[0,0]
        print("weights1:", weights1, "\nweights2:", weights2)
        print("sum of weights2:", np.sum(weights2))
        print("VVT:\n", VVT)
        return None
    
    def checkProjection(self):
        solver = pgdSolver()
        x = np.random.randn(5)
        xd = solver.proj_simplex_array(x)
        print("sum xd", np.sum(xd), "\nxd:", xd)
        return None
    
    def checkChebyshevMatrices(self):
        N = 50
        nIntegers = np.array(list(range(1,N+1)))
        d = 1000
        xs = np.linspace(-1,1,num=d+1,endpoint=True)
        Tkbar = np.zeros((N, d+1))
        # TNd = np.vstack([np.pi*np.ones(d+1), Tkbar])
        
        st1 = time.time()
        for j in range(100):
            for i in range(d+1):
                Tkbar[:, i] = normalizedChebyPolyFixedPoint(xs[i], N)
            TNd = np.divide(Tkbar, nIntegers.reshape(-1,1))
        et1 = time.time()
        # print("TNd:\n",TNd)
        
        scaled_moment_matrix = np.zeros((N, len(xs)))
        st2 = time.time()
        for i in range(100):
            for d in range(1,N + 1): 
                a = np.zeros(N + 1)
                a[d] = 1 
                scaled_moment_matrix[d-1, :] = 2*poly.chebyshev.chebval(xs, a)/(max(1,d)*np.pi)
        et2 = time.time()
        
        # print("scaled_moment_matrix:\n",scaled_moment_matrix)
        
        if np.linalg.norm(TNd - scaled_moment_matrix) <= 1e-14:
            print("True")
            print("Total time for method1:", et1-st1, "\nTotal time for method2:", et2-st2)
        else:
            print("False")
        return None    
        
    def checkWasserstein(self):
        v1 = [0, 1, 3]
        p1 = [0.5, 0.4, 0.1]
        v2 = [5, 9, 2]
        p2 = [0.3, 0.3, 0.4]
        D1 = Distribution(v1, p1)
        D2 = Distribution(v2, p2)
        D1.showDistribution()
        D2.showDistribution()
        print("Wasserstein between the two distributions (shouldn't be 0):", Wasserstein(D1, D2))
        print("Wasserstein between the two distributions (should be 0) :", Wasserstein(D1, D1))
        return None
    
    def checkL1Optimizer(self):
        N = 50
        d = 5000 #int(N**3/ 2)
        T = np.random.randn(N, d)
        z = np.random.rand(N)
        z = z/np.sum(z)
        
        solver1 = cvxpyL1Solver(T, z)
        solver1.minimizer()
        print("smallest value achieved using optimize.linprog:", solver1.res.fun, np.linalg.norm(T@solver1.res.x - z,1))
        print("sum q values:", np.sum(solver1.res.x))
        
        solver2 = pgdSolver(T, z)
        solver2.minimizer(plot=True)
        print("smallest value achieved using optimize.minimize:", solver2.res.fun)
        print("sum q values:", np.sum(solver2.res.x))
        
        return None
    
    def plot_vals(self, x=None, v1=None, v2=None, v1_lo=None, v1_hi=None, v2_lo=None, v2_hi=None,\
                  label1=None, label2=None, xlabel=None, ylabel=None, filename="plot", ext=".pdf"):
        plt.rcParams.update({'font.size': 13})
        if v1 is None:
            pass
        else:
            plt.plot(x, v1, label=label1)
            plt.fill_between(x, v1_lo, v1_hi, alpha=0.2)
        if v2 is None:
            pass
        else:
            plt.plot(x, v2, label=label2)
            plt.fill_between(x, v2_lo, v2_hi, alpha=0.2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale("log")
        plt.grid(linewidth=0.3)
        plt.legend()
        savedir = os.path.join("figures", "unittests")
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        savefilepath = os.path.join(savedir, filename+ext)
        plt.savefig(savefilepath, bbox_inches='tight',dpi=200)
        return None
        
    def testMomentMatchings(self):
        Ns = np.array(list(range(20,88,8)))
        trials = 4
        errors1 = np.zeros((trials, len(Ns)))
        errors2 = np.zeros((trials, len(Ns)))
        
        for i in range(trials):
            for j in tqdm(range(len(Ns))):
                A = np.random.randn(200, 200)
                A = (A+A.T) / 2
                A = A / np.linalg.norm(A, 2)
                support_fx = np.real(np.linalg.eigvals(A))
                fx = np.ones_like(support_fx) / len(support_fx)

                tau = hutchMomentEstimator(A, Ns[j], 20)
                
                support_q, q = approxChebMomentMatching(tau)
                D1, D2 = Distribution(support_fx, fx), Distribution(support_q, q)
                errors1[i,j] = Wasserstein(D1, D2)
                
                support_q, q = discretizedJacksonDampedKPM(tau)
                D1, D2 = Distribution(support_fx, fx), Distribution(support_q, q)
                errors2[i,j] = Wasserstein(D1, D2)
                
        meanErrors1 = np.mean(errors1, axis=0)
        pc20Errors1 = np.percentile(errors1, q=20, axis=0)
        pc80Errors1 = np.percentile(errors1, q=80, axis=0)
        meanErrors2 = np.mean(errors2, axis=0)
        pc20Errors2 = np.percentile(errors2, q=20, axis=0)
        pc80Errors2 = np.percentile(errors2, q=80, axis=0)
        
        self.plot_vals(x=Ns,
                       v1=meanErrors1, 
                       v1_lo=pc20Errors1, 
                       v1_hi=pc80Errors1, 
                       xlabel="Ns", 
                       ylabel="W1 distance",
                       label1="ChebMM",
                       v2=meanErrors2, 
                       v2_lo=pc20Errors2, 
                       v2_hi=pc80Errors2, 
                       label2="KPM",
                       filename="compare_MM")
    
    def visualizeDistributions(self):
        N = 20
        tau = np.random.rand(N)
        tau = tau / np.sum(tau)
        support_tau = np.array(list(range(N)))
        support_tau = -1 + (1-(-1)) * (support_tau - support_tau[0]) / (support_tau[-1] - support_tau[0])
        support_q, q = approxChebMomentMatching(tau)
        
        D1, D2 = Distribution(support_tau, tau), Distribution(support_q, q)
        print("Wasserstein error:", Wasserstein(D1, D2))
        
        plt.plot(support_tau, tau, label="input distribution")
        plt.plot(support_q, q, label="output distribution")
        plt.legend()
        plt.title("comparing distributions")
        savedir = os.path.join("figures", "unittests")
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        savefilepath = os.path.join(savedir, "chebMM_distribution_visualize.pdf")
        plt.savefig(savefilepath, bbox_inches='tight',dpi=200)
        return None
    
    def checkOutputs(self):
        N = 16
        tau = np.random.rand(N)
        tau = tau / np.sum(tau)
        supports, probs = discretizedJacksonDampedKPM(tau)
        return None
    
    def momentMatchingSingleton(self):
        A = np.random.randn(200, 200)
        A = (A+A.T) / 2
        A = A / np.linalg.norm(A, 2)
        support_fx = np.real(np.linalg.eigvals(A))
        fx = np.ones_like(support_fx) / len(support_fx)
        tau = hutchMomentEstimator(A, 20, 100)
        supportq1, q1 = approxChebMomentMatching(tau, method="cvxpy")
        supportq2, q2 = approxChebMomentMatching(tau, method="pgd") #discretizedJacksonDampedKPM(tau)
        supportq3, q3 = approxChebMomentMatching(tau, method="optimize")
        print("TAU:", tau, "\nQ1:", q1, "\nQ2:", q2, "\nQ3:", q3)
        print("sum of vals:", sum(q1), sum(q2), sum(q3))
        D_baseline = Distribution(support_fx, fx)
        D_CMM_CVXPY = Distribution(supportq1, q1)
        D_CMM_PGD = Distribution(supportq2, q2)
        D_CMM_OPT = Distribution(supportq3, q3)
        print("W1 distance with CMM and CVXPY:", Wasserstein(D_baseline, D_CMM_CVXPY))
        print("W1 distance with CMM and PGD:", Wasserstein(D_baseline, D_CMM_PGD))
        print("W1 distance with CMM and Numpy OPTIMIZE:", Wasserstein(D_baseline, D_CMM_OPT))
        return None
    
    def checkHutch(self):
        dataset = "gaussian"
        data, n = get_data(dataset)
        moments = 12
        rand_vecs = 2*np.random.binomial(1, 0.5, size=(n, 1000)) - 1
        
        tau_here = hutchMomentEstimator(data, moments, G=rand_vecs, l=1000)
        tau_baseline = hutchMomentEstimator(data, moments, rand_vecs, l=1000)
        print(tau_here, "\n", tau_baseline)
        return None
    
    def sdeComputer(self, data, degree, method = "CMM", cheb_vals=None, submethod="cvxpy", eigvals=None, random_restarts=1):
        if method == "CMM":
            tau = hutchMomentEstimator(data, degree, random_restarts)
            supports, q = approxChebMomentMatching(tau, method=submethod, cheb_vals=cheb_vals)
            # print(supports, q)
            return supports, q
        if method == "KPM":
            tau = hutchMomentEstimator(data, degree, random_restarts)
            supports, q = discretizedJacksonDampedKPM(tau)
            return supports, q
        if method == "SLQMM":
            return SLQMM(data, degree, random_restarts)
        if method == "VRSLQMM-c12" or method == "VRSLQMM":
            return VRSLQMM(data, degree, random_restarts, constraints="12")
        if method == "VRSLQMM-c1":
            return VRSLQMM(data, degree, random_restarts, constraints="1")
        if method == "VRSLQMM-c2":
            return VRSLQMM(data, degree, random_restarts, constraints="2")
        if method == "BKSDE-CMM":
            return bkde(data, degree, random_restarts, MM="cheb", cheb_vals=cheb_vals)
        if method == "BKSDE-KPM":
            return bkde(data, degree, random_restarts, MM="KPM")
        return None
    
    def checkSDEApproxError(self, data, moments, support_true, method="CMM", cheb_vals=1000, trials=5, submethod="cvxpy", random_restarts=1):
        quantile_lo = 10
        quantile_hi = 90
        errors = np.zeros((trials,len(moments)))
        pdf_true = np.ones_like(support_true) / len(support_true)
        eigvals = np.real(np.linalg.eigvals(data))
        
        # colors = sns.color_palette('hls', len(moments))
        
        for t in tqdm(range(trials)):
            # fig, axs = plt.subplots(1,2)
            # fig.set_size_inches(16, 10)
            # # axs[0].set_prop_cycle('color', colors)
            # axs[0].scatter(support_true, pdf_true, label="true")
            
            for j in range(len(moments)):
                support_current, pdf_current = self.sdeComputer(data, moments[j], method = method, cheb_vals = cheb_vals, submethod=submethod, eigvals=eigvals, random_restarts=random_restarts)
                # if j == len(moments)-1:
                    # axs[0].scatter(support_current, pdf_current, label=str(moments[j]))
                errors[t,j] = sp.stats.wasserstein_distance(support_true, support_current, pdf_true, pdf_current)
                # errors[t,j] = sp.stats.wasserstein_distance(support_true, support_current)
            pass
            # axs[1].plot(moments, errors[t,:])
            # axs[0].legend()
            # plt.savefig("figures/unittests/"+method+"_pdf_and_errors_at_moment_only_support.pdf", bbox_inches='tight', dpi=200)
        
        errors_mean = np.mean(errors, axis=0)
        errors_lo = np.percentile(errors, q=quantile_lo, axis=0)
        errors_hi = np.percentile(errors, q=quantile_hi, axis=0)
        
        return errors_mean, errors_lo, errors_hi
    
    def runSDEexperiments(self, random_restarts=5, dataset_names = "all", methods = ["all"], loadresults = [True, True, True, True, True, True]):
        # colors chosen from https://matplotlib.org/stable/gallery/color/named_colors.html
        colors = ["red", "dodgerblue", "black", "darkorchid", "#D2691E", "#40E0D0"]
        if dataset_names == "all":
            ds = ["gaussian", "uniform", "erdos992", "small_large_diagonal", "low_rank_matrix", "power_law_spectrum", "inverse_spectrum", "square_inverse_spectrum"] # "hypercube"
        else:
            ds = [dataset_names]
        if methods[-1] == "all":
            methods = ["SLQMM", "CMM", "KPM", "VRSLQMM-c1", "VRSLQMM-c2", "VRSLQMM-c12", "BKSDE-CMM", "BKSDE-KPM"]
        else:
            pass
        if len(loadresults) != len(methods):
            print("loadresults should be of same size")
            sys.exit(1)        
        for dataset in ds:
            print("running for dataset:", dataset)
            print("random restarts:", random_restarts)
            # dataset = "hypercube"
            data, n = get_data(dataset)
            eigs_folder = "outputs/"+dataset+"/"
            if not os.path.isdir(eigs_folder):
                os.makedirs(eigs_folder)
            eigs_file = eigs_folder+"true_eigvals.npy"
            if os.path.isfile(eigs_file):
                support_true = np.load(eigs_file)
            else:
                support_true = np.real(np.linalg.eigvals(data))
                np.save(eigs_file, support_true)
            # set up moments
            moments = np.arange(8,120,8, dtype=int)
            
            foldername = "outputs/"+dataset+"/"+str(random_restarts)
            if not os.path.isdir(foldername):
                os.makedirs(foldername)
            
            for i in range(len(methods)):
                print(methods[i])
                # set up file name
                filename = foldername+"/"+methods[i]+".pkl"
                # check if file with results exist, if yes load, else run code
                if os.path.isfile(filename) and loadresults[i] == True:
                    file_ = open(filename, "rb")
                    errors_mean, errors_lo, errors_hi = pickle.load(file_)
                    file_.close()
                else:
                    errors_mean, errors_lo, errors_hi = self.checkSDEApproxError(data, moments, support_true, method=methods[i], cheb_vals=5000, random_restarts=random_restarts)
                    # save results to filename
                    file_ = open(filename, "wb")
                    pickle.dump([errors_mean, errors_lo, errors_hi], file_)
                    file_.close()
                    
                plt.plot(random_restarts*moments, errors_mean, label=methods[i], color=colors[i])
                plt.fill_between(random_restarts*moments, errors_lo, errors_hi, alpha=0.2, color=colors[i])
                
            plt.legend()
            plt.ylabel("Wasserstein error")
            plt.yscale("log")
            plt.xlabel("Total matric-vector queries")
            plt.yticks([10**0, 10**(-1), 10**(-2), 10**(-3)])
            plt.grid()
            if not os.path.isdir("figures/unittests/SDE_approximation_errors/"+str(random_restarts)):
                os.makedirs("figures/unittests/SDE_approximation_errors/"+str(random_restarts))
            plt.savefig("figures/unittests/SDE_approximation_errors/"+str(random_restarts)+"/"+dataset+".pdf", bbox_inches='tight', dpi=200)
            plt.clf()
            plt.close()
    
    def checkChebValNums(self):
        values = [1000]#np.arange(500,3000,500)
        dataset = "low_rank_matrix"
        data, n = get_data(dataset)
        support_true = np.real(np.linalg.eigvals(data))
        moments = list(range(4,104,4))
        
        for i in range(len(values)):
            errors_mean, errors_lo, errors_hi = self.checkSDEApproxError(data, moments, support_true, method="CMM", cheb_vals=values[i])
            
            plt.plot(moments, errors_mean, label=str(values[i]))
            plt.fill_between(moments, errors_lo, errors_hi, alpha=0.2)
        
        plt.legend()
        plt.savefig("figures/unittests/CMM_valriations_with_d_"+dataset+".pdf", bbox_inches='tight', dpi=200)
               
    def checkJacksonPolynomial(self):
        deg = 8
        print("J-Polys Here:", jacksonDampingCoefficients(deg))
        print("J-Polys BLin:", jackson_poly_coeffs(deg))
        return None
    
    def checkChebyshevOptimizers(self):
        dataset = "gaussian"
        data, n = get_data(dataset)
        support_true = np.real(np.linalg.eigvals(data))
        moments = list(range(4,60,4))
        algo = ["cvxpy", "pgd"]
        
        for i in range(len(algo)):
            errors_mean, errors_lo, errors_hi = self.checkSDEApproxError(data, moments, support_true, method="CMM", trials=3, cheb_vals=500, submethod=algo[i])
            
            plt.plot(moments, errors_mean, label=str(algo[i]))
            plt.fill_between(moments, errors_lo, errors_hi, alpha=0.2)
        
        plt.legend()
        plt.savefig("figures/unittests/CMM_variations_with_optimizer_"+dataset+".pdf", bbox_inches='tight', dpi=200)
            
        return None
    
    def checkKrlovCorrectness(self):
        dataset = "gaussian"
        data, n = get_data(dataset)
        print(np.linalg.norm(data, ord=2))
        k = 15
        block_size = np.arange(8,60,4) // 4
        trials = 5
        check_ranks = [0,1,2,3]
        errors = np.zeros((trials, len(block_size)))
        for check_rank in check_ranks:
            # print(check_rank)
            for t in tqdm(range(trials)):
                for i in range(len(block_size)):
                    Q = bki(data, block_size[i], k)
                    # print(k, Q.shape)
                    T = Q.T @ data @ Q
                    Lambdas, Vectors = np.linalg.eig(T)
                    Qv = Q @ Vectors[:,check_rank]
                    errors[t, i] = np.linalg.norm(data @ Qv - Lambdas[check_rank]*Qv)
                    # print(errors[t,i])
                    pass
            plt.plot(block_size, np.mean(errors, axis=0))
            plt.fill_between(block_size, np.percentile(errors, axis=0, q=20), np.percentile(errors, axis=0, q=80), alpha=0.2)
            plt.xlabel("Krylov block size")
            plt.ylabel(r"$\|\mathbf{A} \mathbf{Q} \mathbf{v}_j-  \lambda_j(\mathbf{T}) \mathbf{Q}\mathbf{v}_j\|_2$")
            plt.yscale("log")
            plt.title(r"$j = $"+ str(check_rank) + ", " + r"$\lambda_j(\mathbf{T}) = $"+str(Lambdas[check_rank]))
            plt.savefig("figures/unittests/block_krylov/test_error_"+str(check_rank)+".pdf", bbox_inches='tight', dpi=200)
            plt.clf()
            plt.close()
        
        Q = bki(data, block_size[i], k)
        T = Q.T @ data @ Q
        Lambdas, Vectors = np.linalg.eig(T)
        true_Lambdas, _ = np.linalg.eig(data)
        plt.plot(range(len(Lambdas)), np.sort(Lambdas), label="krylov")
        plt.plot(range(len(true_Lambdas)), np.sort(true_Lambdas), label="true")
        plt.xlabel("eigenvalue ID")
        plt.ylabel("eigenvalues")
        plt.legend()
        plt.savefig("figures/unittests/block_krylov/approx_eigvals_gaussian.pdf", bbox_inches='tight', dpi=200)
        return None
    
    def checkKrylovLInf(self):
        dataset = "gaussian"
        data, n = get_data(dataset)
        true_supports = np.linalg.eigvals(data)
        true_supports = np.sort(true_supports)
        true_weights = np.ones_like(true_supports) / len(true_supports)
        k = 15
        block_size = np.arange(16,120,4) // 2
        trials = 5
        ell_infty_error = np.zeros((5, len(block_size)))
        Wasserstein = np.zeros((5, len(block_size)))
        for t in range(trials):
            for i in range(len(block_size)):
                block = block_size[i]
                Q = bki(data, k=block, q=k, QR=True)
                Lambdas = np.linalg.eigvals(Q.T @ data @ Q)
                # Lambdas = np.linalg.eigvals((data @ Q @ Q.T + Q @ Q.T @ data)/2)
                # print(n, len(Lambdas))
                Lambdas = np.pad(Lambdas, (0, n-len(Lambdas)), 'constant')
                Lambdas = np.sort(Lambdas)
                ell_infty_error[t,i] = np.max(np.abs(true_supports - Lambdas))
                Wasserstein[t,i] = sp.stats.wasserstein_distance(true_supports, Lambdas, true_weights, np.ones_like(Lambdas) / len(Lambdas))
        
        plt.plot(block_size, np.mean(ell_infty_error, axis=0), label="ell_infty")
        plt.plot(block_size, np.mean(Wasserstein, axis=0), label="wasserstein")
        plt.xlabel("block size")
        plt.ylabel("errors")
        plt.yscale("log")
        plt.yticks([10**0, 10**(-1), 10**(-2), 10**(-3)])
        plt.legend()
        plt.savefig("figures/unittests/block_krylov/ell_infty_varied_block.pdf", bbox_inches="tight", dpi=200)
        
        return None
    
    def check_deflation(self):
        dataset = "gaussian"
        data, n = get_data(dataset)
        
        E, L = np.linalg.eig(data)
        print(E)
        
        Z = L[:,0:2]
        Ed = E[2:]
        
        true_supports = Ed
        true_weights = np.ones_like(Ed) / len(Ed)
        
        P = np.eye(n) - Z@Z.T
        deflated_matrix = P @ data @ P.T
        
        ell = 15
        N_hutch = 4
        tau = hutchMomentEstimator(deflated_matrix, N_hutch, ell, G=None)
        tau = (1 / (n-2)) * (n*tau - 2 * normalizedChebyPolyFixedPoint(0, len(tau)))
        
        supports, weights = approxChebMomentMatching(tau, cheb_vals=15000)
        mask = (np.abs(supports) < 1).astype(int)
        q2_supports = supports * mask
        q2_weights = weights * mask
        
        plt.plot(true_supports, true_weights, color="blue", label="true")
        plt.plot(q2_supports, q2_weights, color="red", label="CMM-corrected")
        plt.xlabel("supports")
        plt.ylabel("weights")
        plt.ylim([-0.01, 0.1])
        plt.legend()
        error = sp.stats.wasserstein_distance(true_supports, q2_supports, true_weights, q2_weights)
        plt.title("Wasserstein error:"+ str(error))
        plt.savefig("figures/unittests/checking_CMM_deflated.pdf", bbox_inches="tight", dpi=200)
        return None
        

if __name__ == '__main__':
    # mults = [25] #[5,10,15,20,25]
    # dataset_names = "uniform" # "all"
    # methods = ["SLQMM", "CMM", "KPM", "VRSLQMM-c12", "BKSDE-CMM", "BKSDE-KPM"]# ["SLQMM", "CMM", "KPM", "VRSLQMM-c1", "VRSLQMM-c2", "VRSLQMM-c12"]
    # loadresults = [True, True, False, True, True, True]
    # for i in mults:
    #     TestCalculations().runSDEexperiments(i, dataset_names, methods, loadresults)
    TestCalculations().checkKrylovLInf()