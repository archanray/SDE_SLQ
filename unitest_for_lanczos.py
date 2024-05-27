import numpy as np
from src.lanczos import naive_lanczos, CTU_lanczos
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from src.moment_estimator import approxChebMomentMatching, discretizedJacksonDampedKPM, hutchMomentEstimator, baselineHutch, baselineKPM, baselineCMM, exactCMM, SLQMM, adder, SLQNew
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

def padZeros(eigvals, n):
    # pads zeros and re-sorts the eigenvalues to avoid issues of matching incorrectly to the original spectrum
    eigvals = np.pad(eigvals, (0, n-len(eigvals)), mode='constant', constant_values=0)
    return np.sort(eigvals)

def functionNameMapper(method):
    if method == "naive":
        return naive_lanczos
    elif method == "CTU":
        return CTU_lanczos
    else:
        raise ValueError("Invalid method name")
    return None

def plot_vals(x=None, v1=None, v2=None, v1_lo=None, v1_hi=None, v2_lo=None, v2_hi=None,\
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

class testWrapper():
    def checkLanczosConvergence(self, method="naive", reorthogonalizeFlag=False):
        # this has been tested to work
        trials = 5
        n = 250
        iterations = np.arange(10,250,10)
        error1 = np.zeros((trials, len(iterations)))
        error2 = np.zeros_like(error1)
        func = functionNameMapper(method)
        
        for t in tqdm(range(trials)):
            # in each trial init the matrix first
            data = np.random.randn(n, n)
            data = (data+data.T) / 2
            data /= np.linalg.norm(data, ord=2)
            true_lambda, true_vecs = np.linalg.eig(data)
            true_lambda, true_vecs = sortValues(true_lambda, true_vecs)
            # plt.plot(range(len(true_lambda)), true_lambda)
            index = findMaxIndex(true_lambda)
            # index = 3
            rand_vec = np.random.randn(n)
            rand_vec /= np.linalg.norm(rand_vec)
            
            for q in range(len(iterations)):
                Q, T = func(data, rand_vec, iterations[q], return_type="QT", reorth=reorthogonalizeFlag)
                
                local_lambda, local_vecs = np.linalg.eig(T)
                local_lambda, local_vecs = sortValues(local_lambda, local_vecs)
                local_lambda = padZeros(local_lambda, n) # padding after sorting needs another sort for just the eigenvalues, this is handled by the padZeros function now
                error1[t, q] = np.abs(true_lambda[index] - local_lambda[index])
                
                Z_i = np.dot(Q, local_vecs[:,index].T)
                L_iZ_i = local_lambda[index] * Z_i
                AZ_i = np.dot(data, Z_i.T)
                error2[t, q] = np.linalg.norm(L_iZ_i - AZ_i)
            
            # plt.plot(iterations,error1[t,:])
            # plt.show()
            # plt.close()
            # plt.clf()
        mean_error1 = np.mean(error1, axis=0)
        p20_error1 = np.percentile(error1, axis=0, q=20)
        p80_error1 = np.percentile(error1, axis=0, q=80)
        mean_error2 = np.mean(error2, axis=0)
        p20_error2 = np.percentile(error2, axis=0, q=20)
        p80_error2 = np.percentile(error2, axis=0, q=80)
        
        plt.plot(iterations, mean_error1)
        plt.fill_between(iterations, p20_error1, p80_error1, alpha=0.3)
        plt.yscale("log")
        plt.grid(linewidth=1)
        plt.xlabel("iterations")
        plt.ylabel(r"$|\lambda_1(\mathbf{A}) - \lambda_1(\mathbf{T})|$")
        plt.savefig("figures/unittests/lanczos/ErrorAbs_"+method+"_"+str(reorthogonalizeFlag)+".pdf", bbox_inches='tight',dpi=200)
        plt.close()
        
        plt.plot(iterations, mean_error2)
        plt.fill_between(iterations, p20_error2, p80_error2, alpha=0.3)
        plt.yscale("log")
        plt.grid(linewidth=1)
        plt.xlabel("iterations")
        plt.ylabel(r"$||\lambda_1(\mathbf{A})\mathbf{QZ}_1 - \mathbf{AQZ}_1||_2$")
        plt.savefig("figures/unittests/lanczos/ErrorL2_"+method+"_"+str(reorthogonalizeFlag)+".pdf", bbox_inches='tight',dpi=200)
        plt.close()
        return None
    
    def test_lanczos(self, method="naive", reorthogonalizeFlag=False):
        trials = 10
        n = 250
        ks = np.array(list(range(10, 255, 5)))
        error = np.zeros((trials, len(ks)))
        func = functionNameMapper(method)
        
        for i in tqdm(range(trials)):
            A = np.random.randn(n,n)
            A = (A+A.T) / 2
            A /= np.linalg.norm(A, ord=2)
            v = np.random.randn(n)
            v /= np.linalg.norm(v)
            for k in range(len(ks)):
                Q, T = func(A, v, ks[k], return_type="QT", reorth=reorthogonalizeFlag)
                # after this, T and Q needs realignment.
                
                error[i, k] = np.linalg.norm((Q @ T @ Q.T) - A, ord=2)
                # error[i, k] = np.linalg.norm((Q.T @ Q) - np.eye(ks[k])) / ks[k]
        meanError = np.mean(error, axis=0)
        p20Error = np.percentile(error, q=20, axis=0)
        p80Error = np.percentile(error, q=80, axis=0)
        plot_vals(ks,\
                v1=meanError,\
                v1_lo=p20Error, v1_hi=p80Error,\
                label1=method,\
                xlabel="iterations", ylabel=r"$||\mathbf{A}-\mathbf{QTQ}^T||_2$",\
                filename="lanczos/compare_lanczos_"+method+"_"+str(reorthogonalizeFlag))
        return None
    
    def testPlotEigVals(self, method="naive", reorthogonalizeFlag=False):
        n= 250
        iters = 250
        A = np.random.randn(n,n)
        A = (A+A.T) / 2
        A /= np.linalg.norm(A, ord=2)
        v = np.random.randn(n)
        v /= np.linalg.norm(v)
        func = functionNameMapper(method)
        T = func(A, v, iters, return_type="T", reorth=reorthogonalizeFlag)
        local_eigs = np.real(np.linalg.eig(T)[0])
        local_eigs = padZeros(local_eigs, n)
        global_eigs = np.real(np.linalg.eig(A)[0])
        global_eigs = np.sort(global_eigs)
        plt.scatter(range(len(global_eigs)), global_eigs, color="blue")
        plt.scatter(range(len(local_eigs)), local_eigs, color="red")
        plt.show()
        return None

    def checkEigenvalueAlignment(self, method="naive", reorthogonalizeFlag=False):
        trials = 5
        n = 250
        ks = np.array(list(range(10, 255, 5)))
        error = np.zeros((trials, len(ks)))
        func = functionNameMapper(method)
        
        for i in tqdm(range(trials)):
            A = np.random.randn(n,n)
            A = (A+A.T) / 2
            A /= np.linalg.norm(A, ord=2)
            true_eigvals = np.real(np.linalg.eig(A)[0]) # true eigenvalues
            true_eigvals = np.sort(true_eigvals) # ascending order sort
            v = np.random.randn(n)
            v /= np.linalg.norm(v)
            
            for k in range(len(ks)):
                Q = func(A, v, ks[k], return_type="Q", reorth=reorthogonalizeFlag)
                local_lambda, local_vecs = np.linalg.eig(Q.T @ A @ Q)
                # local_lambda, local_vecs = sortValues(np.real(local_lambda), local_vecs)
                local_lambda = padZeros(np.real(local_lambda), n)
                print("observe values:", local_lambda, true_eigvals)
                error[i, k] = np.abs(local_lambda - true_eigvals).max() / np.abs(true_eigvals).max()
                error[i, k] = np.log(error[i, k])
    
        meanError = np.mean(error, axis=0)
        p20Error = np.percentile(error, q=20, axis=0)
        p80Error = np.percentile(error, q=80, axis=0)
        plt.plot(ks, meanError)
        plt.fill_between(ks, p20Error, p80Error, alpha=0.3)
        # plt.yscale("log")
        plt.grid(linewidth=1)
        plt.xlabel("iterations")
        plt.ylabel(r"$\log \left(\max_{i}|\lambda_i(\mathbf{Q}^T\mathbf{AQ}) - \lambda_i(\mathbf{A})| / \lambda_n(\mathbf{A})\right)$")
        plt.savefig("figures/unittests/lanczos/EigenvalueAlignment_"+method+"_"+str(reorthogonalizeFlag)+".pdf", bbox_inches='tight',dpi=200)
        return None
    
    def lanczosDebugDriver(self, test="eig alignment", method="naive", orthogonalizeFlag=False):
        if test == "eig alignment":
            self.checkEigenvalueAlignment(method, orthogonalizeFlag)
        if test == "convergence":
            # tested, plotted, works perfectly now!
            self.checkLanczosConvergence(method, orthogonalizeFlag)
        if test == "lanczos":
            self.test_lanczos(method, orthogonalizeFlag)
        if test == "check eigs":
            self.testPlotEigVals(method, orthogonalizeFlag)
        return None
        
    
if __name__ == '__main__':
    testWrapper().lanczosDebugDriver(test="check eigs", method="CTU", orthogonalizeFlag=True)