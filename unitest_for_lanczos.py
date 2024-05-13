import numpy as np
from src.lanczos import naive_lanczos, modified_lanczos, exact_lanczos, wiki_lanczos, CGMM_lanczos, QR_lanczos
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
    return np.pad(eigvals, (0, n-len(eigvals)), mode='constant', constant_values=0)

def functionNameMapper(method):
    if method == "naive":
        return naive_lanczos
    elif method == "modified":
        return modified_lanczos
    elif method == "exact":
        return exact_lanczos
    elif method == "wiki":
        return wiki_lanczos
    elif method == "CGMM":
        return CGMM_lanczos
    elif method == "QR":
        return QR_lanczos
    else:
        raise ValueError("Invalid method name")
    return None

class testWrapper():
    def checkLanczosConvergence(self, method="naive", reorthogonalizeFlag=False):
        trials = 5
        n = 500
        iterations = np.arange(10,110,10)
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
            rand_vec = np.random.randn(n)
            rand_vec /= np.linalg.norm(rand_vec)
            
            for q in range(len(iterations)):
                Q, T = func(data, rand_vec, iterations[q], return_type="QT", reorth=reorthogonalizeFlag)
                
                local_lambda, local_vecs = np.linalg.eig(T)
                local_lambda, local_vecs = sortValues(local_lambda, local_vecs)
                local_lambda = padZeros(local_lambda, n)
                # plt.plot(range(len(local_lambda)), local_lambda)
                # plt.show()
                # sys.exit(1)
                error1[t, q] = np.abs(true_lambda[index] - local_lambda[index])
                
                Z_i = np.dot(Q, local_vecs[:,index].T)
                L_iZ_i = local_lambda[index] * Z_i
                AZ_i = np.dot(data, Z_i.T)
                error2[t, q] = np.linalg.norm(L_iZ_i - AZ_i)
            pass
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
        n = 500
        ks = np.array(list(range(10, 505, 5)))
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
                error[i, k] = np.linalg.norm((Q @ T @ Q.T) - A, ord=2)
                # error[i, k] = np.linalg.norm((Q.T @ Q) - np.eye(ks[k])) / ks[k]
        meanError = np.mean(error, axis=0)
        p20Error = np.percentile(error, q=20, axis=0)
        p80Error = np.percentile(error, q=80, axis=0)
        self.plot_vals(ks,\
                        v1=meanError,\
                        v1_lo=p20Error, v1_hi=p80Error,\
                        label1="QR",\
                        xlabel="iterations", ylabel=r"$||\mathbf{A}-\mathbf{QTQ}^T||_2$",\
                        filename="lanczos/compare_lanczos_"+method+"_"+str(reorthogonalizeFlag))
        # self.plot_vals(ks,\
        #                 v1=meanError,\
        #                 v1_lo=p20Error, v1_hi=p80Error,\
        #                 label1="naive",\
        #                 xlabel="iterations", ylabel=r"$||\mathbf{I}-\mathbf{Q}^T\mathbf{Q}||_F$",\
        #                 filename="lanczos/QQ^T_"+str(flag))
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
            true_eigvals = np.sort(true_eigvals)
            v = np.random.randn(n)
            v /= np.linalg.norm(v)
            
            for k in range(len(ks)):
                Q = func(A, v, ks[k], return_type="Q", reorth=reorthogonalizeFlag)
                local_lambda, local_vecs = np.linalg.eig(Q.T @ A @ Q)
                local_lambda, local_vecs = sortValues(local_lambda, local_vecs)
                local_lambda = padZeros(local_lambda, n)
                local_lambda = np.sort(local_lambda)
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
            self.checkLanczosConvergence(method, orthogonalizeFlag)
        if test == "lanczos":
            self.test_lanczos(method, orthogonalizeFlag)
        return None
        
    
if __name__ == '__main__':
    testWrapper().lanczosDebugDriver(test="eig alignment", method="modified", orthogonalizeFlag=True)