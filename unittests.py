import numpy as np
from src.lanczos import naive_lanczos
from src.lanczos import modified_lanczos
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from src.moment_estimator import approxChebMomentMatching, discretizedJacksonDampedKPM
from src.utils import Wasserstein, jacksonDampingCoefficients
from src.distribution import Distribution
from src.optimizers import cvxpyL1Solver, L1Solver, pulpL1solver

class TestCalculations:
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
    
    def checkL1optimizer(self):
        N = 50
        d = 10000 #int(N**3/ 2)
        T = np.random.randn(N, d)
        z = np.random.rand(N)
        z = z/np.sum(z)
        
        solver1 = cvxpyL1Solver(T, z)
        solver1.minimizer()
        
        print("smallest value achieved using optimize.linprog:", solver1.res.func)
        
        # solver2 = pulpL1solver(T, z)
        # solver2.minimizer()
        
        # print("smallest value achieved using optimize.minimize:", solver2.res.func)
        # print("sum q values:", np.sum(solver2.res.x))
        
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
        plt.legend()
        savedir = os.path.join("figures", "unittests")
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        savefilepath = os.path.join(savedir, filename+ext)
        plt.savefig(savefilepath, bbox_inches='tight',dpi=200)
        return None
    
    def test_lanczos(self):
        trials = 10
        eps = 1e-25
        n = 100
        ks = np.array(list(range(10, n+10, 10)))
        error1 = np.zeros((trials, len(ks)))
        error2 = np.zeros((trials, len(ks)))
        
        for i in tqdm(range(trials)):
            A = np.random.randn(n,n)
            v = np.random.randn(n)
            for k in range(len(ks)):
                Q1, T1 = naive_lanczos(A, v, ks[k], return_type="QT")
                Q2, T2 = modified_lanczos(A, v, ks[k], return_type="QT")
                error1[i, k] = np.log((np.linalg.norm(Q1@T1@Q1.T - A, ord=2) / np.linalg.norm(A, ord=2)) + eps)
                error2[i, k] = np.log((np.linalg.norm(Q2@T2@Q2.T - A, ord=2) / np.linalg.norm(A, ord=2)) + eps)
        meanError1 = np.mean(error1, axis=0)
        meanError2 = np.mean(error2, axis=0)
        p20Error1 = np.percentile(error1, q=20, axis=0)
        p80Error1 = np.percentile(error1, q=80, axis=0)
        p20Error2 = np.percentile(error2, q=20, axis=0)
        p80Error2 = np.percentile(error2, q=80, axis=0)
        self.plot_vals(np.log(ks / n),\
                        meanError1, meanError2,\
                        p20Error1, p80Error1,\
                        p20Error2, p80Error2,\
                        "Naive", "Modified",\
                        "trial", "log relative approximation error",\
                        filename="compare_lanczos")
        
    def testMomentMatchings(self):
        Ns = np.array(list(range(10,100,10)))
        trials = 4
        errors = np.zeros((trials, len(Ns)))
        
        for i in range(trials):
            for j in tqdm(range(len(Ns))):
                tau = np.random.rand(Ns[j])
                tau = tau / np.sum(tau)
                support_tau = np.array(list(range(Ns[j])))
                support_q, q = approxChebMomentMatching(tau)
                D1, D2 = Distribution(support_tau, tau), Distribution(support_q, q)
                errors[i,j] = Wasserstein(D1, D2)
        meanErrors = np.mean(errors, axis=0)
        pc20Errors = np.percentile(errors, q=20, axis=0)
        pc80Errors = np.percentile(errors, q=80, axis=0)
        
        self.plot_vals(x=Ns,
                       v1=meanErrors, 
                       v1_lo=pc20Errors, 
                       v1_hi=pc80Errors, 
                       xlabel="Ns", 
                       ylabel="W1 distance",
                       label1="ChebMM",
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
    
    def checkOutputs(self):
        N = 19
        tau = np.random.rand(N)
        tau = tau / np.sum(tau)
        supports, probs = discretizedJacksonDampedKPM(tau)
        return None
        
        
        

if __name__ == '__main__':
    TestCalculations().test_lanczos()