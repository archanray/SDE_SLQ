import numpy as np
from src.lanczos import naive_lanczos
from src.lanczos import modified_lanczos
import unittest
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class TestCalculations(unittest.TestCase):
    def plot_vals(self, x, v1, v2, v1_lo, v1_hi, v2_lo, v2_hi,\
                  label1, label2, xlabel, ylabel, filename="plot", ext=".pdf"):
        plt.rcParams.update({'font.size': 13})
        plt.plot(x, v1, label=label1)
        plt.fill_between(x, v1_lo, v1_hi, alpha=0.2)
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
        n = 200
        ks = np.array(list(range(10, n, 10)))
        error1 = np.zeros((trials, len(ks)))
        error2 = np.zeros((trials, len(ks)))
        
        for i in tqdm(range(trials)):
            A = np.random.randn(n,n)
            v = np.random.randn(n)
            for k in range(len(ks)):
                Q1, T1 = naive_lanczos(A, v, ks[k], return_type="QT")
                Q2, T2 = modified_lanczos(A, v, ks[k], return_type="QT")
                error1[i, k] = np.log((np.linalg.norm(Q1@T1@Q1.T - A) / np.linalg.norm(A, ord=2)) + eps)
                error2[i, k] = np.log((np.linalg.norm(Q2@T2@Q2.T - A) / np.linalg.norm(A, ord=2)) + eps)
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

if __name__ == '__main__':
    unittest.main()