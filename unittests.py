import numpy as np
from src.lanczos import exact_lanczos
from src.lanczos import lanczos
import unittest

class TestCalculations(unittest.TestCase):
    def test_lanczos(self):
        n = 20
        A = np.random.randn(n,n)
        v = np.random.randn(n)
        k = n
        Q1, T1 = lanczos(A, v, k, return_type="QT")
        Q2, T2 = exact_lanczos(A, v, k, return_type="QT")
        error1 = np.linalg.norm(Q1@T1@Q1.T - A)
        error2 = np.linalg.norm(Q2@T2@Q2.T - A)
        print(error1, error2)
        self.assertTrue(error1 == error2, 'Matrices are unequal.')

if __name__ == '__main__':
    unittest.main()