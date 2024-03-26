## Code for our work on sublinear spectral density estimation

## Requirements
1. Python3
2. Scipy (any version with the `optimize` library)
3. Numpy
4. tqdm
5. pickle
6. cvxpy

## Entry points:
1. run `main.py` -- check options in the file to choose from. This code runs the estimators and saves the results in the folder outputs
2. run `analyze.py` -- check options in the file to choose from. This code loads up the previous runs for a specific dataset and all corresponding methods, computes the errors and plots them and stores the figures in the folder `figures`.
