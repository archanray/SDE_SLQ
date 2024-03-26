import numpy as np
import os
import sys
import pickle
from src.distribution import Distribution
from src.utils import Wasserstein

def summarizer(dataset, method, name_adder="single_eval_method_"):
    file_to_read = os.path.join("outputs", dataset, name_adder+method+".pkl")
    if not os.path.exists(file_to_read):
        print("the main.py for this method or dataset hasn't been executed")
        sys.exit(1)
    with open(file_to_read, "rb") as f:
        load_vals = pickle.load(f)
    
    eps = 1e-20
    inputSupports = load_vals["true_spectrum"]
    inputWeights = np.ones_like(inputSupports) / len(inputSupports)
    inputDistro = Distribution(inputSupports, inputWeights)
    
    ts = list(range(load_vals["trials"]))
    ks = load_vals["block_sizes"]
    
    errors = np.zeros((len(ts), len(ks)))
    for t in ts:
        for j in range(len(ks)):
            errors[t,j] = Wasserstein(inputDistro, load_vals["spectral_density_estimates"][str(t)+","+str(ks[j])])
    # print(errors)
    # set axis 0 to compte means and standard deviations along the rows
    logMeanError = np.mean(np.log(errors + eps), axis=0)
    logp20s = np.percentile(np.log(errors + eps), q=20, axis=0)
    logp80s = np.percentile(np.log(errors + eps), q=80, axis=0)
    logBlockSizeProportionals = np.log(np.array(load_vals["block_sizes"]) / len(inputSupports))
    
    # print(logMeanError)
    return logMeanError, logp20s, logp80s, logBlockSizeProportionals