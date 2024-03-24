import numpy as np
from tqdm import tqdm
from src.approximator import slq, bkde

def SDE(A, params):
    """
    wrapper function for all kinds fo SDEs
    """
    assert A == A.T
    outputs = {}
    outputs["distros"] = {}
    outputs["seeds"] = []
    methods = {"slq": slq, "krylov": bkde}
    
    method = methods[params["method"]]
    for t in tqdm(range(params["trials"])):
        seed = t
        np.random.seed(t)
        outputs["seeds"].append(seed)
        ks = params["block_sizes"]
        l = params["iters"]
        for k in ks:
            outputs["distros"][str(t)+","+str(k)] = method(A, k, l, seed=seed)
    return outputs["distros"], outputs["seeds"]