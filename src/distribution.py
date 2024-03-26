import numpy as np

class Distribution:
    def __init__(self, support={}):
        self.support = support
        return None
    
    def set_weights(self, supports, weights):
        assert len(supports) == len(weights)
        # assert np.all(weights > 0)
        
        weights = weights / np.sum(weights)
        
        for i in range(len(supports)):
            self.support[supports[i]] = weights[i]
        return None

def mergeDistributions(D1, D2, func=None):
    keys1 = D1.support.keys()
    keys2 = D2.support.keys()
    keys = list(keys1)+list(keys2)
    newDistro = Distribution()
    for key in keys:
        newDistro.support[key] = func(D1.support.get(key, 0), D2.support.get(key, 0))
    return newDistro