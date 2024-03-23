import numpy as np

class Distribution:
    def __init__(self, support={}):
        self.support = support
        return None
    
    def set_weights(self, supports, weights):
        assert len(supports) == len(weights)
        for i in range(len(supports)):
            self.support[supports[i]] = weights[i]
        return None

def mergeDistributions(D1, D2, func=None):
    keys1 = D1.keys()
    keys2 = D2.keys()
    keys = keys1+keys2
    newDistro = Distribution()
    for key in keys:
        newDistro.support[key] = func(D1.get(key, 0), D2.get(key, 0))
    return newDistro