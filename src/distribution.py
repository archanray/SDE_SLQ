import numpy as np

class Distribution:
    def __init__(self, supports=None, weights=None):
        self.support = {}
        if supports is not None:
            self.set_weights(supports, weights)
        return None
    
    def set_weights(self, supports, weights):
        assert len(supports) == len(weights)
        
        weights = weights / np.sum(weights)
        
        for i in range(len(supports)):
            self.support[supports[i]] = weights[i]
        return None
    
    def showDistribution(self):
        """
        prints the distribution
        """
        print(self.support)
        return None

def mergeDistributions(D1, D2, func=None):
    keys1 = D1.support.keys()
    keys2 = D2.support.keys()
    keys = list(keys1)+list(keys2)
    newDistro = Distribution()
    for key in keys:
        newDistro.support[key] = func(D1.support.get(key, 0), D2.support.get(key, 0))
    return newDistro