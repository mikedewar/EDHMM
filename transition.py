import numpy as np
import logging
import pymc

log = logging.getLogger('transition')

from utils import *


class Transition:
    """
    Defines a Transition distribution as collection of categorical distributions
    """
    @types(A=np.ndarray)
    def __init__(self, A):
        assert A.shape[0] == A.shape[1], \
            "non-square transition matrix"
        assert all(np.diag(A)==0), \
            "the EDHMM does not allow self transitions"
        A += 0.00001
        for row in A:
            row/=row.sum()
        
        for row in A:
            assert isprob(row), \
                "invalid transition matrix:\n%s, \n%s"%(A, sum(row))
        self.A = A
        self.shape = A.shape
        self.dist = [
            pymc.Categorical("transition from %s"%i, a)
            for i, a in enumerate(A)
        ]
    
    def __getitem__(self,key):
        return self.A[key]
    
    def __len__(self):
        return self.A.shape[0]
    
    def __str__(self):
        return str(self.A)
    
    def likelihood(self,i,j):
        return np.log(self.A[i,j])
    
    def sample(self,i):
        return self.dist[i].random()

    def update(self,P):
        self.dist = [
            pymc.Categorical("transition from %s"%i, p)
            for i, p in enumerate(P)
        ]
    
    def report(self):
        log.info("transition matrix:\n%s"%self.A.round(4))
        
    def __call__(self, i, j=None):
        if j is None:
            return self.sample(i)
        else:
            return self.likelihood(i, j)