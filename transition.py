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
        self.K = A.shape[0]
        self.states = range(self.K)
        
        p = np.zeros((self.K, self.K))
        for i in self.states:
            for j in self.states:
                if i!=j:
                    p[i,j] = 1.0/(self.K-1)
        # add a little bit to everywhere and renormalise
        p+=0.05
        for row in p:
            row /= row.sum()
        
        Ps = [
            pymc.Dirichlet("p_%s"%i, p[i])
            for i in self.states
        ]
        
        self.dist = [
            pymc.Categorical('A_%s'%i, Ps[i])
            for i in self.states
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
        
    def update_parameters(self,p):
        pass
    
    def update_observations(self,Z):
        X = [z[0] for z in Z]
        n = [[] for i in self.states]
        now = X[0]
        for s in X:
            if now != s:
                n[now].append(s)
                now = s
        
    
    def update(self,Z):
        pass
    
    def report(self):
        log.info("transition matrix:\n%s"%self.A.round(4))
        
    def __call__(self, i, j=None):
        if j is None:
            return self.sample(i)
        else:
            return self.likelihood(i, j)