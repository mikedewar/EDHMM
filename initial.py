import numpy as np
import logging
import pymc

log = logging.getLogger('initial')

from utils import *

class Initial:
    """
    Defines an Initial distribution
    """
    @types(pi=np.ndarray)
    def __init__(self,pi):
        assert sum(pi)==1, \
            "invalid initial distribution"
        self.pi = pi
        self.pi.shape = (len(self.pi),1)
        self.shape = pi.shape
        self.dist = pymc.Categorical("initial distribution",pi)
    
    def __getitem__(self,key):
        return self.pi[key]
    
    def __len__(self):
        return self.pi.shape[0]
    
    def sample(self):
        return int(self.dist.random())
        
    def update(self, E):
        self.pi *= E[0]
        self.pi /= np.sum(self.pi)
    
    def report(self):
        log.info("\tinitial distribution:\n%s"%[round(p[0],2) for p in self.pi])