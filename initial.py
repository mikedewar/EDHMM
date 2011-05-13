import numpy as np
import logging
import pymc

log = logging.getLogger('initial')

from utils import *

class Initial:
    """
    Defines an Initial distribution
    """
    def __init__(self, K, beta=0.001):
        self.K = K
        self.beta = beta
        state_dist = pymc.Categorical('state_init', [1./K for i in range(K)])
        dur_dist = pymc.Exponential('dur_init', beta)
        self.dist = pymc.Model({
            "s_init":state_dist, 
            "d_init":dur_dist
        })
    
    def __getitem__(self,key):
        return self.pi[key]
    
    def __call__(self, z=None):
        if z is None:
            return self.sample()
        else:
            return self.likelihood(z)
    
    def __len__(self):
        return self.K
    
    def sample(self):
        self.dist.draw_from_prior()
        x = int(self.dist.s_init.value)
        d = int(round(self.dist.d_init.value))
        return x, d
    
    def likelihood(self, z):
        self.dist.s_init.set_value(z[0])
        self.dist.d_init.set_value(z[1])
        l_x = self.dist.s_init.logp
        l_d = self.dist.d_init.logp
        return l_x + l_d
        
    def update(self, E):
        raise NotImplementedError
    
    def report(self):
        report = "initial distribution: K=%s, beta=%s"%(self.K,self.beta)
        log.info(report)