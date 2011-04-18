import pymc
from utils import types
import numpy as np
import logging
log = logging.getLogger('emission')


class Emission():
    """
    Class describing an arbitrary output distribution. You should overload 
    self.likelihood and self.sample to make a new distribution.
    """
    def __init__(self):
        # every Emission distribution should have a dist attribute, which
        # should be a pymc.Stochastic object
        assert all(
            [
                isinstance(dist_i, pymc.Stochastic) 
                for dist_i in self.dist
            ]
        ) , self.dist
        assert type(self.dim) is int, self.dim
    
    def likelihood(self, state, Y):
        """
        returns the log likelihood of the data Y under the model
        
        Parameters
        ----------
        state: int
            state of the EDHMM
        Y: float, list or array
            observation 
        """
        self.dist[state].set_value(Y)
        return np.exp(self.dist[state].logp)
    
    def sample(self, state):
        """
        returns a sampled output from the emission distribution
        
        Parameters
        ----------
        state: int
            state of the EDHMM
        """
        return self.dist[state].random()
    
    def update(self, Y, gamma):
        raise NotImplementedError
        
    def compare(self, other):
        raise NotImplementedError
    
    def report(self):
        raise NotImplementedError
    
    def __call__(self, state, Y=None):
        if Y is None:
            return self.sample(state, duration)
        else:
            return self.likelihood(state, Y)
    
    def __len__(self):
        return len(self.dist)

class Gaussian(Emission):
    
    @types(means=list, precisions=list)
    def __init__(self, means, precisions):
        self.dist = [
            pymc.Normal('emission_%s'%j, mean, precision) 
            for j,(mean, precision) in enumerate(zip(means, precisions))
        ]
        self.dim = 1
        Emission.__init__(self)
    
    def update(self, gamma, Y):
        """
        gamma : list of np.ndarray
            gamma[t][i] = p(x_t = i | Y)
        """
        mean = np.zeros(len(self.dist))
        for i,d in enumerate(self.dist):
            num = np.sum([g[i] * y for (g,y) in zip(gamma,Y)])
            den = np.sum([g[i] for g in gamma])
            mean =  float(num / den)
            # this calculation of the precision is overly complex, but I only
            # really want to write one update method for this and the multi-
            # variate case. Sorry.
            num = np.sum([
                g[i] * np.outer(y - mean, y-mean) 
                for (g,y) in zip(gamma,Y)
            ])
                
            precision = float(1.0/(num/den))
            self.dist[i] = pymc.Normal('emission_%s'%i, mean, precision)
    
    def report(self):
        log.info(
            'means:\n%s'%[round(p.parents['mu'],2) for p in self.dist]
        )
        log.info(
            'precisions:\n%s'%[round(p.parents['tau'],2) for p in self.dist]
        )
        

class MultivariateGaussian(Emission):
    
    @types(means=list, covariances=list)
    def __init__(self, means, covariances):
        self.dist = [
            pymc.MvNormalCov('emission_%s'%j, mean, covariance)
            for j, (mean,covariance) in enumerate(zip(means,covariances))
        ]
        self.dim = 2
        Emission.__init__(self)
    
    def update(self, gamma, Y):
        """
        gamma : list of np.ndarray
            gamma[t][i] = p(x_t = i | Y)
        """
        mean = np.zeros(len(self.dist))
        for i,d in enumerate(self.dist):
            num = np.sum([g[i] * y for (g,y) in zip(gamma,Y)],0)
            den = np.sum([g[i] for g in gamma],1)
            mean = num / den
            # this calculation of the precision is overly complex, but I only
            # really want to write one update method for this and the multi-
            # variate case. Sorry.
            num = np.sum([
                g[i] * np.outer(y - mean[i], y-mean[i]) 
                for (g,y) in zip(gamma,Y)
            ],0)
            precision = num / den
            self.dist[i] = pymc.MvNormalCov('emission_%s'%i, mean, precision) 

if __name__ == "__main__":
    O = Gaussian(
        means=[0,1],
        precisions=[1,1]
    )
    O = MultivariateGaussian(
        means=[[0,0],[1,1]],
        covariances=[[[1,0],[0,1]],[[1,0],[0,1]]]
    )