import pymc
from utils import types
import numpy as np

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
    
    def learn(self, Y, gamma):
        raise NotImplementedError
        
    def compare(self, other):
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

class MultivariateGaussian(Emission):
    
    @types(means=list, covariances=list)
    def __init__(self, means, covariances):
        self.dist = [
            pymc.MvNormalCov('emission_%s'%j, mean, covariance)
            for j, (mean,covariance) in enumerate(zip(means,covariances))
        ]
        self.dim = 2
        Emission.__init__(self)

if __name__ == "__main__":
    O = Gaussian(
        means=[0,1],
        precisions=[1,1]
    )
    O = MultivariateGaussian(
        means=[[0,0],[1,1]],
        covariances=[[[1,0],[0,1]],[[1,0],[0,1]]]
    )