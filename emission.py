import pymc
from utils import types
import pylab as pb
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
        return self.dist[state].logp
    
    def sample(self, state):
        """
        returns a sampled output from the emission distribution
        
        Parameters
        ----------
        state: int
            state of the EDHMM
        """
        return self.dist[state].random()
    
    def update(self, sample):
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
    
    def update(self,sample):
        self.dist = [
            pymc.Normal(
                'emission_%s'%j, 
                sample[0][j], 
                1.0/sample[1][j].flatten()
            ) 
            for j in range(len(sample[0]))
        ]
    
    def report(self):
        log.info(
            'means:\n%s'%[round(p.parents['mu'],4) for p in self.dist]
        )
        log.info(
            'precisions:\n%s'%[round(p.parents['tau'],4) for p in self.dist]
        )
    
    def plot(self,x):
        y = pb.zeros(len(x))
        for i, xi in enumerate(x):
            for k in range(len(self.dist)):
                y[i] += self.likelihood(k, xi)
        pb.plot(x,y)
        pb.show()

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
        means=[-5,1],
        precisions=[1,1]
    )
    O = MultivariateGaussian(
        means=[[0,0],[1,1]],
        covariances=[[[1,0],[0,1]],[[1,0],[0,1]]]
    )