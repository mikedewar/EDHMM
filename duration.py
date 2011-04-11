import pymc
import numpy as np
from utils import *

class Duration(object):
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
    
    def likelihood(self, state, duration):
        """
        returns the log likelihood of the duraiton given the current state 
        under the model
        """
        self.dist[state].set_value(duration)
        return np.exp(self.dist[state].logp)
    
    def sample(self, state):
        """
        returns a sampled duration from the model
        """
        try:
            return self.dist[state].random()
        except IndexError:
            print "your sampling Duration distribution is throwing an index error."
            print "\tthe state chosen is: %s"%state
            print "\tand the expected max is: %s"%(len(self)-1)
            raise
    
    def learn(self, eta):
        """
        returns a parameter set given 
        :math: `\\eta[t][i,d] = p(x_{t-d+1}=i, ldots x_{t}=i,Y)`
        """
        raise NotImplementedError
    
    def plot(self):
        raise NotImplementedError
    
    def compare(self, new_dist):
        raise NotImplementedError
    
    def __getitem__(self, i):
        raise NotImplementedError
    
    def __call__(self, state, duration=None):
        if duration is None:
            return self.sample(state)
        else:
            return self.likelihood(state, duration)
    
    def __len__(self):
        return len(self.dist)


class Poisson(Duration):
    "Poisson Duration distribution"
    @types(lambdas=list)
    def __init__(self, lambdas):
        """
        Parameters
        ---------
        mus: list
            list of rate parameters, one per state
        """
        self.dist = [
            pymc.Poisson('duration_%s'%j,l)
            for j,l in enumerate(lambdas)
        ]
        Duration.__init__(self)
        
    def learn(self, eta):
        mus = np.zeros(len(self.mus))
        dur_range = range(1, eta[0].shape[1] + 1)
        for j in range(len(self.mus)):
            num = 0.0
            den = 0.0
            for t,e in enumerate(eta):
                for di,d in enumerate(dur_range):
                    num += eta[t][j, di] * d
                    den += eta[t][j, di]
            mus[j] = num/den
            assert not np.isnan(mus[j]), (num, den, dur_range)
            
        return Poisson(mus)
    
    def plot(self, max_duration=30):
        num_states = len(self.mus)
        for j in range(num_states):
            pb.subplot(num_states, 1, j+1)
            pb.plot(
                range(1, max_duration+1),
                [self.likelihood(j, d) 
                 for d in range(1, max_duration+1)]
            )
    
    def compare(self, dur_dist):
        assert dur_dist.__class__ is Poisson
        for (mu_self,mu_test) in zip(self.mus,dur_dist.mus):
            yield abs(mu_self-mu_test)
    

if __name__ == "__main__":
    D = Poisson([10,20])