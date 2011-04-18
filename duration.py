import pymc
import numpy as np
from utils import *
import logging

log = logging.getLogger('duration')


class Duration(object):
    """
    Class describing an arbitrary output distribution. Built on top of pymc,
    you need to provide a K-length list of vald pymc.Stochastic objects in the
    as the `dist` attribute, before calling Duration.__init__().
    """
    def __init__(self):
        # every Duration distribution should have a dist attribute, which
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
            return np.ceil(self.dist[state].random())
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


class LogNormal(Duration):
    "LogNormal Duration distribution"
    
    def __init__(self,mus,taus):
        self.dist = [
            pymc.Lognormal('duration_%s'%j,m,t)
            for j,(m,t) in enumerate(zip(mus,taus))
        ]
        Duration.__init__(self)

class Poisson(Duration):
    "Poisson Duration distribution"
    
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
    
    def report(self):
        log.debug(
            'duration paramters:\n%s'%
            [round(p.parents['mu'],2) for p in self.dist]
        )
        
    def update(self, Dcal):
        log.debug('updating duration distribution')
        D = sum(Dcal) / (len(Dcal) - 1)
        
        #print D
        #print np.arange(1,D.shape[1]+1)
        #print D * np.arange(1,D.shape[1]+1)
        #print (D * np.arange(1,D.shape[1]+1)).sum(1)
        
        
        lambdas = ((D * np.arange(1,D.shape[1]+1)).sum(1) / D.sum(1)) 
        #print lambdas
        
        for l in lambdas:
            assert not np.isnan(l)
            assert l >= 1, l
        self.dist = [
            pymc.Poisson('duration_%s'%j,l)
            for j,l in enumerate(lambdas)
        ]
    
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
    

