import pymc
import numpy as np
from utils import *
import logging
import pylab as pb

log = logging.getLogger('duration')

class Duration(object):
    """
    Class describing an arbitrary output distribution. Built on top of pymc,
    you need to provide a K-length list of vald pymc.Stochastic objects in the
    as the `dist` attribute, before calling Duration.__init__().
    """
    
    def likelihood(self, state, duration):
        """
        returns the log likelihood of the duraiton given the current state 
        under the model
        """
        self.prior[state].value = duration
        return self.prior[state].logp
    
    def sample(self, state):
        """
        returns a sampled duration from the model
        """
        try:
            return np.ceil(self.M.dist[state].random())
        except IndexError:
            print "your sampling Duration distribution is throwing an index error."
            print "\tthe state chosen is: %s"%state
            print "\tand the expected max is: %s"%(len(self)-1)
            raise
    
    def sample_parents(self):
        """
        draws a sample from the parameters
        """
        self.M.sample(iter=101, burn=100)
        samples = np.array(
            [self.M.trace('mu_%s'%i)[:] for i in self.states]
        ).flatten()
        return samples

        
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
        return len(self.M.dist)


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
    
    def __init__(self, alpha, beta, prior_mu, observations=None):
        """
        Parameters
        ---------
        mus: list
            list of rate parameters, one per state
        """
        assert len(alpha) == len(beta)
        assert len(alpha) == len(prior_mu)
        if observations:
            assert len(alpha) == len(observations), (len(alpha),len(observations),observations)
        
        self.alpha = alpha
        self.beta = beta
        self.prior_mu = prior_mu
        self.states = range(len(alpha))
        mus = [
            pymc.Gamma("mu_%s"%i, alpha=alpha[i], beta=beta[i])
            for i in range(len(alpha))
        ]
        if observations:
            try:
                dist = [
                    pymc.Poisson('dist_%s'%j,l,observed=True,value=observations[j])
                    for j,l in enumerate(mus)
                ]
            except IndexError:
                print observations
                print mus
                raise
        else:
            dist = [
                pymc.Poisson('dist_%s'%j,l,observed=True)
                for j,l in enumerate(mus)
            ]
        self.prior = [
            pymc.Poisson('dist_%s'%j,prior_mu[j])
            for j,l in enumerate(mus)
        ]
        self.M = pymc.MCMC(pymc.Model({'dist':dist, 'mu':mus}))
    
    def report(self):
        pass
    
    def update_observations(self, Z):
        
        X = [z[0] for z in Z]
        durations = dict([(i,[]) for i in self.states])
        now = X[0]
        for s in X:
            if now == s:
                try:
                    durations[now][-1] += 1
                except IndexError:
                    # initial condition
                    durations[now] = [1]
            else:
                now = s
                durations[now].append(1)
        
        return Poisson(
            alpha=self.alpha, 
            beta=self.beta, 
            prior_mu=self.prior_mu,
            observations = durations
        )
    
    def update_parameters(self, prior_mu):
        return Poisson(
            alpha=self.alpha, 
            beta=self.beta, 
            prior_mu=prior_mu,
        )
    
    def update(self,Z):
        log.info('updating D')
        D = self.update_observations(Z)
        mu = D.sample_parents()
        D = D.update_parameters(mu)
        return D, mu
    
    def plot(self, max_duration=30):
        for j in self.states:
            pb.subplot(num_states, 1, j+1)
            pb.plot(
                range(1, max_duration+1),
                [self.likelihood(j, d) 
                 for d in range(1, max_duration+1)]
            )
    
    def support(self, state, threshold=0.001):
        mu = self.M.dist[state].parents['mu'].value
        # walk left
        d, dl = mu, 1
        lold = self(state,d)
        while dl > threshold:
            lnew = pb.exp(self(state,d))
            dl = abs(lnew-lold)
            lold = lnew
            d -= 1
        left = d
        # walk right
        d, dl = mu, 1
        lold = self(state,d)
        while dl > threshold:
            d += 1
            lnew = pb.exp(self(state,d))
            dl = abs(lnew-lold)
            lold = lnew
        right = d
        return int(left), int(right)
    
    def sample_from_prior(self,state):
        return self.prior[state].random()
    

if __name__ == "__main__":
    Z = np.load('Z.npy')
    Y = np.load('Y.npy')
    D = Poisson(alpha=[3,3,3], beta=[4,4,4], prior_mu=[3,5,10])
    D = D.update_observations(Z)
    mu = D.sample_parents()
    D = D.update_parameters(mu)
    
    for j in range(3):
        pb.hist(
            [
                D.sample_from_prior(j) 
                for i in range(1000) 
            ],
            bins=20,
            alpha=0.5
            
        )
    pb.show()