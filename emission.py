import pymc
from utils import types
import pylab as pb
import numpy as np
import logging
log = logging.getLogger('emission')


class Emission():
    """
    Class describing an arbitrary output distribution. You should overload 
    self.likelihood and self.sample to make a new distribution.
    """
    
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
        self.M.dist[state].value = Y
        return self.M.dist[state].logp
    
    def sample(self, state):
        """
        returns a sampled output from the emission distribution
        
        Parameters
        ----------
        state: int
            state of the EDHMM
        """
        self.M.dist[state].observed = False
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
        return len(self.M.dist)

class Gaussian(Emission):
    
    def __init__(self, mu, alpha, beta, prior_mean, prior_precision, observations=None):
        
        assert len(mu) == len(alpha)
        assert len(mu) == len(beta)
        assert len(mu) == len(prior_mean)
        assert len(mu) == len(prior_precision)
        
        states = range(len(mu))
        
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        
        mean = [
            pymc.Normal('mean_%s'%i, mu[i], 0.1)
            for i in states
        ]
        precision = [
            pymc.Gamma('prec_%s'%i, alpha[i], beta[i])
            for i in states
        ]
        if observations:
            dist = [
                pymc.Normal(
                    'emission_%s'%j, 
                    mean[j], 
                    precision[j],
                    values=observations[j],
                    observed = True
                ) 
                for j in states
            ]
        else:
            dist = [
                pymc.Normal('emission_%s'%j, mean[j], precision[j]) 
                for j in states
            ]
        self.dim = 1
        self.M = pymc.MCMC(pymc.Model(
            {
                'dist': dist, 
                'precision': precision,
                'mean' : mean
            }
        ))
    
    def update_parameters(self, prior_mean, prior_precision):
        return Gaussian(
            mu=self.mu,
            alpha=self.alpha,
            beta=self.beta,
            prior_mean = prior_mean,
            prior_precision = prior_precision
        )
    
    def update_observations(self,Z,Y):
        X = [z[0] for z in Z]
        states = pb.unique(X)
        states.sort()
        n = dict([(i,[]) for i in states])
        for t,s in enumerate(X):
            n[s].append(np.array([Y[t]]))
        
        return Gaussian(
            mu=self.mu,
            alpha=self.alpha,
            beta=self.beta,
            prior_mean = self.prior_mean,
            prior_precision = self.prior_precision,
            observations = n
        )
        
    def update(self,Z,Y):
        log.info('updating O')
        O = self.update_observations(Z,Y)
        mean, precision = O.sample_parents()
        O = O.update_parameters(mean, precision)
        return O, mean, precision
    
    def report(self):
        pass
    
    def plot(self,x):
        y = pb.zeros(len(x))
        for i, xi in enumerate(x):
            for k in range(len(self.dist)):
                y[i] += self.likelihood(k, xi)
        pb.plot(x,y)
        pb.show()
    
    def sample_parents(self):
        self.M.sample(iter=101,burn=100)
        states = range(len(self.M.mean))
        means = np.array(
            [self.M.trace('mean_%s'%i)[:] for i in states]
        ).flatten()
        precisions = np.array(
            [self.M.trace('prec_%s'%i)[:] for i in states]
        )
        return means, precisions
    
    def sample_from_prior(self,state):
        self.M.mean[state].observed = False
        self.M.mean[state].value = self.prior_mean[state]
        self.M.precision[state].observed = False
        self.M.precision[state].value = self.prior_precision[state]
        return self.M.dist[state].random()

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
        mu=[-10,0,10],
        alpha=[1,1,1],
        beta=[1,1,1],
        prior_mean = [-10,0,10],
        prior_precision = [1,1,1]
    )
    
    Z = np.load('Z.npy')
    Y = np.load('Y.npy')
    
    O.update(Z,Y)
    
    print O.M.dist[0].value
    
    print O.likelihood(Z[0][0],Y[0])
    
    pb.hist(
        [
            O.sample_from_prior(j) 
            for i in range(1000) 
            for j in range(3)
        ],
        bins=100
    )
    pb.show()
    