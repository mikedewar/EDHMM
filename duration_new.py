import math
import numpy as np
import logging

log = logging.getLogger('duration') 

class Poisson:
    def __init__(self, mu, alpha, beta, support_step=1):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.states = range(len(mu))
        self.support_step = support_step
    
    def likelihood(self,state,k):
        assert state in self.states
        return (k*np.log(self.mu[state])) - sum([np.log(ki+1) for ki in range(k)]) - self.mu[state]
    
    def sample_d(self, state):
        return np.random.poisson(self.mu[state])
    
    def sample_mu(self, Z):
        
        X = [z[0] for z in Z]
        k = dict([(i,[]) for i in self.states])
        now = X[0]
        for s in X:
            if now == s:
                try:
                    k[now][-1] += 1
                except IndexError:
                    # initial condition
                    k[now] = [1]
            else:
                now = s
                k[now].append(1)
        
        for i in self.states:
            log.debug("state: %s"%i)
            log.debug("observations: %s"%k[i])
        
        out=[]
        for i in self.states:
            alpha = self.alpha[i] + sum(k[i])
            beta = self.beta[i] + len(k[i])
            out.append(np.random.gamma(alpha, 1./beta))
        return out
    
    def update(self, Z):
        self.mu = self.sample_mu(Z)
    
    def support(self, state, threshold=0.001):
        log.info('finding support')
        # walk left
        d, dl = int(self.mu[state]), 1
        lold = self.likelihood(state,d)
        while dl > threshold:
            lnew = np.exp(self.likelihood(state,d))
            dl = abs(lnew-lold)
            lold = lnew
            d -= self.support_step
        left = max(1,d)
        # walk right
        d, dl = int(self.mu[state]), 1
        lold = self.likelihood(state,d)
        while dl > threshold:
            d += self.support_step
            lnew = np.exp(self.likelihood(state,d))
            dl = abs(lnew-lold)
            lold = lnew
        right = max(1,d)
        return int(left), int(right)
        
        
if __name__ == "__main__":
    import pylab as pb
    Z = np.load('Z.npy')
    D = Poisson(alpha=[3,3,3], beta=[0.8,0.5,0.25], mu=[3,5,10])
    for j in range(3):
        mus = np.array([D.sample_mu(Z)[j] for i in range(100)])
        pb.hist(mus,alpha=0.5)
    pb.show()