import pymc
import numpy as np
import logging

log = logging.getLogger('emissions') 


log_2_pi = np.log(2*np.pi)
invwishart = lambda nu, L: pymc.InverseWishart("invwishart", nu, L).random()
mvnormal = lambda mu, tau: pymc.MvNormal('mvnormal', mu, tau).random()
#invwishart_like = lambda x, nu, L: pymc.inverse_wishart_like(x,nu,L)






class Gaussian:
    
    def __init__(self, nu, Lambda, mu_0, kappa, mu, tau):
        self.nu = nu
        self.Lambda = Lambda
        self.mu_0 = mu_0 # mu_0 is the mean of the prior on the mean
        self.kappa = float(kappa)
        
        self.mu = mu # mu is the current value of the mean for each state
        self.tau = tau # tau is the current precision matrix for each state
        
        try:
            self.log_det_inv_tau = [
                np.log(np.linalg.det(np.linalg.inv(t)))
                for t in self.tau
            ]
        except IndexError:
            self.log_det_inv_tau = np.log([1.0/t for t in self.tau])
        
        self.states = range(len(mu))
        self.K = len(self.states)
        
        
    def likelihood(self, state, obs):
        assert state in self.states, (state, self.states)
        x = (obs - self.mu[state])
        try:
            k = len(x)
        except TypeError:
            k = 1
        y = float(
            - (0.5 * k *log_2_pi) 
            - (0.5 * self.log_det_inv_tau[state]) 
            - (0.5 * np.inner(np.inner(x,self.tau[state]),x))
        )
        return y
    
    def sample_obs(self,state):
        assert state in self.states, (state, self.states)
        return mvnormal(self.mu[state], self.tau[state])
    
    def sample_mean_prec(self, Z, Y):
        X = [z[0] for z in Z]
        n = dict([(i,[]) for i in self.states])
        for t,s in enumerate(X):
            n[s].append(np.array([Y[t]]))
        
        for i in self.states:
            n[i] = np.array(n[i]).T
        
        for i in self.states:
            log.debug("state: %s"%i)
            log.debug("observations: %s"%n[i].round(2))
        
        
        taus, mus = [], []
        for i in self.states:
            
            S = np.cov(n[i])
            if len(n[i]):
                ybar = np.mean(n[i],1)
            else:
                #wtf? we don't have any of these observations...
                # fall back on the prior mean
                ybar = np.array(self.mu_0[i])
            log.debug("ybar[%s]: %s"%(i,ybar))
            mu_n = (
                (
                    (self.kappa/(self.kappa + len(n[i]))) * self.mu_0[i]
                ) + 
                (
                    (len(n[i])/(self.kappa + len(n[i]))) * ybar
                )
            )
            log.debug("mu_n[%s]: %s"%(i,mu_n))
            kappa_n = self.kappa + len(n[i])
            nu_n = self.nu + len(n[i])
            Lambda_n = (
                self.Lambda + 
                S + 
                (self.kappa * len(n[i]))/(self.kappa + len(n[i])) *
                (ybar - self.mu_0[i])*(ybar-self.mu_0[i]).T
            )
            try:
                sigma = invwishart(nu_n, np.linalg.inv(Lambda_n))
            except np.linalg.LinAlgError:
                sigma =  (invwishart(nu_n, 1.0/Lambda_n))
            except:
                print Lambda_n
                raise
            # form precion matrix
            tau = 1.0 / sigma
            log.debug("tau[%s]: %s"%(i,tau))
            tau_scaled = 1.0 / (sigma/kappa_n)
            log.debug("tau_scaled[%s]: %s"%(i,tau_scaled))
            mu = mvnormal(mu_n, tau_scaled)
            taus.append(tau)
            mus.append(mu)
        return mus, taus
    
    def update(self, Z, Y):
        mu, tau = self.sample_mean_prec(Z, Y)
        self.mu = mu
        self.tau = tau
    

if __name__ == "__main__":    
    import pylab as pb
    Z = np.load('Z.npy')
    Y = np.load('Y.npy')
    O = Gaussian(
        nu = 1, 
        Lambda = np.array([1]), 
        mu_0 = [-5, 0, 5], 
        kappa = 1, 
        mu = [-1,0,1], 
        tau = [0.5,0.5,0.5]
    )
    
    x = np.linspace(-4,4,100)
    for i in range(3):
        pb.plot(x,[pb.exp(O.likelihood(i,xi)) for xi in x])
    pb.show()
    
    #print Y
    #mus, sigmas = O.sample_mean_prec(Z,Y)
    #for j in range(3):
    #    mus = np.array([O.sample_mean_prec(Z,Y)[0][j] for i in range(100)]).flatten()
    #    pb.hist(mus,alpha=0.5)
    #pb.show()