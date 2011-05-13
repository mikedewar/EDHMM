import numpy as np
import logging
import pymc
import scipy.cluster.vq

import pprint

pp = pprint.PrettyPrinter(indent=4)

from utils import *
from emission import *
from duration import *
from transition import *
from initial import *

np.seterr(all='warn')
log = logging.getLogger('edhmm')

class Categorical:
    """
    Defines a Categorical Distribution
    """
    @types(p=np.ndarray)
    def __init__(self,p):
        p += 0.000000001
        self.p = p/p.sum()
    
    def sample(self):
        x = np.random.multinomial(1,self.p)
        return np.where(x==1)[0][0]


class EDHMM:
    """
    Defines an Explicit Duration Hidden Markov Model
    
    Parameters
    ----------
    A : Transition Object
        The transition distribution
    O : Emission Object
        The emission distribution
    D : Duration object
        The duration distribution
    pi : Initial object
        The initial distribution
    
    Attributes
    ----------
    K : int
        The number of states in the system
    states : list
        list of states
    durations : list
        list of possible durations
    """
    @types(A=Transition, O=Emission, D=Duration, pi=Initial)
    def __init__(self,A,O,D,pi):
        self.A = A
        self.O = O
        self.D = D
        self.pi = pi
        self.K = len(pi)
        self.states = range(self.K)
        self.durations = range(1,self.get_max_duration())
        assert all(np.array([len(A), len(O), len(D), len(pi)])==self.K)
        log.info('initialised EDHMM:')
        self.report()
    
    def get_max_duration(self):
        """
        Finds the maximum duration by sampling lots from all the duration 
        distributions
        """
        return int(max([
            self.D(state=q)
            for i in range(1000) 
            for q in self.states
        ]))
    
    def report(self):
        self.A.report()
        self.O.report()
        self.pi.report()
        self.D.report()
    
    @types(T=int)
    def gen(self,T):
        """
        generator that yields state/observation tuples
        
        See Also
        --------
        see EDHMM.sim for more details
        """
        # draw initial state and duration
        x,d = self.pi.sample()
        d = self.D.sample(x)
        for t in range(T):
            yield x, self.O.sample(x), d
            if d > 1:
                d -= 1
            else:
                xold = x
                x = self.A.sample(x)
                assert x in self.states, (xold,x)
                d = self.D.sample(x)
    
    @types(T=int)
    def sim(self,T):
        """
        Draws a sequence of length T from the EDHMM
        
        Parameters
        ----------
        T : int
            number of time points
        """
        X, Y, D = [], [], []
        for x,y,d in self.gen(T):
            X.append(x)
            Y.append(y)
            D.append(d)
        return X, Y, D
    
    def slice_sample(self,Z):
        log.info('forming slice')
        u = [0.0001]
        for t in range(1,len(Z)):
            i = Z[t-1][0]
            j = Z[t][0]
            di  = Z[t-1][1]
            l = self.A[j,i] * pb.exp(self.D(i,di))
            u.append(np.random.uniform(low=0, high=l))
            #assert u[-1] < l, (l,u,self.A[j,i])
        return np.array(u)
    
    def worthy_transitions(self, U):
        log.info('calculating transitions worthy of u')
        # find those z_t and z_t-1 that are worthy, given u
        l,r = zip(*[self.D.support(i) for i in self.states])
        worthy = [None for u in U]
        for t,u_t in enumerate(U):
            #print "u_t: %s"%u_t
            worthy[t] = {}
            for i in self.states:
                for di in range(1,r[i]+1):
                    for j in self.states:
                        for dj in range(1,r[j]+1):
                            # the restriction below enforces consistency between pairs:
                            # we only consider those transitions that are possible from 
                            # t-1 
                            if t > 0: 
                                
                                if (j,dj) in worthy[t-1]: 
                                    # so for every possible state duration pair, we 
                                    # calculate the probability `l` of transition from the
                                    # previous state (j,dj) to the current state (i,di).
                                    if dj == 1:
                                        l = self.A[j,i] * pb.exp(self.D(i,di))
                                    else:
                                        if i==j and dj != 1 and di==dj-1:
                                            l = 1
                                        else:
                                            l = 0
                                    # if the probaility `l` is greater than u_t then we 
                                    # add that pair to our dictionary of worthy. The 
                                    # dictionary's keys index the alphas we will
                                    # calculate at time t, and the values of the 
                                    # dictionary are the indices into alpha at time t-1
                                    # that we need to sum over.
                            
                                    if l > u_t:
                                        try:
                                            worthy[t][(i,di)].append((j,dj))
                                        except KeyError:
                                            worthy[t][(i,di)] = [(j,dj)]
                            else:
                                if dj == 1:
                                    l = self.A[j,i] * np.exp(self.D(i,di))
                                    #print "l at the transition: %s"%l 
                                else:
                                    if i==j and dj != 1 and di==dj-1:
                                        l = 1
                                    else:
                                        l = 0
                                # if the probaility `l` is greater than u_t then we 
                                # add that pair to our dictionary of worthy. The 
                                # dictionary's keys index the alphas we will
                                # calculate at time t, and the values of the 
                                # dictionary are the indices into alpha at time t-1
                                # that we need to sum over.
                        
                                if l > u_t:
                                    try:
                                        worthy[t][(i,di)].append((j,dj))
                                    except KeyError:
                                        worthy[t][(i,di)] = [(j,dj)]
            try:
                assert worthy[t], (worthy[t-1], worthy[t-2], u_t)                          
            except AssertionError:
                for i in self.states:
                    for j in self.states:
                        for di in range(1,r[i]+1):
                            l = self.A[j,i] * np.exp(self.D(i,di)).round(3)
                            print "%s, %s, %s : %s"%(i,j,di,l)
                raise
                
        return worthy
            
    def beam_forward(self, Y, U=None, W=None):        
        # initialise alphahat
        alphahat = [{} for y in Y]
        l,r = zip(*[self.D.support(i) for i in self.states])
        
        if W is None:
            W = self.worthy_transitions(U)
               
        for t,(y,worthy) in enumerate(zip(Y,W)):
            if t == 0:
                # TODO this should be restricted
                for i in self.states:
                    alphahat[t][i] = {}
                    for d in range(1,r[i]+10):
                        alphahat[t][i][d] = self.pi.likelihood((i,d))
            else:
                for i,J in worthy.items():
                    
                    # initialise alpahat[t] if necessary
                    if i[0] not in alphahat[t]:
                        alphahat[t][i[0]] = {i[1]:0}
                    else:
                        if i[1] not in alphahat[t][i[0]]:
                            alphahat[t][i[0]][i[1]] = 0
                    
                    # here i is those (state,duration)s worth figuring out for 
                    # alpha hat. Then J is a list of those indices into the 
                    # previous alpha hat we should sum over to find the next 
                    # alpha hat.
                
                    # so you can read this indexing as 
                    # alphahat[time][state][duration]
                                    
                    for j in J:
                        try:
                            alphahat[t][i[0]][i[1]] += alphahat[t-1][j[0]][j[1]]
                        except KeyError:
                            # if a KeyError occurred, then we already decided
                            # that alphahat[t-1][state][duration] was zero, so
                            # we can just ignore it
                            pass
                    
                    alphahat[t][i[0]][i[1]] *= self.O(i[0],y)

                
                # find sum(alpha[t])
                n = 0
                for i in alphahat[t]:
                    for v in alphahat[t][i].values():
                        n += v
                        
                assert n, (n, W[:t+4])
                # normalise
                for i in alphahat[t]:
                    for d in alphahat[t][i].keys():
                        alphahat[t][i][d] /= n
        return alphahat
    
    def beam_backward_sample(self, alphahat, W):
        
        def sample_z(a):
            xi = Categorical(
                np.array([sum(a[i].values()) for i in a.keys()])
            ).sample()
            x = a.keys()[xi]
            di = Categorical(
                np.array(a[x].values())
            ).sample()
            d = a[x].keys()[di]
            return x,d
        
        T = len(alphahat)
        try:
            Z = [sample_z(alphahat[-1])]
        except IndexError:
            #pp.pprint(alphahat)
            #print len(alphahat)
            raise
        
        for t in reversed(xrange(T-1)):
            # pick the subset of alphahats
            # here w[t+1][Z[-1]] is a list of the possible zs you can sample
            # from in alphahat[t] given that the next state is Z[-1]
            a = {}            
            
            for j in W[t+1][Z[-1]]:
                
                a[j[0]] = {}
                try:
                    a[j[0]][j[1]] = alphahat[t][j[0]][j[1]]
                except KeyError:
                    a[j[0]][j[1]] = 0
                z = sample_z(a)
            
            Z.append(z)
        Z.reverse()
        return Z
                
    def beam(self,Y):
        U = [np.random.uniform(0,0.0000001) for y in Y]
        bored = False
        # run once to get samples 
        W = self.worthy_transitions(U)
        alpha = self.beam_forward(Y, W=W)
        assert len(alpha) == len(Y)
        Z = self.beam_backward_sample(alpha,W)
        assert len(Z) == len(Y)
        D_sample = self.sample_D(Z)
        A_sample = self.sample_A(Z)
        O_sample = self.sample_O(Z,Y)
        count = 0
        
        A = []
        D = []
        O = []
        
        while not bored:
            
            self.A.update_new(A_sample)
            self.D.update_new(D_sample)
            self.O.update_new(O_sample)
            
            U = self.slice_sample(Z)
            assert len(U) == len(Y)
            W = self.worthy_transitions(U)
            try:
                alpha = self.beam_forward(Y, W=W)
            except:
                print "Normalisation failed in alpha. Rejecting this sample."
                pass
            try:
                Z = self.beam_backward_sample(alpha,W)
            except KeyError:
                print "Tried to sample an impossible state sequence. Rejecting this sample."
                pass
            
            D_sample = self.sample_D(Z)
            A_sample = self.sample_A(Z)
            O_sample = self.sample_O(Z,Y)
            
            count +=1
            if count > 100:
                bored = True
            
            A.append(A_sample)
            D.append(D_sample)
            O.append(O_sample)
            
        
        return A, D, O
    
    def sample_D(self,Z):
        log.info('sampling from D')
        # let's count the durations
        X = [z[0] for z in Z]        
        durations = dict([(i,[0]) for i in self.states])
        now = X[0]
        for s in X:
            if now == s:
                durations[now][-1] += 1
            else:
                now = s
                durations[now].append(1)
        # build a little pymc model (this should probably live in the duration
        # class)
        mu = [
            pymc.Gamma("mu_%s"%i, alpha=1, beta=1)
            for i in self.states
        ]
        D = [
            pymc.Poisson("duration_%s"%i, mu[i], value=durations[i], observed=True) 
            for i in self.states
        ]
        M = pymc.MCMC(pymc.Model({'D':D, 'mu':mu}))
        M.sample(iter=101,burn=100)
        samples = np.array([M.trace('mu_%s'%i)[:] for i in self.states])
        return samples.flatten()
    
    def sample_O(self, Z, Y):
        log.info('sampling from O')
        # lets gather all the observations associated with each state
        X = [z[0] for z in Z]
        n = [[] for i in self.states]
        now = X[0]
        for t,s in enumerate(X):
            n[now].append(np.array([Y[t]]))
            if now != s:
                now = s
        
        mu_prior = [
            pymc.Normal('mu_%s'%i, 0, 10)
            for i in self.states
        ]
        cov_prior = [
            pymc.InverseWishart(
                'sigma_%s'%i,n = self.O.dim, 
                Tau = np.eye(self.O.dim)
            )
            for i in self.states
        ]
        O = [
            pymc.MvNormalCov(
                'emission_%s'%j, 
                mu_prior[j], 
                cov_prior[j], 
                value=n[j], 
                observed=True
            ) 
            for j in self.states
        ]
        model = pymc.Model({"sigma":cov_prior, "mu":mu_prior, "O":O})
        M=pymc.MCMC(model)
        M.sample(iter=101,burn=100)
        mu_sample = np.array([M.trace('mu_%s'%i)[:] for i in self.states])
        sigma_sample = np.array([M.trace('sigma_%s'%i)[:] for i in self.states])
        return mu_sample, sigma_sample
        
    
    def sample_A(self, Z):
        log.info('sampling from A')
        # let's count the transitions
        X = [z[0] for z in Z]
        n = [[] for i in self.states]
        now = X[0]
        for s in X:
            if now != s:
                n[now].append(s)
                now = s
                
        # form prior variables
        p = np.zeros((self.K, self.K))
        for i in self.states:
            for j in self.states:
                if i!=j:
                    p[i,j] = 1.0/(self.K-1)
        # add a little bit to everywhere and renormalise
        p+=0.05
        for row in p:
            row /= row.sum()
        
        A_prior = [
            pymc.Dirichlet("p_%s"%i, p[i])
            for i in self.states
        ]
        A = [
            pymc.Categorical('A_%s'%i, A_prior[i], value=n[i], observed=True)
            for i in self.states
        ]
        model = pymc.Model({"A_prior":A_prior, "A":A})
        M=pymc.MCMC(model)
        M.sample(iter=101,burn=100)
        samples = np.array([M.trace('p_%s'%i)[:] for i in self.states])
        return samples.reshape((self.K, self.K-1))
        
if __name__ == "__main__":
    import sys
    import logging
    import pylab as pb
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG
    )
        
    A = Transition(np.array([[0, 0.3, 0.7], [0.6, 0, 0.4], [0.2, 0.8, 0]]))
    O = Gaussian([-1,0,1],[1,1,1])
    D = Poisson([10,20,30])
    pi = Initial(np.array([0.33, 0.33, 0.34]))
    m = EDHMM(A,O,D,pi)
    X,Y,D = m.sim(1000)
    
    l, m_est = baum_welch(Y,K=3)
    
    #S = m.beam(Y)
    
    #alpha, beta, alpha_smooth = m.forward_backward(Y)
    #S = [s for s in m.backward_sample(alpha)]
    
#   pb.plot(l)
#   pb.show()
