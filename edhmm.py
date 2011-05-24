import numpy as np
import logging
import pymc
import copy
import time

import pprint

pp = pprint.PrettyPrinter(indent=4)

from utils import *
from emission_new import Gaussian
from duration_new import Poisson
from transition_new import Transition
from initial import *

from log_space import *

np.seterr(all='warn')
log = logging.getLogger('edhmm')

class Categorical:
    """
    Defines a Categorical Distribution
    """
    @types(p=np.ndarray)
    def __init__(self,p):
        assert all(p>=0), p
        p += 0.000000001
        assert p.sum()
        self.p = p/p.sum()
        assert self.p.sum().round(5) == 1, (p, self.p,  self.p.sum())
        self.p = np.squeeze(self.p)
        #print self.p
    
    def sample(self):
        try:
            x = np.random.multinomial(1,self.p)
        except ValueError:
            #print self.p
            #print self.p.sum()
            raise
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
    def __init__(self,A,O,D,pi):
        self.A = A
        self.O = O
        self.D = D
        self.pi = pi
        self.K = len(pi)
        self.states = range(self.K)
       
        
        
    
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
        d = self.D.sample_d(x)
        for t in range(T):
            yield x, self.O.sample_obs(x), d
            if d > 1:
                d -= 1
            else:
                xold = x
                x = self.A.sample_x(x)
                assert x in self.states, (xold,x)
                d = self.D.sample_d(x)
    
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
    
    def loglikelihood(self,Z,Y):
        l = 0
        for t in range(1,len(Z)):
            i  = Z[t-1][0]
            j  = Z[t][0]
            di = Z[t-1][1]
            y = Y[t]
            
            if i==j:
                l += self.O.likelihood(j,y)
            else:    
                l += (
                    self.A.likelihood(i,j) + 
                    self.D.likelihood(i,di) + 
                    self.O.likelihood(j,y)
                )
        return l
    
    def slice_sample(self,Z):
        log.info('forming slice')
        u = [np.log(0.0000001)]
        for t in range(1,len(Z)):
            i  = Z[t-1][0]
            j  = Z[t][0]
            di = Z[t-1][1]
            l = self.A.likelihood(i,j) + self.D.likelihood(i,di)
            u.append(np.random.uniform(low=0, high=np.exp(l)))
        return np.array(u)
        
    def get_worthy(self,u,old_worthy):
        worthy = {}
        # we only consider those transitions that are possible from 
        # t-1
        for j,dj in old_worthy:                    
            # if a transition occured...
            if dj == 1:
                # which transitions are worthy?
                for i in self.states:
                    for di in range(1, self.right[i]+1):
                        # if the probability is worthy..
                        if self.l[(i,j,di)] > u:
                            # add it to the list!
                            try:
                                worthy[(i,di)].append((j,dj))
                            except KeyError:
                                # (or start a new list)
                                worthy[(i,di)] = [(j,dj)]
            # if a transition didn't occur, then we only add the 
            # decrement i==j, di = dj-1
            else:
                i = j
                di = dj - 1
                try:
                    worthy[(i,di)].append((j,dj))
                except KeyError:
                    worthy[(i,di)] = [(j,dj)]
        return worthy
    
    def get_initial_worthy(self,u):
        worthy = {}
        for j in self.states:
            for dj in range(self.left[j],self.right[j]+1):
                if dj == 1:
                    for i in self.states:
                        for di in range(self.left[i],self.right[i]+1):
                            if self.l[(i,j,di)] > u:
                                try:
                                    worthy[(i,di)].append((j,dj))
                                except KeyError:
                                    worthy[(i,di)] = [(j,dj)]
                else:
                    i = j
                    di = dj - 1
                    try:
                        worthy[(i,di)].append((j,dj))
                    except KeyError:
                        worthy[(i,di)] = [(j,dj)]
        return worthy
    
    def worthy_transitions(self, U):
        log.info('calculating transitions worthy of u')
        # find those z_t and z_t-1 that are worthy, given u
        worthy = [None for u in U]
        
        log.debug('finding worthy durations for initial condition')
        worthy[0] = {}
        for j in self.states:
            for dj in range(self.left[j],self.right[j]+1):
                if dj == 1:
                    for i in self.states:
                        for di in range(self.left[i],self.right[i]+1):
                            if self.l[(i,j,di)] > U[0]:
                                try:
                                    worthy[0][(i,di)].append((j,dj))
                                except KeyError:
                                    worthy[0][(i,di)] = [(j,dj)]
                else:
                    i = j
                    di = dj - 1
                    try:
                        worthy[0][(i,di)].append((j,dj))
                    except KeyError:
                        worthy[0][(i,di)] = [(j,dj)]
        
        log.debug('finding worthy durations over time')
        for t,u_t in enumerate(U):
            if t > 0: 
                worthy[t] = {}
                # we only consider those transitions that are possible from 
                # t-1
                for j,dj in worthy[t-1]:                    
                    # if a transition occured...
                    if dj == 1:
                        # which transitions are worthy?
                        for i in self.states:
                            for di in range(1, self.right[i]+1):
                                # if the probability is worthy..
                                if self.l[(i,j,di)] > u_t:
                                    # add it to the list!
                                    try:
                                        worthy[t][(i,di)].append((j,dj))
                                    except KeyError:
                                        # (or start a new list)
                                        worthy[t][(i,di)] = [(j,dj)]
                    # if a transition didn't occur, then we only add the 
                    # decrement i==j, di = dj-1
                    else:
                        i = j
                        di = dj - 1
                        try:
                            worthy[t][(i,di)].append((j,dj))
                        except KeyError:
                            worthy[t][(i,di)] = [(j,dj)]
                
            assert worthy[t], (worthy[t-1], worthy[t-2], u_t)                          
                
        return worthy
            
    def beam_forward(self, Y, U, W=None):        
        
        log.info('running forward algorithm')
        
        # initialise alphahat
        log.debug('getting support')
        alphahat = [{} for y in Y]            
        
        log.debug('calculating observation likelihoods')
        ol = np.zeros((self.K,len(Y)))
        for i in self.states:
            for t,y in enumerate(Y):
                ol[i,t] = self.O.likelihood(i,y)
        
        log.debug('starting iteration')
        for t,y in enumerate(Y):
            
            log.debug('getting worthy for t: %s'%t)
            
            if W is None:
                if t == 0:
                    worthy = self.get_initial_worthy(U[t])
                else:
                    worthy = self.get_worthy(U[t],worthy)
            else:
                worthy = W[t]
                
            
            log.debug('calculating alpha[t]: %s'%t)
            
            if t == 0:
                # TODO this should be restricted
                for i in self.states:
                    alphahat[t][i] = {}
                    for d in [1]+range(self.left[i],self.right[i]+10):
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
                            alphahat[t][i[0]][i[1]] = elnsum(
                                alphahat[t][i[0]][i[1]], 
                                alphahat[t-1][j[0]][j[1]]
                            )
                        except KeyError:
                            # if a KeyError occurred, then we already decided
                            # that alphahat[t-1][state][duration] was zero, so
                            # we can just ignore it
                            #print "skipping over a key error"
                            pass
                    
                    alphahat[t][i[0]][i[1]] += ol[i[0],t]
                
        return alphahat
    
    def beam_backward_sample(self, alphahat, U=None, W=None):
        
        log.info('sampling state sequence')
        
        def sample_z(a):
            m = max(max([a[i].values() for i in a.keys()]))
            p = [np.exp(np.array(a[i].values()) - m).sum() for i in a.keys()]
            xi = Categorical(np.array(p)).sample()
            x = a.keys()[xi] 
            try:
                di = Categorical(
                    np.exp(np.array(a[x].values()) - max(a[x].values()))
                ).sample()
            except TypeError:
                di=0
            d = a[x].keys()[di]
            return x,d
        
        T = len(alphahat)
        Z = [sample_z(alphahat[-1])]
        for t in reversed(xrange(T-1)):
            # pick the subset of alphahats
            # here w[t+1][Z[-1]] is a list of the possible zs you can sample
            # from in alphahat[t] given that the next state is Z[-1], i.e.
            # w[t+1][Z[t+1]] is the next state
            
            
            #a = dict([(i,{}) for i in self.states])        
            #for j in worthy[Z[-1]]:
            #    try:
            #        a[j[0]][j[1]] = alphahat[t][j[0]][j[1]]
            #    except KeyError:
            #        a[j[0]][j[1]] = 0
            z = sample_z(alphahat[t])
            
            Z.append(z)
        Z.reverse()
        return Z
                
    def beam(self,Y, its=100, burnin=50, name=None,online=False):
        
        bored = False
        
        self.left,self.right = zip(*[self.D.support(i) for i in self.states])
        
        log.debug('calculating likelihoods')
        self.l = {}
        for i in self.states:
            for j in self.states:
                for di in range(1,max(self.right)+1):
                    self.l[(i,j,di)] = np.exp(
                        self.A.likelihood(j,i) + self.D.likelihood(i,di)
                    )
        # sample auxillary variables from some small value
        U = [np.random.uniform(0,np.log(0.0000001)) for y in Y]
        # get worthy samples given the relaxed U 
        if online:
            alpha = self.beam_forward(Y,U)
            Z_sample = self.beam_backward_sample(alpha,U)
        else:
            W = self.worthy_transitions(U)
            # get an initial state sequence
            alpha = self.beam_forward(Y, U, W=W)
            Z_sample = self.beam_backward_sample(alpha,U,W)
        # do the initial update
        self.D.update(Z_sample)
        self.O.update(Z_sample, Y)
        self.A.update(Z_sample)
        
        # count how many iterations we've done so far
        count = 0
        # storage for reporting
        As = []
        O_means = []
        O_precisions = []
        D_mus = []
        Zs = []
        L = []
        
        # block gibbs
        while not bored:
            log.debug('getting support')
            self.left,self.right = zip(*[self.D.support(i) for i in self.states])
            log.debug('calculating likelihoods')
            self.l = {}
            for i in self.states:
                for j in self.states:
                    for di in range(1,max(self.right)+1):
                        self.l[(i,j,di)] = np.exp(
                            self.A.likelihood(j,i) + self.D.likelihood(i,di)
                        )
            log.info('running sample %s'%count)
            # slice
            start = time.time()
            U = self.slice_sample(Z_sample)
            W = self.worthy_transitions(U)
            log.debug('slice sample took %ss'%(time.time() - start))
            # states
            start = time.time()
            if online:
                alpha = self.beam_forward(Y,U)
                Z_sample = self.beam_backward_sample(alpha,U)
                log.debug('inference took %ss'%(time.time() - start))
            else:
                alpha = self.beam_forward(Y, U, W=W)
                log.debug('forward pass took %ss'%(time.time() - start))
                start = time.time()
                Z_sample = self.beam_backward_sample(alpha,U,W)
                log.debug('backward sample took %ss'%(time.time() - start))
            # parameters
            self.D.update(Z_sample)
            self.O.update(Z_sample, Y)
            self.A.update(Z_sample)
            # loglikelihood
            l = self.loglikelihood(Z_sample,Y)
            L.append(l)
            log.info("log likelihood at iteration %s: %s"%(count,l))
            if count > burnin:
                As.append(self.A.A)
                O_means.append(self.O.mu)
                log.debug("O means: %s"%O_means[-1])
                O_precisions.append(self.O.tau)
                log.debug("O precisions: %s"%O_precisions[-1])
                D_mus.append(self.D.mu)
                log.debug("D rates: %s"%D_mus[-1])
                Zs.append(Z_sample)
                
                if name:
                    if not count % 20:
                        log.debug('writing samples to disk')
                        # continually overwrite so we can quit at any time
                        # this will slow things down a LOT
                        np.save("%s_As"%name,As)
                        np.save("%s_O_m"%name, O_means)
                        np.save("%s_O_p"%name, O_precisions)
                        np.save("%s_D_mus"%name, D_mus)
                        np.save("%s_Zs"%name, Zs)
                        np.save("%s_L"%name, L)

            # stop
            if count > its:
                bored = True
            count += 1
        
        As = np.array(As).squeeze()
        O_means = np.array(O_means).squeeze()
        O_precisions = np.array(O_precisions).squeeze()
        D_mus = np.array(D_mus).squeeze()
        Zs = np.array(Zs).squeeze()
        L = np.array(L).squeeze()
            
        return As, O_means, O_precisions, D_mus, Zs, L
        
if __name__ == "__main__":
    
    T = 400

    import sys
    import logging
    import pylab as pb
    import pprint

    pp = pprint.PrettyPrinter(indent=4)
    logging.basicConfig(
        stream=sys.stdout,
        #filename="EDHMM.log", 
        #filemode="w",
        level=logging.DEBUG
    )

    A = Transition(
        K=3,
        A=pb.array([[0, 0.3, 0.7], [0.6, 0, 0.4], [0.3, 0.7, 0]])
    )
    O = Gaussian(
        nu = 1, 
        Lambda = np.array([1]), 
        mu_0 = [-10, 0, 10], 
        kappa = 1, 
        mu = [-10, 0, 10], 
        tau = [np.array([[1]]),np.array([[1]]),np.array([[1]])]
    )
    D = Poisson(mu = [3,5,10], alpha=[3,3,3], beta=[4,4,4])
    pi = Initial(K=3,beta=0.001)
    m = EDHMM(A,O,D,pi)
    X,Y,Dseq = m.sim(T)
    m.beam(Y)
    