import numpy as np
import logging
import time
import cPickle

import pprint

pp = pprint.PrettyPrinter(indent=4)

from log_space import elnsum

np.seterr(all='warn')
log = logging.getLogger('edhmm')

class Categorical:
    """
    Defines a Categorical Distribution
    """
    def __init__(self,p):
        assert all(p>=0), p
        assert not any(np.isinf(p))
        p += 0.000000001
        assert p.sum()
        self.p = p/p.sum()
        assert self.p.sum().round(5) == 1, (p, self.p,  self.p.sum())
        self.p = np.squeeze(self.p)
    
    def sample(self):
        if self.p.shape == ():
            return 0
        try:
            x = np.random.multinomial(1,self.p)
        except ValueError:
            #print self.p
            #print self.p.sum()
            raise
        except TypeError:
            print self.p
            print type(self.p)
            print self.p.shape
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
    
    def loglikelihood(self,Zs,Ys):
        """
        Calculates the log likelihood of the model given state
        and observation sequences
        
        Parameters
        ----------
        Zs : list
            list of state sequences
        Ys : list
            list of observation sequences
        """
        l = 0
        for Z,Y in zip(Zs,Ys):
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
    
    def set_transition_likelihood(self):
        """
        builds a dictionary of possible transitions and their likelihoods
        """
        
        if hasattr(self,'l'):
            old_l = self.l
            old_durations = np.array([k[2:] for k in old_l]).flatten()
            right = max(old_durations.max(), max(self.right))
        else:
            right = max(self.right)
        
        log.info('forming transition likelihoods from d = 1 to %s'%right)
        
        self.l = {}
        for i in self.states:
            for j in self.states:
                for di in range(1,right+1):
                    for dj in range(1,right+1):
                        if di == 1:
                            self.l[(i,j,di,dj)] = np.exp(
                                self.A.likelihood(i,j) + self.D.likelihood(j,dj)
                            )
                        elif i == j and dj==di-1:
                            self.l[(i,j,di,dj)] = 1
                        else:
                            self.l[(i,j,di,dj)] = 0
                        
    
    def set_duration_support(self):
        self.left,self.right = zip(*[self.D.support(i) for i in self.states])
        
    def slice_sample(self,Z, min_u=0):
        log.info('forming slice')
        u = [min_u]
                
        for t in range(1,len(Z)):
            i  = Z[t-1][0]
            j  = Z[t][0]
            di = Z[t-1][1]
            dj = Z[t][1]
            try:
                u.append(
                    np.random.uniform(
                        low=min_u, 
                        high=self.l[(i,j,di,dj)])
                    )
            except KeyError:
                raise
        return np.array(u)
        
    def get_worthy(self,u,old_worthy):
        """
        decides which transitions are valid given the auxilliary variable
        """
        worthy = {}
        # we only consider those transitions that are possible from 
        # t-1
        for i,di in old_worthy:                    
            # which transitions are worthy?
            if di == 1:
                for j in self.states:
                    for dj in range(1, self.right[i]+1):
                        # if the probability is worthy..
                        if self.l[(i,j,di,dj)] > u:
                            # add it to the list!
                            try:
                                worthy[(j,dj)].append((i,di))
                            except KeyError:
                                # (or start a new list)
                                worthy[(j,dj)] = [(i,di)]
            else:    
                j=i
                dj = di-1
                try:
                    worthy[(j,dj)].append((i,di))
                except KeyError:
                    worthy[(j,dj)] = [(i,di)]
        
        assert worthy, (u, old_worthy)
        return worthy
    
    def get_initial_worthy(self,u):
        """
        gets the intial set of worthy states
        """
        worthy = {}
        for i in self.states:
            for di in range(self.left[i],self.right[i]+1):
                for j in self.states:
                    for dj in range(self.left[j],self.right[j]+1):
                        if self.l[(i,j,di,dj)] > u:
                            try:
                                worthy[(j,dj)].append((i,di))
                            except KeyError:
                                worthy[(j,dj)] = [(i,di)]
        return worthy
            
    def beam_forward(self, Y, U, W=None):   
        """
        runs the forwrd algorithm, sampling only from valid transitions
        """     
        
        log.info('running forward algorithm')
        
        # initialise alphahat
        alphahat = [{} for y in Y]            
        
        log.debug('calculating observation likelihoods')
        ol = np.zeros((self.K,len(Y)))
        for i in self.states:
            for t,y in enumerate(Y):
                ol[i,t] = self.O.likelihood(i,y)
            
        log.debug('starting iteration')

        worthy_time = 0
        alpha_time = 0

        for t,y in enumerate(Y):
            start = time.time() 
            if W is None:
                if t == 0:
                    worthy = self.get_initial_worthy(U[t])
                else:
                    worthy = self.get_worthy(U[t],worthy)
            else:
                worthy = W[t]
            worthy_time += time.time() - start
                        
            start = time.time() 
            if t == 0:
                for i in self.states:
                    alphahat[t][i] = {}
                    for d in [1]+range(self.left[i],self.right[i]+1):
                        alphahat[t][i][d] = self.pi.likelihood((i,d))
            
            else:
                for i,J in worthy.items():
                    # initialise alpahat[t] if necessary
                    if i[0] not in alphahat[t]:
                        alphahat[t][i[0]] = {i[1]:-1000000000000}
                    else:
                        if i[1] not in alphahat[t][i[0]]:
                            alphahat[t][i[0]][i[1]] = -1000000000000
                    
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
                    #print "alpha[%s][%s][%s] = %s"%(t,i[0],i[1], alphahat[t][i[0]][i[1]])
                    assert not np.isinf(alphahat[t][i[0]][i[1]])
            
            try:
                assert alphahat[t], "alpha[%s]:%s"%(t,alphahat[t])
            except AssertionError:
                print "alpha[%s]:%s"%(t-1,alphahat[t-1])
                print worthy
                raise
            if hasattr(self,'W_fh'):
                num_transitions = sum([len(worthy[key]) for key in worthy])
                cPickle.dump(num_transitions, self.W_fh)
            alpha_time += time.time() - start
                

        log.debug('time spent building alpha: %s'%alpha_time)
        log.debug('time spent finding worthy: %s'%worthy_time)
        return alphahat
    
    def beam_backward_sample(self, alphahat, U, W=None):
        """
        perfomrs the backwards sweep given the forwards sweep and the auxilliary variables
        """
        
        log.info('backward sampling state sequence')
        
        def sample_z(a):
            vals = []
            for i in a:
                vals.extend(a[i].values())
            try:
                m = np.array(vals).max()
            except ValueError:
                print a
                print vals
                raise
            p = [np.exp(np.array(a[i].values()) - m).sum() for i in a.keys()]
            
            xi = Categorical(np.array(p)).sample()
            x = a.keys()[xi] 
            try:
                p = np.exp(np.array(a[x].values()) - max(a[x].values()))
                di = Categorical(p).sample()
            except AssertionError:
                print p
                raise
            d = a[x].keys()[di]
            return x,d
        
        T = len(alphahat)
        try:
            Z = [sample_z(alphahat[-1])]
        except ValueError:
            print alphahat[-1]
            raise
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
            
            # we need to build up a pair of worthys
            
            # first, the get_worthy method uses old_worthy to make sure that 
            # the transitions are consistent. So we need just the keys in
            # alphahat as we know that this is 'old worthy' for the worthy
            # variables at t+1
            old_worthy = {}
            for state in alphahat[t]:
                for duration in alphahat[t][state]:
                    key = (state, duration)
                    old_worthy[key] = 0
            
            worthy = self.get_worthy(U[t+1],old_worthy)
            
            a = dict([(i,{}) for i in self.states])
            try:
                worthy[Z[-1]]
            except KeyError:
                print worthy
                raise
            
            for j in worthy[Z[-1]]:
                try:
                    a[j[0]][j[1]] = alphahat[t][j[0]][j[1]]
                except KeyError:
                    a[j[0]][j[1]] = -10000000
            z = sample_z(a)
            Z.append(z)
        Z.reverse()
        return Z
                
    def beam(self, 
        Y, min_u=0, its=100, burnin=50, name='beamer', online=True, sample_U=True, update_D=True, force_U = None
        ):
        """
        Runs the beam sampling approach for the EDHMM
        
        Parameters
        ----------
        Y : list
            list of observation sequences
            
        Optional Parameters
        -------------------
        min_u : scalar (0)
            the minimum auxilliary variable to consider. You can use this to tune
            the algorithm. See the paper for details.
        its : integer (100)
            number of iterations to perform
        burning : integer (50)
            allow this many iterations before writing samples to disk
        name : string (beamer)
            name of the experiment. This will be prepended to all output files
        online : boolean (True)
            whether or not to run the algo online. Currently this has to be True
        sample_U : boolean (True)
            whether or not to update the auxilliary variable. You probably should leave
            this to True, unless you're poking at the algorithm to see what it does
        updated_D : boolean (True)
            whether or not to update the duration distribution. Again, you should leave
            this to True.
        force_U : None or list of lists (None)
            you can force U to start off from a specific starting place if you like. If 
            you set this and set sample_U to False then this auxilliary variable won't 
            change throughout the algo.
        """
        bored = False
        
        # get support of duration distributions
        self.set_duration_support()
        self.set_transition_likelihood()
        
        # sample auxillary variables from some small value
        U = [[np.random.uniform(min_u, 0.000000000001) for y in Yi] for Yi in Y]
        
        # get worthy samples given the relaxed U 
        alphas = []
        Z_samples = []
        
        log.debug('perfomring inference')
        if online:
            for i, Yi in enumerate(Y):
                alphas.append(self.beam_forward(Yi,U[i]))
                Z_samples.append(self.beam_backward_sample(alphas[i],U[i]))
        else:
            raise NotImplementedError
        
        # count how many iterations we've done so far
        count = 0
        # storage for reporting
        A_fh = open('A_%s.dat'%name,'w')
        O_m_fh = open('O_m_%s.dat'%name,'w')
        O_p_fh = open('O_p_%s.dat'%name,'w')
        D_m_fh = open('D_m_%s.dat'%name,'w')
        Z_fh = open('Z_%s.dat'%name,'w')
        L_fh = open('L_%s.dat'%name,'w')
        self.W_fh = open('W_%s.dat'%name,'w')
        
        L = []
        
        # block gibbs
        while not bored:
            log.info('running sample %s'%count)
            
            log.debug('getting support')
            self.left,self.right = zip(*[self.D.support(i) for i in self.states])
            self.set_transition_likelihood()
            
            # slice
            start = time.time()
            if sample_U:
                U = []
                for Z in Z_samples:
                    U.append(self.slice_sample(Z,min_u))
            
            log.debug('slice sample took %ss'%(time.time() - start))
            
            # states
            start = time.time()
            alphas = []
            Z_samples = []
            if online:
                for i, Yi in enumerate(Y):
                    alphas.append(self.beam_forward(Yi,U[i]))
                    Z_samples.append(self.beam_backward_sample(alphas[i],U[i]))
                log.debug('inference took %ss'%(time.time() - start))
            else:
                raise NotImplementedError
                #for i, Yi in enumerate(Y):
                #    W = self.worthy_transitions(U[i])
                #    get an initial state sequence
                #    alphas.append(self.beam_forward(Yi, U[i], W=W))
                #    Z_samples.append(self.beam_backward_sample(alphas[i],U[i],W))
                #log.debug('inference took %ss'%(time.time() - start))
            # parameters
            if update_D:
                self.D.update(Z_samples)
            self.O.update(Z_samples, Y)
            self.A.update(Z_samples)
            # loglikelihood
            l = self.loglikelihood(Z_samples, Y)
            L.append(l)
            log.info("log likelihood at iteration %s: %s"%(count,l))
                        
            if count > burnin:
                if count % 5 == 0:
                    log.debug('writing iteration %s to disk'%count)
                    # start writing to disk                
                    cPickle.dump(self.A.A, A_fh )
                    cPickle.dump(self.O.mu, O_m_fh )
                    cPickle.dump(self.O.tau, O_p_fh )
                    cPickle.dump(self.D.mu, D_m_fh )
                    cPickle.dump(Z_samples, Z_fh)
                    cPickle.dump(l, L_fh)
            # stop
            if count > its:
                bored = True
            count += 1
        
        return L
