import numpy as np
np.seterr(all='raise')
import logging
import pymc

from utils import *
from emission import *
from duration import *

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
        try:
            return np.where(x==1)[0][0]
        except IndexError:
            print x
            raise


class Transition:
    """
    Defines a Transition distribution as collection of categorical distributions
    """
    @types(A=np.ndarray)
    def __init__(self, A):
        assert A.shape[0] == A.shape[1], \
            "non-square transition matrix"
        assert all(np.diag(A)==0), \
            "the EDHMM does not allow self transitions"
        for row in A:
            assert sum(row)==1, \
                "invalid transition matrix"
        self.A = A
        self.shape = A.shape
        self.dist = [
            pymc.Categorical("transition from %s"%i, a)
            for i, a in enumerate(A)
        ]
    
    def __getitem__(self,key):
        return self.A[key]
    
    def __len__(self):
        return self.A.shape[0]
    
    def sample(self,i):
        return int(self.dist[i].random())
        

class Initial:
    """
    Defines an Initial distribution
    """
    @types(pi=np.ndarray)
    def __init__(self,pi):
        assert sum(pi)==1, \
            "invalid initial distribution"
        self.pi = pi
        self.pi.shape = (len(self.pi),1)
        self.shape = pi.shape
        self.dist = pymc.Categorical("initial distribution",pi)
    
    def __getitem__(self,key):
        return self.pi[key]
    
    def __len__(self):
        return self.pi.shape[0]
    
    def sample(self):
        return int(self.dist.random())

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
        log.info("number of states: %s"%self.K)
        log.info("maximum duration: %s"%self.durations[-1])
        log.info("number of features: %s"%self.O.dim)
    
    @types(T=int)
    def gen(self,T):
        """
        generator that yields state/observation tuples
        
        See Also
        --------
        see EDHMM.sim for more details
        """
        # draw initial state
        x = self.pi.sample()
        # draw initial distribution
        d = self.D.sample(x)
        for t in range(T):
            yield x, self.O.sample(x)
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
        X, Y = [], []
        for x,y in self.gen(T):
            X.append(x)
            Y.append(y)
        return X, Y
    
    def duration_likelihood(self,durations=None):
        """
        Calculates the duration likelihood per state per duration
        
        Parameters
        ----------
        durations : list, optional
            durations over which to calulate likelihoods
        
        Returns
        -------
        D : np.ndarray
            K x d array, where the i,j th element is the likleihood of the 
            duration j in given that we're in state i. Here d is the length
            of the durations list.
            
        Notes
        -----
        If you specify a durations list, then the resulting matrix will be the
        likelihood of each duration in the list. Otherwise self.durations will
        be used. 
        """
        log.info('calculating duration likelihood')
        if durations is None:
            durations = self.durations
        return np.array(
            [[self.D(i,di) for di in durations] for i in self.states]
        )
    
    def forward(self,Y,D=None):
        """
        forward algorithm as specified by Yu and Kobayashi 2006.
        
        Paramters
        ---------
        Y : list of np.ndarray
            data
        D : np.ndarray, optional
            duration likelihoods
        Returns
        -------
        alpha : list of np.ndarray
            forward variable, defined here as p(x_t | y_1 .. y_t)
        bstar : list of np.ndarray
            p(y_t|x_t) / p(y_t|y_1 .. y_t-1)
            
        Notes
        -----
        Note that this forward variable is not p(x_t, y_1 .. y_t) which is 
        usually calculated in the forward variable. Note also that the variable
        calculated here is properly normalised, and hence suffers no scaling
        issues.  
        """
        log.info('running forward algorithm')
        T = len(Y)
        alpha = [np.zeros((self.K, len(self.durations))) for y in Y]
        bstar = [np.zeros((self.K,1)) for y in Y]
        
        E = np.zeros((self.K,1))
        U = np.zeros((self.K,1))
        
        if D is None:
            D = self.duration_likelihood()
        
        for t in range(T):
            if t == 0:
                alpha[t] = self.pi.pi * D
            else:
                # alpha shifted is to vectorise the calculation below
                # we need alpha[t-1][:,d_index+1]), so shift the whole 
                # alpha[t-1] matrix to the left, discarding alpha[t-1][:,0]
                # and then padding with a column of zeros to take care of
                # the d = self.durations[-1] case
                alpha_shifted = np.hstack([alpha[t-1][:, 1:], np.zeros((self.K, 1))])
                alpha[t] = (S * D) + (bstar[t-1] * alpha_shifted)
            
            # let's just re-normalise to avoid propagating numerical errors
            assert isprob(alpha[t]), "forward variable should be a valid probability"
            alpha[t] = alpha[t] / alpha[t].sum()
            
            # U = p(y_t|x_t)
            U[:,0] = np.array([self.O(i,Y[t]) for i in self.states])
            # rinv = \sum_mn alpha_t(m,d) p(y_t|x_t=m)
            rinv = (alpha[t] * U).sum()
            # bstar = p(y_t|x_t=m) / p(y_t|y_1 .. y_t-1)
            bstar[t] = U / rinv
            # E = p(x_t, d_t=1 | y_1 ... y_t)
            E[:,0] = alpha[t][:,0] * bstar[t][:,0]
            # S = p(x_t+1, d_t=1 | y_1 ... y_t)
            S = np.dot(self.A.A.T,E)
        return alpha, bstar
    
    def backward(self,Y,bstar, D=None):
        """
        backward algorithm as specified by Yu and Kobayashi 2006.
        
        Paramters
        ---------
        Y : list of np.ndarray
            data
        bstar : list of np.ndarray
            p(y_t|x_t) / p(y_t|y_1 .. y_t-1)
        D : np.ndarray, optional
            duration likelihoods
        
        Returns
        -------
        beta : list of np.ndarray
            backward variable
            
        Notes
        -----
        Note that this backward variable is not p(y_t+1 .. y_T|x_t) which is 
        usually calculated in the backward algorithm. 
        """
        log.info('running backward algorithm')
        T = len(Y)
        beta = [np.zeros((self.K, len(self.durations))) for y in Y]
        if D is None:
            D = self.duration_likelihood()
        for t in reversed(xrange(T)):
            if t == T-1:
                for d_index, d in enumerate(self.durations):
                    beta[t][:,d_index] = bstar[t][:,0]
            else:
                for d_index, d in enumerate(self.durations):
                    if d == 1:
                        beta[t][:,d_index] = bstar[t][:,0] * Sstar[:,0]
                    else:
                        beta[t][:,d_index] = beta[t+1][:, d_index-1] * bstar[t][:,0]
            Estar = (D * beta[t]).sum(1)
            Sstar = np.zeros((self.K,1))
            for j in self.states:
                for i in self.states:
                    Sstar[j,0] += Estar[i] * self.A.A[j,i]
        return beta
    
    @types(Y=list)
    def forward_backward(self,Y,D=None):
        if D is None:
            D = self.duration_likelihood()
        alpha, bstar = m.forward(Y,D)
        beta = m.backward(Y,bstar,D)
        alpha_smooth = [f*b for f,b in zip(alpha,beta)]
        assert all([a.sum() for a in alpha_smooth])
        return alpha, beta, alpha_smooth
    
    def backward_sample(self, alpha):
        T = len(alpha)
        s = Categorical(alpha[-1].sum(1)).sample()
        d_index = Categorical(alpha[-1][s,:]).sample()
        d = self.durations[d_index]
        for t in reversed(xrange(T-1)):
            yield s
            if d > 1:
                d -= 1
            else:
                s = Categorical(alpha[t].sum(1)*self.A.A[:,s]).sample()
                d = Categorical(alpha[t][s,:]).sample() + 1 # note the +1!
        yield s
    
    def slice_sample(self,S):
        log.info('forming slice')
        u = []
        for s in S:
            d = self.D(s)
            y = self.D(s,d)
            u.append(np.random.uniform(low=0, high=y))
            assert u[-1] < y
        return np.array(u)
    
    def beam_forward(self, Y, u=None):
        
        log.info('running forward algorithm w/ slice sampling')
        
        T = len(Y)
        E = np.zeros((self.K,1))
        U = np.zeros((self.K,1))
        
        if u is None:
            S, Z = self.sim(len(Y)) # ignore Z
            u = self.slice_sample(S)
                
        assert len(u) == T, len(u)
        
        #### <HACK>
        log.debug('finding max duration for current iteration')
        max_d = 1
        for i in self.states:
            # TODO this won't work for non-Poisson distributions
            # better ways to choose a starting d for this little search?
            d = self.D.dist[i].parents['mu']
            while self.D(i,d) > min(u):
                d += 1
            max_d = max(d,max_d)
        durations = range(1,max_d+10)
        log.debug('found max duration: %s'%(max_d+10))
        #### <\HACK>         
        
        alpha = [np.zeros((self.K, len(durations))) for y in Y]
        bstar = [np.zeros(self.K) for y in Y]
        
        D0 = self.duration_likelihood(durations)
        for t,ut in enumerate(u):
            assert ut < D0.max(), (t,ut,D0.max())
        
        log.debug('starting iteration')
        for t in range(T):
            # D is repeatedly pruned depending on the value of u[t]
            # this sets D with likelihoods greater than the threshold to zero
            # the alternative and probably better way to do this is to keep
            # evaluating the likelihood until it drops below some value. Not
            # sure though...
            D = D0 * (D0 > u[t])
            if t == 0:
                alpha[t] = self.pi.pi * D
            else:
                alpha_shifted = np.hstack([alpha[t-1][:, 1:], np.zeros((self.K, 1))])
                alpha[t] = (S * D) + (bstar[t-1] * alpha_shifted)
            
            U[:,0] = np.array([self.O(i,Y[t]) for i in self.states])
            rinv = (alpha[t] * U).sum()
            bstar[t] = U / rinv
            E[:,0] = alpha[t][:,0] * bstar[t][:,0]
            S = np.dot(self.A.A.T,E)
        
            alpha[t] = alpha[t] / alpha[t].sum()
        return alpha, bstar
        
    def beam(self, Y, its = 100, burn_in=20):
        S, Z = self.sim(len(Y)) # ignore Z
        Sout = []
        for i in range(its):
            log.info("beam iteration: %s of %s"%(i,its))
            u = self.slice_sample(S)
            alpha, bstar = self.beam_forward(Y,u)
            S = [s for s in m.backward_sample(alpha)]
            if i > burn_in:
                Sout.append(S)
        return Sout
        
    def learn(self):
        pass
    
    @types(Y=list)
    def BaumWelch(self,Y):
        pass
    
@types(Y=list, K=int)
def initialise_EDHMM(self,Y,K):
    pass

@types(models=list)
def average_models(self,models):
    pass

    
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
    D = Poisson([10,20,100])
    pi = Initial(np.array([0.33, 0.33, 0.34]))
    m = EDHMM(A,O,D,pi)
    X,Y = m.sim(1000)
    
    S = m.beam(Y,burn_in=90)
    
    #alpha, beta, alpha_smooth = m.forward_backward(Y)
    #S = [s for s in m.backward_sample(alpha)]
    
    pb.plot(X)
    for Si in S:
        pb.plot(S)
    pb.ylim([-0.1,2.1])
    pb.show()
    
    
    
    
    