import numpy as np
import logging
import pymc
import scipy.cluster.vq

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
        try:
            return np.where(x==1)[0][0]
        except IndexError:
            print x
            raise

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
        # draw initial state
        x = self.pi.sample()
        # draw initial distribution
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
    
    def expected_log_likelihood(self,gamma,Y,Dcal):

        def ln(x):
            assert np.all(x>0)
            return np.nan_to_num(np.log(x))

        log.debug('calculating the expected log likelihood')
        X = [np.sum(g,1) for g in gamma]
        D = [np.sum(d,0) for d in Dcal]
        duration_likelihood = self.duration_likelihood()
        l = np.dot(X[0],ln(self.pi.pi))
        assert not np.isnan(l),  np.dot(X[0],ln(self.pi.pi))
        for x,xi,y,d in zip(X[1:],X[:-1],Y[1:],D[1:]):
            l += np.dot(xi, np.dot(ln(self.A.A), x))
            assert not np.isnan(l),  np.dot(xi, np.dot(ln(self.A.A), x))
            l += np.dot([ln(self.O(i,y)) for i in self.states], x)
            assert not np.isnan(l), np.dot([ln(self.O(i,y)) for i in
                                            self.states], x)
            l += np.dot(x, np.dot(duration_likelihood, d))
            assert not np.isnan(l),  np.dot(x, np.dot(duration_likelihood, d))
        return l[0]
    
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
        
        Parameters
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
        usually calculated in the forward algorithm. Note also that the variable
        calculated here is properly normalised, and hence suffers no scaling
        issues.  
        """
        log.info('running forward algorithm')
        T = len(Y)
        alpha = [np.zeros((self.K, len(self.durations))) for y in Y]
        bstar = [np.zeros((self.K,1)) for y in Y]
        
        E = [np.zeros((self.K,1)) for y in Y]
        S = [np.zeros((self.K,1)) for y in Y]
        U = np.zeros((self.K,1))
        
        if D is None:
            D = self.duration_likelihood()
        
        assert D.shape == alpha[0].shape
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
                alpha[t] = (S[t-1] * D) + (bstar[t-1] * alpha_shifted)
            
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
            E[t][:,0] = alpha[t][:,0] * bstar[t][:,0]
            # S = p(x_t+1, d_t=1 | y_1 ... y_t)
            S[t] = np.dot(self.A.A.T,E[t])
                
        return alpha, bstar, E, S
    
    def backward(self,Y,bstar, D=None):
        """
        backward algorithm as specified by Yu and Kobayashi 2006.
        
        Parameters
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
        Estar = [np.zeros((self.K,1)) for y in Y]
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
            Estar[t][:,0] = (D * beta[t]).sum(1)
            Sstar = np.zeros((self.K,1))
            for j in self.states:
                for i in self.states:
                    Sstar[j,0] += Estar[t][i,0] * self.A.A[j,i]
        return beta, Estar
    
    @types(Y=list)
    def forward_backward(self,Y,D=None):
        """
        Forward Backward algorithm.
        
        Parameters
        ---------
        Y : list of np.ndarray
            data
        bstar : list of np.ndarray
            p(y_t|x_t) / p(y_t|y_1 .. y_t-1)
        D : np.ndarray, optional
            duration likelihoods
        """
        if D is None:
            D = self.duration_likelihood()
        alpha, bstar, E, S = self.forward(Y,D)
        beta, Estar = self.backward(Y,bstar,D)
        gamma = [f*b for f,b in zip(alpha,beta)]
        assert all([a.sum() for a in gamma])
        Tcal = [np.zeros((self.K,self.K)) for y in Y]
        Dcal = [np.zeros((self.K,len(self.durations))) for y in Y]
        for t in range(1,len(Y)):
            for i in self.states:
                for j in self.states:
                    Tcal[t][i,j] = E[t-1][i] * self.A.A[i,j] * Estar[t][j]
        for t in range(1,len(Y)):
            for i in self.states:
                for d_index,d in enumerate(self.durations):
                    Dcal[t][i,d_index] = S[t-1][i,0] * D[i,d_index] * beta[t][i,d_index]
        return gamma, Tcal, Estar, Dcal
    
    def backward_sample(self, alpha):
        """
        Samples a state sequence, in reverse order, given the forward variable
        """
        T = len(alpha)
        s = Categorical(alpha[-1].sum(1)).sample()
        d_index = Categorical(alpha[-1][s,:]).sample()
        d = self.durations[d_index]
        for t in reversed(xrange(T-1)):
            yield s,d
            if d > 1:
                d -= 1
            else:
                s = Categorical(alpha[t].sum(1)*self.A.A[:,s]).sample()
                d = Categorical(alpha[t][s,:]).sample() + 1 # note the +1!
        yield s,d
    
    def slice_sample(self,S,D):
        log.info('forming slice')
        u = []
        for s,d in zip(S,D):
            l = self.D(s,d)
            u.append(np.random.uniform(low=0, high=l))
            assert u[-1] < l
        return np.array(u)
    
    def beam_forward(self, Y, u=None):
        
        log.info('running forward algorithm w/ slice sampling')
        
        T = len(Y)
        E = np.zeros((self.K,1))
        U = np.zeros((self.K,1))
        
        if u is None:
            S, Z, D_seq = self.sim(len(Y)) # ignore Z
            u = self.slice_sample(S,D_seq)
                
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
        
        log.debug('starting forward recursion')
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
                alpha_shifted = np.hstack(
                    [alpha[t-1][:, 1:], np.zeros((self.K, 1))]
                )
                alpha[t] = (S * D) + (bstar[t-1] * alpha_shifted)
            
            U[:,0] = np.array([self.O(i,Y[t]) for i in self.states])
            rinv = (alpha[t] * U).sum()
            bstar[t] = U / rinv
            E[:,0] = alpha[t][:,0] * bstar[t][:,0]
            S = np.dot(self.A.A.T,E)
            alpha[t] = alpha[t] / alpha[t].sum()
        return alpha, bstar       
    
    def beam(self, Y, S=None, Dseq=None, its = 100, burn_in=20):
        if S == None or Dseq == None:
            S, Z, Dseq = self.sim(len(Y)) # ignore Z
        Sout, Dout = [], []
        for i in range(its):
            log.info("beam iteration: %s of %s"%(i,its))
            U = self.slice_sample(S,Dseq)
            alpha, bstar = self.beam_forward(Y,U)
            S,D = zip(*[(s,d) for (s,d) in self.backward_sample(alpha)])
            if i > burn_in:
                Sout.append(S)
                Dout.append(D)
        return Sout, Dout
    
    def update(self, gamma, Tcal, Estar, Y, Dcal):
        log.info('updating parameters')        
        self.A.update(Tcal)
        self.pi.update(Estar)
        self.O.update(gamma,Y)
        self.D.update(Dcal)
    
def baum_welch(Y,K,stopping_threshold=0.001,multiple_restarts=10):
    model_store = []
    l_end_store = []
    l_store = []
    for restart in range(multiple_restarts):
        m = initialise_EDHMM(Y, K)
        l = [-100000]
        it = 1
        delta_l = 10000
        while abs(delta_l) > stopping_threshold:
            D = m.duration_likelihood()
            gamma, Tcal, Estar, Dcal = m.forward_backward(Y, D)
            m.update(gamma, Tcal, Estar, Y, Dcal)
            log.debug('finished BW iteration %s'%it)
            it += 1
            m.report()
            l.append(m.expected_log_likelihood(gamma, Y, Dcal))
            delta_l = l[-1]-l[-2]
            log.debug("change in likelihood: %s"%delta_l)
            if delta_l < 0:
                log.warn("negative change in likelihood!")
        print "\n\n"
        model_store.append(m)
        l_store.append(l[1:])
        l_end_store.append(l[-1])
    log.debug('final likelihoods: %s'%l_end_store)
    i = l.index(max(l))    
    return model_store[i], l_store[i]
    
def initialise_EDHMM(Y,K):
    log.info("initialising new EDHMM")
    pi = Initial(np.array([1./K for k in range(K)]))
    # some random matrix with zeros on the diagonal
    A = (np.ones((K,K)) - np.eye(K))*np.random.random((K,K))
    # normalise rows
    A = (A.T / A.sum(1)).T
    A = Transition(A)
    # for the initial means just do a quick k-means...
    T = len(Y)
    try:
        N = Y[0].shape[0]
    except IndexError:
        N = 1
    Y = np.array(Y)
    Y.shape = (T,N)
    book, distortion = scipy.cluster.vq.kmeans(Y,K)
    means = list(book.flatten())
    # find the variances of the data that's been clustered using kmeans
    code, dist = scipy.cluster.vq.vq(Y,book)
    precisions = [1/np.var(Y[code==k]) for k in range(K)]
    O = Gaussian(means,precisions)
    # estimate the durations from the kmeans code book
    dmax = np.zeros(K)
    for k in range(K):
        this_duration = 0
        for c in code:
            if c==k:
                this_duration += 1
            else:
                dmax[k] = max(this_duration, dmax[k])
                this_duration = 0
    D = Poisson(dmax)
    return EDHMM(A,O,D,pi)
    

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
