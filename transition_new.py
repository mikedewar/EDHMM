import math
import numpy as np

categorical = lambda alpha: np.random.multinomial(1,alpha).nonzero()[0][0]

class Transition:
    def __init__(self, K, A):
        self.alpha = [np.array([1.0/(K-1) for i in range(K)]) for j in range(K)]
        for i in range(K):
            self.alpha[i][i] = 0
        self.K = K
        self.states = range(K)
        A = A + 0.0001
        for i in self.states:
            A[i] = A[i]/A[i].sum()
        self.A = A
    
    def likelihood(self,i,j):
        assert i in self.states
        assert j in self.states
        return np.log(self.A[i,j])
    
    def sample_x(self, i):
        try:
            return categorical(self.A[i].flatten())
        except:
            print self.A[i].flatten()
            raise

    
    def sample_A(self, Z):

        X = [z[0] for z in Z]
        
        n = dict([(i,dict([(j,0) for j in self.states])) for i in self.states])
        now = X[0]
        for x in X:
            if now != x:
                n[now][x] += 1
                now = x
        
        A = np.zeros((self.K,self.K))
        
        for i in self.states:
            A[i] = np.random.dirichlet(self.alpha[i] + n[i].values())
        
        A = A + 0.0001
        for i in self.states:
            A[i] /= A[i].sum()
        
        return A      
    
    def update(self, Z):
        self.A = self.sample_A(Z)
        
if __name__ == "__main__":
    import pylab as pb
    Z = np.load('Z.npy')
    A = Transition(
        K=3,
        A=pb.array([[0, 0.3, 0.7], [0.6, 0, 0.4], [0.3, 0.7, 0]])
    )
    for i in range(10):
        print A.sample_A(Z)
        print "\n"