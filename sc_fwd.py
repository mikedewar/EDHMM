from emission_new import Gaussian
from duration_new import Poisson
from transition_new import Transition
from initial import Initial
from edhmm import EDHMM

import pylab as pb
import numpy as np
import logging
import sys

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
    mu_0 = [0, 0, 0], 
    kappa = 0.01, 
    mu = [-4, 0, 4], 
    tau = [
        np.array([[1]]),
        np.array([[1]]),
        np.array([[1]])
    ]
)

D = Poisson(
    mu = [5,15,30], 
    alpha=[1, 1, 1],
    beta=[0.0001, 0.0001, 0.0001],
    support_step = 1
)

pi = Initial(K=3, beta=0.001)
m = EDHMM(A,O,D,pi)

T = 500

X,Y,Dseq = m.sim(T)

m.A.A = pb.array(
   [[0, 0.5, 0.5], 
    [0.5, 0, 0.5], 
    [0.5, 0.5, 0]]
)
m.O.mu = [-1,0,1]
m.D.mu = [1,1,1]

m.left,m.right = zip(*[m.D.support(i) for i in m.states])

m.l = {}
for i in m.states:
    for j in m.states:
        for di in range(1,max(m.right)+1):
            for dj in range(1,max(m.right)+1):
                if di == 1:                
                    m.l[(i,j,di,dj)] = np.exp(
                        m.A.likelihood(i,j) + m.D.likelihood(j,dj)
                    )
                else:
                
                    if i == j and dj==di-1:
                        m.l[(i,j,di,dj)] = 1
                    else:
                        m.l[(i,j,di,dj)] = 0

U = [np.random.uniform(0,0.00001) for y in Y]
#U = [0 for y in Y]


pb.subplot(2,1,1)
pb.plot(Y)
pb.subplot(2,1,2)
pb.plot(X)

alpha = m.beam_forward(Y, U=U)
Z = m.beam_backward_sample(alpha,U)

m.l = {}
for i in m.states:
    print i
    for j in m.states:
        for di in range(1,max(m.right)+1):
            for dj in range(1,max(m.right)+1):
                if di == 1:
                    m.l[(i,j,di,dj)] = np.exp(
                        m.A.likelihood(i,j) + m.D.likelihood(j,dj)
                    )
                else:
                
                    if i == j and dj==di-1:
                        m.l[(i,j,di,dj)] = 1
                    else:
                        m.l[(i,j,di,dj)] = 0


U = m.slice_sample(Z)
X_sample = [z[0] for z in Z]
pb.plot(X_sample,'r')

m.D.mu = m.D.sample_mu([Z])
print m.D.mu

alpha = m.beam_forward(Y, U=U)
Z = m.beam_backward_sample(alpha,U)

m.l = {}
for i in m.states:
    print i
    for j in m.states:
        for di in range(1,max(m.right)+1):
            for dj in range(1,max(m.right)+1):
                if di == 1:
                    m.l[(i,j,di,dj)] = np.exp(
                        m.A.likelihood(i,j) + m.D.likelihood(j,dj)
                    )
                else:
                
                    if i == j and dj==di-1:
                        m.l[(i,j,di,dj)] = 1
                    else:
                        m.l[(i,j,di,dj)] = 0

U = m.slice_sample(Z)
X_sample = [z[0] for z in Z]
pb.plot(X_sample,'g')

m.D.mu = m.D.sample_mu([Z])
print m.D.mu

alpha = m.beam_forward(Y, U=U)
Z = m.beam_backward_sample(alpha,U)

m.l = {}
for i in m.states:
    print i
    for j in m.states:
        for di in range(1,max(m.right)+1):
            for dj in range(1,max(m.right)+1):
                if di == 1:
                
                        m.l[(i,j,di,dj)] = np.exp(
                            m.A.likelihood(i,j) + m.D.likelihood(j,dj)
                        )
                else:
                
                    if i == j and dj==di-1:
                        m.l[(i,j,di,dj)] = 1
                    else:
                        m.l[(i,j,di,dj)] = 0

U = m.slice_sample(Z)
X_sample = [z[0] for z in Z]
pb.plot(X_sample,'m')

m.D.mu = m.D.sample_mu([Z])
print m.D.mu


pb.ylim((-0.1,2.1))
pb.show()



