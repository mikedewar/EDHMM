from utils import *
from emission_new import Gaussian
from duration_new import Poisson
from transition_new import Transition
from initial import *
from edhmm import *

import pylab as pb
import numpy as np
import logging
import sys
import time

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
pi = Initial(K=3, beta=0.001)
m = EDHMM(A,O,D,pi)

T = 1000

X,Y,Dseq = m.sim(T)

U = [np.random.uniform(0,0.00001) for y in Y]

t = time.time()
m.left,m.right = zip(*[m.D.support(i) for i in m.states])

m.l = {}
for i in m.states:
    for j in m.states:
        for di in range(1,max(m.right)+1):
            m.l[(i,j,di)] = np.exp(
                m.A.likelihood(j,i) + m.D.likelihood(i,di)
            )

alpha = m.beam_forward(Y, U=U)
print time.time() - t

Z_sample = m.beam_backward_sample(alpha,U)
X_sample = [z[0] for z in Z_sample]

pb.plot(X)
pb.plot(X_sample)
pb.ylim((-0.1,2.1))
pb.show()



