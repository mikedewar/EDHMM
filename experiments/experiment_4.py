# this experiment runs the algorithm having set U to zero for all time
# and manually picking minimum and maximum durations. This simulates the 
# forward backward algo, and is liable to not be all that great / fast etc

import sys
sys.path.append('..')

from emission import Gaussian
from duration import Poisson
from transition import Transition
from initial import Initial
from edhmm import EDHMM

import pylab as pb
import numpy as np
import logging


logging.basicConfig(
    stream=sys.stdout,
    filename="experiment_4.log", 
    filemode="w",
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
    mu = [-3, 0, 3], 
    tau = [
        np.array([[1]]),
        np.array([[1]]),
        np.array([[1]])
    ]
)
D = Poisson(
    mu = [5,15,20], 
    alpha=[1, 1, 1],
    beta=[0.0001, 0.0001, 0.0001],
    support_step = 20
)
pi = Initial(K=3,beta=0.001)

m = EDHMM(A,O,D,pi)

T = 500

X,Y,Dseq = m.sim(T)

m.A.A = pb.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
m.O.mu = [0,0,0]
m.D.mu = [1,1,1]

np.save("exp4_X.npy", X)
np.save("exp4_D.npy", Dseq)
np.save("exp4_Y.npy", Y)
np.save("exp4_Z.npy", zip(X,Dseq))

for md in range(5,30):
    ### OK so we force some variables here, not generally reccommended!
    U = [0 for y in Y]
    min_d = [1,1,1]
    max_d = [md for i in range(3)]

    L = m.beam(
        [Y], its=1000, burnin=500, name = "exp4_%s"%md, online=True, 
        force_U = [U], min_d = min_d, max_d = max_d, sample_U=False
    )
    np.save("exp4_L_%s"%md, L)
