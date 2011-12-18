"""
A simple experiment with three states to demonstrate the capability of the algorithm
in a situation where we'd expect it to succeed. Tail experiment_1.log to watch the progress
of the algorithm as it runs.
"""

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
import sys

logging.basicConfig(
    filename="experiment_1.log", 
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

np.save("exp1_X.npy", X)
np.save("exp1_D.npy", Dseq)
np.save("exp1_Y.npy", Y)
np.savetxt('exp1_Y.dat',Y)
np.save("exp1_Z.npy", zip(X,Dseq))

if True:

    m.A.A = pb.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    m.O.mu = [-1,0,1]
    m.D.mu = [1,1,1]

    L = m.beam(
        [Y],its=3000, burnin=500, name = "exp1", 
        online=True, sample_U = True
    )
    np.save("exp1_L", L)
