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
    tau = [np.array([[3]]),np.array([[3]]),np.array([[3]])]
)
D = Poisson(mu = [3,5,10], alpha=[3,3,3], beta=[0.8,0.5,0.25])
pi = Initial(K=3,beta=0.001)
m = EDHMM(A,O,D,pi)

T = 1000

X,Y,Dseq = m.sim(T)

np.save("X.npy", X)
np.save("D.npy", Dseq)
np.save("Y.npy", Y)
np.save("Z.npy", zip(X,Dseq))

if True:
    As, O_means, O_precisions, D_mus, Zs = m.beam(Y, maxits = 1000)

    np.save("As",As)
    np.save("O_m", O_means)
    np.save("O_p", O_precisions)
    np.save("D_mus", D_mus)
    np.save("Zs", Zs)
