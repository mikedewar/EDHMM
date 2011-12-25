# This experiment runs the forward backward algorithm on experiment 1's data 

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
    filename="experiment_5.log", 
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

max_durations = range(5,max(Dseq)+5)

pb.figure(figsize=(8,2))

for sample in range(5):
    L = []
    for max_d in max_durations:
        alphahat = m.forward(Y, max_d)
        Z = m.backward_sample(alphahat, max_d)
        l = m.loglikelihood([Z],[Y])
        L.append(l)
    pb.plot(max_durations, L, color='blue', alpha=0.3)
pb.xlabel('maximum durations considered')
pb.ylabel('log likelihood')

beam_likelihoods = np.load('experiment_4_data/exp4a_L.npy')[:10]
for beam_likelihood in beam_likelihoods:
    plt.plot(max_durations, [beam_likelihood for d in max_durations], color="red", alpha=0.3)

lTrue = m.loglikelihood([zip(X,Dseq)], [Y])
pb.plot(max_durations, [lTrue for i in max_durations], color="green")

pb.savefig('forward_backward_likleihoods.pdf')