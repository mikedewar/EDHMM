import edhmm
from transition import *
from initial import *
from emission import *
from duration import *
from utils import *

reload(edhmm)

T = 200

import sys
import logging
import pylab as pb
import pprint

pp = pprint.PrettyPrinter(indent=4)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

A = Transition(pb.array([[0, 0.3, 0.7], [0.6, 0, 0.4], [0.3, 0.7, 0]]))
O = Gaussian([-10,0,10],[1,1,1])
D = Poisson([2,5,8])
pi = Initial(K=3,beta=0.001)
m = edhmm.EDHMM(A,O,D,pi)
X,Y,Dseq = m.sim(T)

if True:
    # set the above to true for some awesome testing action!
    U = [np.random.uniform(0,0.01) for y in Y]
    W = m.worthy_transitions(U)
    alpha = m.beam_forward_new(Y, W=W)
    Z = m.beam_backward_sample_new(alpha,W)
    D_sample = m.sample_D(Z)
    A_sample = m.sample_A(Z)
    O_sample = m.sample_O(Z,Y)
else:
    A,D,O = m.beam_new(Y)
    

    D = pb.array(D)
    pb.subplot(1,3,1)
    pb.hist(D[:,0])
    pb.subplot(1,3,2)
    pb.hist(D[:,1])
    pb.subplot(1,3,3)
    pb.hist(D[:,2])
    pb.show()


#pp.pprint(W)
#pp.pprint(alpha)


"""
S_out, D_out = m.beam(Y, S=X, Dseq=Dseq, its=1000)

gamma_sampling = np.sum(S_out,0)

K = len(pi)
gs = np.zeros((K,T))
for t in range(T):
    for k in range(K):
        for S in S_out:
            if S[t] == k:
                gs[k,t] += 1

gamma_dp, Tcal, Estar, Dcal = m.forward_backward(Y)
gd = np.array([g.sum(1) for g in gamma_dp]).T


pb.subplot(2,1,1)
pb.imshow(gs, aspect='auto', interpolation='nearest',origin='lower')
pb.subplot(2,1,2)
pb.imshow(gd, aspect='auto', interpolation='nearest',origin='lower')
pb.show()


"""