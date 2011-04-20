import numpy as np

from edhmm import EDHMM, baum_welch
from emission import Emission
from duration import Duration
from gen_test_data import gen_models
from utils import *

import logging
logging.basicConfig(filename="unittesting.log", level=logging.DEBUG)


T = 600

def pytest_generate_tests(metafunc):
    if "model" in metafunc.funcargnames:
        models = gen_models()
        for model in models:
            metafunc.addcall(funcargs=dict(model=model))

def test_forward(model):
    X,Y = model.sim(T)
    alpha, bstar, E, S = model.forward(Y)
    assert all(isprob(f) for f in alpha)
    
def test_backward(model):
    X,Y = model.sim(T)
    alpha, bstar, E, S = model.forward(Y)
    beta, Estar = model.backward(Y, bstar)
    assert all(isprob(f*b) for f,b in zip(alpha,beta))
    
def test_forward_backward(model):
    X,Y = model.sim(T)
    gamma, Tcal, Estar, Dcal = model.forward_backward(Y)
    assert all(isprob(g) for g in gamma)

def test_baum_welch(model):
    X,Y = model.sim(T)
    m_est,l = baum_welch(Y,K = model.K, stopping_threshold=0.01)
    assert all(np.diff(l) > 0)
    
