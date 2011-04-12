from edhmm import EDHMM
from emission import Emission
from duration import Duration
from gen_test_data import gen_models
from utils import *

T = 200

def pytest_generate_tests(metafunc):
    if "model" in metafunc.funcargnames:
        models = gen_models()
        for model in models:
            metafunc.addcall(funcargs=dict(model=model))

def test_forward(model):
    X,Y = model.sim(T)
    F,N = model.forward(Y)
    assert all(isprob(f) for f in F)
    
def test_backward(model):
    X,Y = model.sim(T)
    F,bstar = model.forward(Y)
    B = model.backward(Y, bstar)
    assert all(isprob(f*b) for f,b in zip(F,B))
    
def test_forward_backward(model):
    X,Y = model.sim(T)
    F, B, As = model.forward_backward(Y)
    assert all(isprob(f) for f in F)
    assert all(isprob(f*b) for f,b in zip(F,B))
    assert all(isprob(a) for a in As)
    
def test_baum_welch(model):
    pass