from edhmm import EDHMM
from emission import Emission
from duration import Duration
from gen_test_data import gen_models

T = 200

def pytest_generate_tests(metafunc):
    if "model" in metafunc.funcargnames:
        models = gen_models()
        for model in models:
            metafunc.addcall(funcargs=dict(model=model))

def test_forward(model):
    X,Y = model.sim(T)
    F,N = model.forward(Y)
    assert all(f.sum()==1 for f in F)
    
def test_backward(model):
    pass
    
def test_forward_backward(model):
    pass
    
def test_baum_welch(model):
    pass