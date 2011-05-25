import numpy as np

def elnsum(lnx,lny):
    if lnx > lny:
        return lnx + np.log(1 + np.exp(lny - lnx))
    else:
        return lny + np.log(1 + np.exp(lnx - lny))