import numpy as np

def elnsum(lnx,lny):
    if np.isnan(lnx) or np.isnan(lny):
        if np.isnan(lnx):
            return lny
        else:
            return lnx
    else:
        if lnx > lny:
            return lnx + np.log(1 + np.exp(lny - lnx))
        else:
            return lny + np.log(1 + np.exp(lnx - lny))