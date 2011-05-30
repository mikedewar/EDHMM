from numpy import log, exp

def elnsum(lnx,lny):
    if lnx > lny:
        return lnx + log(1 + exp(lny - lnx))
    else:
        return lny + log(1 + exp(lnx - lny))