import math

def lr_cosine_schedule(t, amax, amin, Tw , Tc):
    if t<Tw :
        return t/Tw*amax
    elif Tw <= t and t <= Tc:
        return amin +1/2*(1+math.cos((t-Tw)/(Tc-Tw)*math.pi))*(amax-amin)
    else:
        return amin