import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
twopi=np.arctan(1.)*8.

# demonstration how to do a simple folding 

#example kernels
gauss =lambda x,s:  1./np.sqrt(twopi)/s*np.exp(-x*x/2./s/s)
th    =lambda x,s:  1./2./s * np.ones(x.size)*(np.abs(x)<s)

# folding f1 with f2, both are simple_parametric functions
conv = lambda x0,xp,f1,f2,s1,s2: simps(f1(x0-xp,s1) * f2(xp,s2),xp)

#parameters
th_rad=3.0
ga_sig=0.8

# test positions
xpos=np.arange(-10,10+0.1,0.1)
res=[]
for x0 in xpos:
    res.append(conv(x0,xpos,gauss,th,ga_sig,th_rad))

plt.plot(xpos,res,xpos,th(xpos,th_rad))
plt.show()
