import numpy as np
from scipy import optimize as opti
import matplotlib.pyplot as plt
#x,dndo,dndo_unc,sig,back=np.loadtxt('result2.0.dat',unpack=True)
#x,dndo,dndo_unc,sig,back=np.loadtxt('result.dat',unpack=True)
#x,dndo,dndo_unc,sig,back=np.loadtxt('result0.5.dat',unpack=True)
#x,dndo,dndo_unc,sig,back=np.loadtxt('result0.25.dat',unpack=True)
x,dndo,dndo_unc,sig,back=np.loadtxt('result1.0.dat',unpack=True)



def chisquare(x):
    a=x[0]
    resu = np.sum( np.square((dndo - (a*sig + (1.-a)*back))/dndo_unc))
    print a,resu
    return resu


method='Nelder-Mead'
result=opti.minimize(chisquare,[1.0],method=method)
a=result.x[0]
b=1.-a
f, axarr = plt.subplots(2, sharex=True)
axarr[0].set_yscale("log")
axarr[0].errorbar(x,dndo,yerr=dndo_unc,fmt='o')
axarr[0].plot(x,a*sig,label='norm='+str(a))
axarr[0].plot(x,b*back,label='back='+str(b))
axarr[0].plot(x,(a*sig+b*back))
axarr[0].legend()
axarr[1].plot(x,np.zeros(x.size))
axarr[1].errorbar(x,(dndo-a*sig-b*back)/dndo_unc,yerr=np.ones(x.size),fmt='o',label='chi2='+str(result.fun)+', dof='+str(x.size-2))
axarr[1].legend()
plt.show()


