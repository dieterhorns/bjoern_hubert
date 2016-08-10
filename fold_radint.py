import numpy as np
import matplotlib.pyplot as plt
from numpy import logical_and as _and
import scipy
from scipy.integrate import quad
from scipy import optimize as opti
import sys

if len(sys.argv)>1:
    h=np.float(sys.argv[1])
else:
    h=0.03

s=0.1
epsrel=1e-3
# read
R,dndo,dndo_unc,f,back,nevt,do=np.loadtxt('./results/result0.25_100.dat',unpack=True)



# no, I want to calculate the very same function
# for discrete values of r, folded with the 
# tophat function

# I have had numerical issues, when also integrating over the same grid,
# therefore, I will do it using interpolation
def finter(x):
    return x*scipy.interp(x,R,f)

def finterdphi(x,r):
    val = 2.*x*np.arccos( (x*x+r*r-h*h)/2./x/r) * scipy.interp(x,R,f)
    return val

res=[]
for r in R:
    if r<=h:
        y,err= quad(finter,0,h-r,epsrel=epsrel)
        result = y * 2. * np.pi 
        if r>0:
            y,err= quad(finterdphi,h-r,h+r,args=(r,),epsrel=epsrel)
            result = result + y
    else:
        y,err = quad(finterdphi,r-h,r+h,args=(r,),epsrel=epsrel)
        result= y

    res.append(result)

# res contains the folded psf 
res = np.asarray(res)/np.pi/h/h

twopi=np.arctan(1.)*8.
pi=twopi/2.
d2r=pi/180.
r2d=180./pi


def likeli(x):
    a=x[0]
    b=x[1]
    if h<1e-3:
        mui = f*do*a+b*do
    else:
        mui = res*do*a+b*do
    logli = np.sum(nevt*np.log(mui) - mui)
    return(-logli)


method='Nelder-Mead'
x0=[1.0,0.1]
mini = opti.minimize(likeli,x0,method=method)
a=mini.x[0]
b=mini.x[1]



k, axarr = plt.subplots(2, sharex=True)
ma=dndo_unc>0
axarr[0].set_yscale("log")
axarr[0].set_xlim([0,0.25])
if h>1e-3:
    axarr[0].plot(R,res)
axarr[0].plot(R,f)
axarr[0].errorbar(R[ma],dndo[ma],yerr=dndo_unc[ma],fmt='o')
axarr[1].plot(R,np.zeros(R.size))
axarr[1].errorbar(R[ma],(dndo[ma]-(a*res[ma]+b*do[ma]))/dndo_unc[ma],yerr=np.ones(R.size)[ma],fmt='o')
plt.show()
