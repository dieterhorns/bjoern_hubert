
import numpy as np
import pyfits
from scipy.integrate import simps
import matplotlib.pyplot as plt

twopi=np.arctan(1.)*8.
pi=twopi/2.

def Sp(E,c0,c1,beta):
    return np.sqrt((c0*(E/100.)**beta)**2.+c1**2.)

def k(X,sig,gam,E,c0,c1,beta):			#king, tdev abweichung vom zentrum in rad
#	X=tdev/np.sqrt((c0*(E/100.)**beta)**2.+c1**2.)
	res=1./(2.*pi*sig**2.) *(1.-1./gam) *(1.+1./(2.*gam)*(X/sig)**2.)**(-gam)
	return res

def f(N,sigt,sigc):					#norm
	res=1./(1.+N*(sigt/sigc)**2.)
	return res

def p(tdev,sigc,gamc,sigt,gamt,N,E,c0,c1,beta):		#psf 	x=dRA, y=dDEC	umrechnung in rad noetig
	X  =tdev/np.sqrt((c0*(E/100.)**beta)**2.+c1**2.)
	res=f(N,sigt,sigc)*k(X,sigc,gamc,E,c0,c1,beta)+(1.-f(N,sigt,sigc))*k(X,sigt,gamt,E,c0,c1,beta)
	return res



psf_fits=pyfits.open('./data/psf_P8R2_SOURCE_V6_PSF.fits')


elist = np.power(10.,np.arange(1.,6.,0.1))
cthe = 0.9*np.ones(elist.size)
psf  = 2*np.ones(elist.size)

ebin     = np.int_(np.floor((np.log10(elist)/0.25-3)))
tbin     = np.int_(np.floor((cthe/0.1-2)))
psfind   = np.int_(psf*3+1)

sigc=map(lambda pind,tind,eind: psf_fits[pind].data.field('SCORE')[0][tind][eind] ,psfind,tbin,ebin)
gamc=map(lambda pind,tind,eind: psf_fits[pind].data.field('GCORE')[0][tind][eind], psfind,tbin,ebin)
sigt=map(lambda pind,tind,eind: psf_fits[pind].data.field('STAIL')[0][tind][eind], psfind,tbin,ebin)
gamt=map(lambda pind,tind,eind: psf_fits[pind].data.field('GTAIL')[0][tind][eind], psfind,tbin,ebin)
N   =map(lambda pind,tind,eind: psf_fits[pind].data.field('NTAIL')[0][tind][eind], psfind,tbin,ebin)
c0  =map(lambda pind: psf_fits[pind+1].data.field('PSFSCALE')[0][0], psfind)
c1  =map(lambda pind: psf_fits[pind+1].data.field('PSFSCALE')[0][1], psfind)
beta=map(lambda pind: psf_fits[pind+1].data.field('PSFSCALE')[0][2], psfind)

drad = np.linspace(0.,pi,1000)
resu=[]
for i in np.arange(0,elist.size):
#    prad=twopi*np.sin(drad)*(np.asarray(map(lambda x: p(x,sigc[i],gamc[i],sigt[i],gamt[i],N[i],elist[i],c0[i],c1[i],beta[i]),drad)))
    prad=(np.asarray(map(lambda x: twopi*x/np.square(Sp(elist[i],c0[i],c1[i],beta[i]))*p(x,sigc[i],gamc[i],sigt[i],gamt[i],N[i],elist[i],c0[i],c1[i],beta[i]),drad)))
    print (simps(prad,drad))


#plt.plot(np.log10(elist),np.log10(np.asarray(resu)*180./pi))
#plt.show()
