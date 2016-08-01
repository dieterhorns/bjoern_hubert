
import matplotlib.pyplot as plt
import numpy as np
import pyfits
from scipy.integrate import simps

def k(tdev,sig,gam,E,c0,c1,beta):			#king, tdev abweichung vom zentrum in rad
	X=tdev/np.sqrt((c0*(E/100.)**beta)**2.+c1**2.)
	res=1./(2.*pi*sig**2.) *(1.-1./gam) *(1.+1./(2.*gam)*(X/sig)**2.)**(-gam)
	return res

def Sp(E,c0,c1,beta):
    return np.sqrt((c0*(E/100.)**beta)**2.+c1**2.)


def f(N,sigt,sigc):					#norm
	res=1./(1.+N*(sigt/sigc)**2.)
	return res

def p(tdev,sigc,gamc,sigt,gamt,N,E,c0,c1,beta):		#psf 	x=dRA, y=dDEC	umrechnung in rad noetig
	res=f(N,sigt,sigc)*k(tdev,sigc,gamc,E,c0,c1,beta)+(1.-f(N,sigt,sigc))*k(tdev,sigt,gamt,E,c0,c1,beta)
	return res


def dangle(ra1,ra2,dec1,dec2): # each given in degrees
    dec1= 90.-dec1 # go from long to sphere
    dec2= 90.-dec2 # go from long to sphere

    sr1 = np.sin(ra1*d2r)
    cr1 = np.cos(ra1*d2r)

    sd1 = np.sin(dec1*d2r)
    cd1 = np.cos(dec1*d2r)

    sr2 = np.sin(ra2*d2r)
    cr2 = np.cos(ra2*d2r)

    sd2 = np.sin(dec2*d2r)
    cd2 = np.cos(dec2*d2r)

    resu = cr1*sd1*cr2*sd2 + sr1*sd1*sr2*sd2 + cd1*cd2
    resu[resu>1.0] = 1.0
    resu[resu<-1.0]= -1.0
    return(np.arccos(resu)) # in radiant




psf_fits=pyfits.open('./data/psf_P8R2_SOURCE_V6_PSF.fits')
dat=pyfits.open('./data/source_psf2_3.fits')	
twopi=np.arctan(1.)*8.
pi=twopi/2.
d2r=pi/180.
r2d=180./pi
RAc=166.11798415
DECc=38.20630703
frac=.97
frac=0.67289062
cosTlist=np.cos(np.array(dat[1].data.field('THETA   '))*pi/180)	#liste der photonen neigungswinkel zum detektor (fuer thetabin) in cos(theta)
Elist=np.array(dat[1].data.field('ENERGY '))			#liste der photonenenergien in MeV
Ralist= np.array(dat[1].data.field('RA'))
Declist= np.array(dat[1].data.field('DEC'))
evtype = np.array(dat[1].data.field('EVENT_TYPE')[:,26])

drad = dangle(RAc*np.ones(Ralist.size),Ralist,DECc*np.ones(Declist.size),Declist)

ebin     = np.int_(np.floor((np.log10(Elist)/0.25-3)))
tbin     = np.int_(np.floor((cosTlist/0.1-2)))
psfclass = np.int_(np.ones(evtype.size)*2*np.logical_not(evtype)+ np.ones(evtype.size)*3*evtype)
psfind   = np.int_(psfclass*3+1)


sigc=map(lambda pind,tind,eind: psf_fits[pind].data.field('SCORE')[0][tind][eind] ,psfind,tbin,ebin)
gamc=map(lambda pind,tind,eind: psf_fits[pind].data.field('GCORE')[0][tind][eind], psfind,tbin,ebin)
sigt=map(lambda pind,tind,eind: psf_fits[pind].data.field('STAIL')[0][tind][eind], psfind,tbin,ebin)
gamt=map(lambda pind,tind,eind: psf_fits[pind].data.field('GTAIL')[0][tind][eind], psfind,tbin,ebin)
N   =map(lambda pind,tind,eind: psf_fits[pind].data.field('NTAIL')[0][tind][eind], psfind,tbin,ebin)
c0  =map(lambda pind: psf_fits[pind+1].data.field('PSFSCALE')[0][0], psfind)
c1  =map(lambda pind: psf_fits[pind+1].data.field('PSFSCALE')[0][1], psfind)
beta=map(lambda pind: psf_fits[pind+1].data.field('PSFSCALE')[0][2], psfind)
npred=drad.size
sp = np.asarray(map(lambda ener,c_0,c_1,b: Sp(ener,c_0,c_1,b),Elist,c0,c1,beta))
maxa = 0.25
sa = twopi * (1. - np.cos(maxa * d2r))


# plot radial profile
hist,bins=np.histogram(drad*r2d,bins=100,range=(0,maxa))
bcent = bins[:-1]+0.5*(bins[2]-bins[1])
cosb=np.cos(bins*d2r)
do= -twopi*np.diff(cosb)
dndo=hist/do
dndo_unc = np.sqrt(hist)/do
#print bcent
#print dndo
#print dndo_unc
ax=plt.subplot(111)
ax.set_yscale("log")
ax.errorbar(bcent,dndo,yerr=dndo_unc,fmt='o')
#ax.plot(bcent,pi*np.genfromtxt('tst.dat'))
#plt.show()
sigfit=[]
bkgfit=[]
for bc in bcent:
    prob_pt = np.asarray(map(lambda dist,sc,gc,st,gt,no,c_0,c_1,b,ener: p(dist,sc,gc,st,gt,no,ener,c_0,c_1,b),
                  bc*d2r*np.ones(npred),sigc,gamc,sigt,gamt,N,c0,c1,beta,Elist))
    sig = prob_pt/sp/sp
#    sig = prob_pt/sp/sp * drad/np.sin(drad)
    sig[np.isnan(sig)] = prob_pt[np.isnan(sig)]/np.square(sp[np.isnan(sig)])
    b   = np.ones(npred)/sa
    sigfit.append(np.sum(sig))
    bkgfit.append(np.sum(b))

ax.plot(bcent,np.asarray(sigfit))
ax.plot(bcent,np.asarray(bkgfit)*0.03)
plt.show()
np.savetxt('result'+str(maxa)+'.dat',zip(bcent,dndo,dndo_unc,sigfit,bkgfit))
