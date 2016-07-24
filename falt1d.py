import matplotlib.pyplot as plt
import numpy as np
import pyfits
from scipy.integrate import simps
from matplotlib.ticker import NullFormatter, MaxNLocator
import scipy.optimize as opt

# added a bit of documentation
# This should only appear in DH branch

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




# Define a function to make the ellipses
def ellipse(ra,rb,ang,x0,y0,Nb=100):
    xpos,ypos=x0,y0
    radm,radn=ra,rb
    an=ang
    co,si=np.cos(an),np.sin(an)
    the=np.linspace(0,2*np.pi,Nb)
    X=radm*np.cos(the)*co-si*radn*np.sin(the)+xpos
    Y=radm*np.cos(the)*si+co*radn*np.sin(the)+ypos
    return X,Y


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
    return(np.arccos(resu))


# read in the data
psf_fits=pyfits.open('./data/psf_P8R2_SOURCE_V6_PSF.fits')
dat=pyfits.open('./data/source_psf2_3.fits')	

#cut on evclass -> nur source (evclass=128)--- cut on evtype -> nur psf2/3 (evtype=16+32=48)

# define some constants
twopi=np.arctan(1.)*8.
pi=twopi/2.
d2r=pi/180.
r2d=180./pi

# position of source
RAc=166.114			#position des zentrums
DECc=38.2088


cosTlist=np.cos(np.array(dat[1].data.field('THETA   '))*pi/180)	#liste der photonen neigungswinkel zum detektor (fuer thetabin) in cos(theta)
Elist=np.array(dat[1].data.field('ENERGY '))			#liste der photonenenergien in MeV
Ralist= np.array(dat[1].data.field('RA'))
Declist= np.array(dat[1].data.field('DEC'))
evtype = np.array(dat[1].data.field('EVENT_TYPE')[:,26])

drad = dangle(RAc*np.ones(Ralist.size),Ralist,DECc*np.ones(Declist.size),Declist)


# transformation:
Ralist = (Ralist-RAc)/np.cos(DECc*pi/180.)
Declist= Declist - DECc

# maximum cutout angle
maxa = 0.5

# adding bootstrapped background
bcosTlist = cosTlist[drad>(d2r*maxa)]
bElist    = Elist[drad>(d2r*maxa)]
bevtype   = evtype[drad>(d2r*maxa)]
bRalist   = Ralist[drad>(d2r*maxa)]
bDeclist  = Declist[drad>(d2r*maxa)]
#shuffling separation angle and phase
# uniform in costheta 
bdradcos     = np.random.uniform(np.cos(maxa*d2r),1,bElist.size)
bdrad        = np.arccos(bdradcos)
bdradsin     = np.sqrt(1.-bdradcos*bdradcos)
bphase    = np.random.rand(bElist.size)*twopi
# transforming 
# u,v,w -> u',v',w' 
u= np.cos(bphase)*bdradsin
v= np.sin(bphase)*bdradsin
w= bdradcos



th = pi/2.-DECc*d2r
ph = RAc * d2r

cph = np.cos(ph)
sph = np.sin(ph)
cth = np.cos(th)
sth = np.sin(th)

rotz  = np.matrix  ([ [cph,  -sph ,  0. ], 
                      [sph,   cph ,  0. ],
                      [ 0.,   0.  ,  1. ] ])

roty = np.matrix  ([ [cth,   0.  , sth ],
                      [0. ,   1.  , 0.  ],
                      [-sth,   0.  , cth ] ])



# combined rotation around y and then around z
rot = np.dot(rotz,roty)

# rotate all vectors
tt = np.asarray(map(lambda U,V,W: np.dot(rot,np.asarray([U,V,W])),u,v,w))
# extract the angles
up=tt[:,0,0]
vp=tt[:,0,1]
wp=tt[:,0,2]
omwp = np.sqrt(1.-wp*wp)

phip  = np.arctan2(vp/omwp,up/omwp)*r2d
thp   = np.arcsin(wp)*r2d              # declination

#plt.plot(phip,thp,'g.')
#plt.show()








cosTlist = cosTlist[drad<(d2r*maxa)]
Elist  = Elist[drad<(d2r*maxa)]
Ralist = Ralist[drad<(d2r*maxa)]
Declist = Declist[drad<(d2r*maxa)]
drad = drad[drad<(d2r*maxa)]
evtype = evtype[drad<(d2r*maxa)]


bck=0
evtype  = np.append(evtype,bevtype[:bck])
Declist = np.append(Declist,bDeclist[:bck])
Ralist  = np.append(Ralist, bRalist[:bck])
Elist   = np.append(Elist,bElist[:bck])
cosTlist= np.append(cosTlist,bcosTlist[:bck])
drad    = np.append(drad,bdrad[:bck])




print str(drad.size)+' Photons within '+str(maxa)+' degrees'

xlims=[min(Ralist),max(Ralist)]
ylims=[min(Declist),max(Declist)]

left,width=0.12,0.55
bottom,height=0.12,0.55
bottom_h=left_h=left+width+0.02

rect_sky = [left,bottom,width,height]
rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram

fig = plt.figure(1, figsize=(9.5,9))

axSky = plt.axes(rect_sky)
axHistx = plt.axes(rect_histx) # x histogram
axHisty = plt.axes(rect_histy) # y histogram

# remove the zero
nullfmt = NullFormatter()
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

xmin = min(xlims)
xmax = max(xlims)
ymin = min(ylims)
ymax = max(ylims)

# Define the number of bins
nxbins = 50
nybins = 50
nbins = 100

xbins = np.linspace(start = xmin, stop = xmax, num = nxbins)
ybins = np.linspace(start = ymin, stop = ymax, num = nybins)
xcenter = (xbins[0:-1]+xbins[1:])/2.0
ycenter = (ybins[0:-1]+ybins[1:])/2.0
aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)


H, xedges,yedges = np.histogram2d(Declist,Ralist,bins=(ybins,xbins))
X = xcenter
Y = ycenter
Z = H

cax = (axSky.imshow(H, extent=[xmin,xmax,ymin,ymax],
           interpolation='nearest', origin='lower',aspect=aspectratio))

xcenter = np.mean(Ralist)
ycenter = np.mean(Declist)
rx = np.std(Ralist)
ry = np.std(Declist)

ang=0
X,Y=ellipse(rx,ry,ang,xcenter,ycenter)
axSky.plot(X,Y,"k:",ms=1,linewidth=2.0)
axSky.annotate('$1\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                       textcoords='offset points', horizontalalignment='right',
                       verticalalignment='bottom',fontsize=25,color='white')

axSky.set_xlabel('Delta Ra',fontsize=25)
axSky.set_ylabel('Delte Dec',fontsize=25)

ticklabels = axSky.get_xticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('serif')
 
ticklabels = axSky.get_yticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('serif')

#Set up the plot limits
axSky.set_xlim(xlims)
axSky.set_ylim(ylims)
 
#Set up the histogram bins
xbins = np.arange(xmin, xmax, (xmax-xmin)/nbins)
ybins = np.arange(ymin, ymax, (ymax-ymin)/nbins)
 
#Plot the histograms
axHistx.hist(Ralist, bins=xbins, color = 'blue')
axHisty.hist(Declist, bins=ybins, orientation='horizontal', color = 'red')
 
#Set up the histogram limits
axHistx.set_xlim( min(Ralist), max(Ralist) )
axHisty.set_ylim( min(Declist), max(Declist) )
 
#Make the tickmarks pretty
ticklabels = axHistx.get_yticklabels()
for label in ticklabels:
    label.set_fontsize(12)
    label.set_family('serif')
 
#Make the tickmarks pretty
ticklabels = axHisty.get_xticklabels()
for label in ticklabels:
    label.set_fontsize(12)
    label.set_family('serif')
 
#Cool trick that changes the number of tickmarks for the histogram axes
axHisty.xaxis.set_major_locator(MaxNLocator(4))
axHistx.yaxis.set_major_locator(MaxNLocator(4))
plt.show()


# calculate the likelihood as a function of background-fraction
# assuming a point source at nominal position

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
prob_pt = np.asarray(map(lambda dist,sc,gc,st,gt,no,c_0,c_1,b,ener: p(dist,sc,gc,st,gt,no,ener,c_0,c_1,b),
                  drad,sigc,gamc,sigt,gamt,N,c0,c1,beta,Elist))
sp = np.asarray(map(lambda ener,c_0,c_1,b: Sp(ener,c_0,c_1,b),Elist,c0,c1,beta))
# normalize the sum  
print np.sum(prob_pt)
# normalized pdf to the full solid angle and differential in "x" -> d\theta/sp
sig = prob_pt/npred/sp/2./twopi
b   = np.ones(npred)/npred
plt.plot(drad*180./pi,np.log(sig),'.')
plt.plot(drad*180./pi,np.log(b),'g.')
#plt.hist(drad*180./pi,bins=100,weights=np.log(sig))
#plt.hist(drad*180./pi,bins=100,weights=np.log(b),alpha=.3)
plt.show()


x=np.arange(0.30,1.00,0.01)
# diagnostic plot - 
# log(mu) for signal as a function of fraction
#plt.plot(x,np.asarray(map(lambda f: np.sum(np.log(sig*f)),x)),'r')
# log(mu) for background as a function of 1-fraction
#plt.plot(x,np.asarray(map(lambda f: np.sum(np.log(b*(1.-f))),x)),'g')

# calculate the likelihood - vary the only free parameter - fraction of background
# mu = signal * f + (1-f)*background
# 

def neglikelihood(x):
    x=max(1e-30,min(1.-1e-8,x))
    res = -1. * np.sum(np.log(sig*x + b*(1.-x)))
    print x,res
    return(res)


likeli  = lambda f: -1.*np.sum( np.log(sig*f + b*(1.-f)))
likeli0 = np.asarray(map(lambda f: np.sum( np.log(sig*f + b*(1.-f)) ),x))
print opt.minimize(neglikelihood,0.80,method='Nelder-Mead',options={'disp':True})
print neglikelihood(0.83)
print likeli(0.83)
# maximize likelihood

plt.plot(x,likeli0)
plt.show()

