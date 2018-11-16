import numpy as np
import sys
from fns import * 

dt = float(sys.argv[3])
N=int(sys.argv[1])
M=int(sys.argv[2])
seed=int(sys.argv[4])
eps=float(sys.argv[5]) 

meth=str(sys.argv[6])

NBINS = 11
np.random.seed(seed)
twopi = np.pi*2.0 
sq2dt=np.sqrt(2*dt) 

x = np.random.rand( M , 1 ) * twopi 
y = np.random.rand( M , 1 ) * twopi 
px = np.random.randn( M,1)
py = np.random.randn(M,1)

if (meth=='euler'):
    fn_upd = fn_euler
if (meth=='hummer'):
    fn_upd = fn_hummer
if (meth=='baoab_im'):
    fn_upd = fn_baoab_im
    px,py=fn_InitMomenta_baoab([x,y],[px,py],dt,sq2dt,eps,fn_A_im)
if (meth=='baoab_rk4'):
    fn_upd = fn_baoab_rk4
    px,py=fn_InitMomenta_baoab([x,y],[px,py],dt,sq2dt,eps,fn_A_rk4)

t = 0 

hh,_,_2 = np.histogram2d( x.ravel()% twopi,y.ravel()% twopi, bins=NBINS,range=[[0,twopi],[0,twopi]] ) 
print str(t) + " " + " ".join(map(str,hh.ravel()))

for ii in xrange(N):
    x,y,px,py = fn_upd([x,y],[px,py],dt,sq2dt,eps) 
    t += dt
    hh,_,_2 = np.histogram2d( x.ravel()% twopi,y.ravel()% twopi, bins=NBINS,range=[[0,twopi],[0,twopi]] ) 
    print str(t) + " " + " ".join(map(str,hh.ravel())) 

