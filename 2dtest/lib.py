import numpy as np
import time as tm
import matplotlib
import matplotlib.pyplot as plt

class Model:

    def __init__(self, potential , diffmatrix, sqdiffmatrix ):
    
        V,F    = potential
        D,divD     = diffmatrix
        sqD,divsqD = sqdiffmatrix
        
        
        self.fn_V = V
        self.fn_F = F

        self.fn_D = D
        self.fn_divD = divD

        self.fn_sqD = sqD
        self.fn_divsqD = divsqD


def d_dx(Z,dx,axis):

    return (np.roll(Z,-1,axis) - np.roll(Z,1,axis))/(2.*dx)

def compute_drho(rho, F, D, divD,dx   ):

    # dq = D(q) F(q) dt + div(D(q)) dt + sqrt(2 D(q)) dW
    
    x_axis = 1
    y_axis = 0
    
    Fx,Fy = F
    Da,Db,Dc = D
    divDx,divDy = divD
    
    term1 = Da*Fx + Db*Fy
    term2 = Db*Fx + Dc*Fy
    
    drho = -(d_dx( term1*rho ,dx,x_axis) + d_dx( term2*rho,dx,y_axis ))
    drho -= d_dx( divDx*rho,dx,x_axis )
    drho -= d_dx( divDy*rho,dx,y_axis )
    
    drho += d_dx(d_dx( Da*rho,dx,x_axis), dx, x_axis)
    drho += 2*d_dx(d_dx( Db*rho,dx,x_axis), dx, y_axis)
    drho += d_dx(d_dx( Dc*rho,dx,y_axis), dx, y_axis)
    

    return drho

def CheckPositiveD( D ):

    Da,Db,Dc = D

    det = Da*Dc - Db*Db

    if (np.any(det<=0)):
        return False

    return True


def SolveFokkerPlanck(model, dt, NBINS , printevery, Nprints, verbose=True   ):

    fullxx = np.linspace(0,2*np.pi,NBINS+1)
    xx = fullxx[:-1]
    dx = xx[2]-xx[1]
    dx2 = dx*dx
    
    X,Y = np.meshgrid(xx,xx)

    rho = np.ones( (NBINS,NBINS) )
    rho = rho / np.sum(rho )
    rho = rho / dx2
    sumrho = np.sum(rho)
    

    F = model.fn_F(X,Y)
    D = model.fn_D(X,Y)
    divD = model.fn_divD(X,Y)
    
    if (not CheckPositiveD(D) ):
        print('Your D has negative determinant!')
        return np.nan
    

    t = 0
    tii = 0
    printii = 0
    
    RV = np.zeros((Nprints, NBINS*NBINS+1))
    
    st = tm.time()

    while (1):

        drho = compute_drho(rho, F, D, divD,dx )

        rho += dt*drho

        t = t + dt
        tii += 1

        if ((tii % printevery)==0):
            RV[printii,0] = t
            RV[printii,1:] = np.copy(rho.ravel())
            printii += 1
            if (verbose):
                print([tii, printii,tm.time()-st,np.sum(rho) - sumrho ])
        if (printii>= Nprints):
            break

    return RV




def plot_rho(rho, idx,title=''):

    
    RV = rho[idx,1:]
    NBINS = int( np.sqrt(len(RV)))
    RV = np.reshape(RV,(NBINS,NBINS) )
    
    plt.imshow( RV ,origin='lower' )
    
    t = rho[idx,0]
    tstring = title # + '(T = ' + '{0:.4f}'.format(t) +')'
    
    plt.xticks([0,NBINS/2.,NBINS-1],[0,'$\pi$','$2\pi$'])
    plt.yticks([0,NBINS/2.,NBINS-1],[0,'$\pi$','$2\pi$'])
    plt.xlabel('x')
    plt.ylabel('y',rotation=0)
    plt.title(tstring)
    ax = plt.gca()
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.title.set_fontsize(18)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    #plt.colorbar(p)
    #plt.show()





def sample(model, method,M, dt, NBINS, printevery, Nprints, verbose=True ):

    twopi = 2*np.pi
    sq2dt = np.sqrt(2*dt)

    x = np.random.rand( M , 1 ) * twopi
    y = np.random.rand( M , 1 ) * twopi

    px = np.random.randn( M , 1 )
    py = np.random.randn( M , 1 )

    if (method=='euler'):
        fn_upd = fn_euler
    if (method=='hlm'):
        fn_upd = fn_hlm

    RV = np.zeros((Nprints, NBINS*NBINS+1))


    st = tm.time()
    printii = 0
    t = 0
    tii = 0

    while (1):

        Q = [x,y]
        P = [px,py]

        x,y,px,py = fn_upd(model,Q,P,dt,sq2dt)

        t = t + dt
        tii += 1

        if ((tii % printevery)==0):
            hh,_,_2 = np.histogram2d( x.ravel()% twopi,y.ravel()% twopi,
                                        bins=NBINS,range=[[0,twopi],[0,twopi]] )
            RV[printii,0] = t
            RV[printii,1:] = (hh.T).ravel()
            printii += 1
            if (verbose):
                print([tii, printii,tm.time()-st ])
        if (printii>= Nprints):
            break

    return RV


def fn_euler(model,Q,P,dt,sq2dt):
    x,y = Q
    px,py = P
    fx,fy = model.fn_F(x,y)
    Da,Db,Dc = model.fn_D(x,y)
    DivDx,DivDy = model.fn_divD(x,y)
    sqDa,sqDb,sqDc = model.fn_sqD(x,y)

    xx = x + dt*(Da*fx+Db*fy)
    yy = y + dt*(Db*fx+Dc*fy)

    xx += dt*DivDx
    yy += dt*DivDy

    px = np.random.randn( xx.shape[0], xx.shape[1] )
    py = np.random.randn( xx.shape[0], xx.shape[1] )

    xx += sq2dt*(sqDa*px + sqDb*py)
    yy += sq2dt*(sqDb*px + sqDc*py)

    return xx,yy,px,py



def fn_hlm(model,Q,P,dt,sq2dt):
    x,y = Q
    px,py = P
    fx,fy = model.fn_F(x,y)
    Da,Db,Dc = model.fn_D(x,y)
    DivDx,DivDy = model.fn_divD(x,y)
    sqDa,sqDb,sqDc = model.fn_sqD(x,y)

    xx = x + dt*(Da*fx+Db*fy)
    yy = y + dt*(Db*fx+Dc*fy)

    xx += (0.75*dt)*DivDx
    yy += (0.75*dt)*DivDy
    
    xx += (0.5*sq2dt)*(sqDa*px + sqDb*py)
    yy += (0.5*sq2dt)*(sqDb*px + sqDc*py)

    px = np.random.randn( xx.shape[0], xx.shape[1] )
    py = np.random.randn( xx.shape[0], xx.shape[1] )

    xx += (0.5*sq2dt)*(sqDa*px + sqDb*py)
    yy += (0.5*sq2dt)*(sqDb*px + sqDc*py)

    return xx,yy,px,py



def get_L2diff( rho1, rho2, NBINS):

    fullxx = np.linspace(0,2*np.pi,NBINS+1)
    xx = fullxx[:-1]
    dx = xx[2]-xx[1]
    dx2 = dx*dx

    nt = rho1.shape[0]
    Z = np.zeros( (nt,1) )
    
    d1 =downsample( rho1, NBINS )
    d2 =downsample( rho2, NBINS )
    for ii in range(nt):
        dd1 = d1[ii,1:]
        dd2 = d2[ii,1:]
        dist1 = dd1/np.sum(dd1)/dx2
        dist2 = dd2/np.sum(dd2)/dx2
        Z[ii,0] = np.sqrt( np.sum( dx2*(dist1-dist2)**2) )

    return Z



def downsample(Z,NBINS):

    RV = np.zeros( (Z.shape[0], 1+NBINS*NBINS ) )
    ZBINS = int( np.sqrt(Z.shape[1]-1) )
    
    subw = int(ZBINS / NBINS)
    
    for ii in range( Z.shape[0]):
        RV[ii,0] = Z[ii,0]
        X = Z[ii,1:]
        X = np.reshape(X, (ZBINS,ZBINS) )
        XX = np.zeros( (ZBINS+1,ZBINS+1) )
        XX[0:-1,0:-1] = X
        XX[-1,:-1] = X[0,:]
        XX[:-1,-1] = X[:,0]
        XX[-1,-1] = X[0,0]

        ZZ = np.zeros( (NBINS,NBINS) )
        for xx in range(NBINS):
            for yy in range(NBINS):
                x1 = xx*subw
                y1 = yy*subw
                x2 = x1 + subw+1
                y2 = y1 + subw+1
                subXX = XX[x1:x2,y1:y2]
                s1 = subXX[0,0]+subXX[-1,0]+subXX[0,-1]+subXX[-1,-1]
                s2 = np.sum(subXX[1:-1,0])+np.sum(subXX[1:-1,-1])
                s3 = np.sum(subXX[0,1:-1])+np.sum(subXX[-1,1:-1])
                s4 = np.sum(subXX[1:-1,1:-1])
                ZZ[xx,yy] = (s1+2*s2+2*s3+4*s4)/4.0


        RV[ii,1:] = ZZ.ravel()

    drho = np.abs(np.sum(RV) - np.sum(Z))
    
    if (drho>1e-5):
        print('Error downsampling, drho = ' + str(drho))

    return RV

