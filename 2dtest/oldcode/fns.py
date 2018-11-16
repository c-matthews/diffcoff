import numpy as np

def fn_F(x,y):

    Fx = -np.cos(x) * np.cos(y)
    Fy = np.sin(x) * np.sin(y)

    return Fx,Fy

def fn_D(x,y,eps):

    eps2 = eps*eps
    ss = np.sin(2*x+y)
    cc = np.cos(2*x+y)
    ccc = np.cos(4*x+2*y)

    Da = 1+eps2*(ss*ss)
    Db = eps*(2-eps*cc)*ss
    Dc = 1+eps2-2*eps*cc

    Da_dx = 4*eps2*cc*ss
    Db_dx = -2*eps*(-2*cc+eps*ccc)
    Db_dy = -eps*(-2*cc+eps*ccc)
    Dc_dy = 2*eps*ss

    sqDa = 1
    sqDb = eps * ss
    sqDc = 1-eps* cc

    sqDa_dx = 0 
    sqDb_dx = eps* cc * 2
    sqDb_dy = eps * cc
    sqDc_dy = eps * ss

    divD = [(Da_dx+Db_dy),(Db_dx+Dc_dy) ]
    divsqD=[(sqDa_dx+sqDb_dy),(sqDb_dx+sqDc_dy)]


    return Da,Db,Dc,sqDa,sqDb,sqDc,divD,divsqD

def fn_euler(Q,P,dt,sq2dt,eps):
    x,y = Q
    px,py = P
    fx,fy = fn_F(x,y)
    Da,Db,Dc,sqDa,sqDb,sqDc,divD,divsqD = fn_D(x,y,eps)
    DivDx,DivDy = divD

    xx = x + dt*(Da*fx+Db*fy)
    yy = y + dt*(Db*fx+Dc*fy)

    xx += dt*DivDx
    yy += dt*DivDy

    px = np.random.randn( xx.shape[0], xx.shape[1] )
    py = np.random.randn( xx.shape[0], xx.shape[1] )

    xx += sq2dt*(sqDa*px + sqDb*py)
    yy += sq2dt*(sqDb*px + sqDc*py)

    return xx,yy,px,py

def fn_hummer(Q,P,dt,sq2dt,eps):
    x,y = Q
    px,py = P
    fx,fy = fn_F(x,y)
    Da,Db,Dc,sqDa,sqDb,sqDc,divD,divsqD = fn_D(x,y,eps)
    DivDx,DivDy = divD

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

def fn_baoab_im(Q,P,odt,dt,eps):

    QQ,PP = fn_B(Q,P,dt,eps)
    QQ,PP = fn_A_im(QQ,PP,dt*0.5,eps)
    QQ,PP = fn_O(QQ,PP,dt,eps)
    QQ,PP = fn_A_im(QQ,PP,dt*0.5,eps)

    x,y = QQ
    px,py = PP

    return x,y,px,py

def fn_baoab_rk4(Q,P,odt,dt,eps):

    QQ,PP = fn_B(Q,P,dt,eps)
    QQ,PP = fn_A_rk4(QQ,PP,dt*0.5,eps)
    QQ,PP = fn_O(QQ,PP,dt,eps)
    QQ,PP = fn_A_rk4(QQ,PP,dt*0.5,eps)

    x,y = QQ
    px,py = PP

    return x,y,px,py



def fn_B(Q,P,dt,eps):
    x,y = Q
    px,py = P
    fx,fy = fn_F(x,y)
    Da,Db,Dc,sqDa,sqDb,sqDc,divD,divsqD = fn_D(x,y,eps)
    DivDx,DivDy = divD
 
    px = px + dt*(sqDa*fx + sqDb*fy)
    py = py + dt*(sqDb*fx + sqDc*fy)

    return [x,y],[px,py]

def fn_O(Q,P,dt,eps):
    x,y = Q
    px,py = P

    px = np.random.randn( px.size, 1 )
    py = np.random.randn( py.size, 1 )

    return [x,y],[px,py]

def fn_A_im(Q,P,dt,eps,maxsteps=100,tol=1e-3 ):
    x,y = Q
    px,py = P

    # Implicit midpoint

    xx = np.copy(x)
    yy = np.copy(y)
    ppxx = np.copy(px)
    ppyy = np.copy(py)

    for ii in xrange(maxsteps):
        oxx = np.copy(xx)
        oyy = np.copy(yy)
        oppxx = np.copy(ppxx)
        oppyy = np.copy(ppyy)
        original = [x,y,px,py]
        midpt = [(x+xx)/2,(y+yy)/2,(ppxx+px)/2,(ppyy+py)/2]
        xx,yy,ppxx,ppyy = fn_Arhs( original,midpt,dt,eps )
        dq = np.abs(oxx-xx)+np.abs(oyy-yy)
        dp = np.abs(oppxx-ppxx)+np.abs(oppyy-ppyy)
        if (np.max(dq+dp)<tol):
            break
    #print ii

    return [xx,yy],[ppxx,ppyy]


def fn_Arhs( original, midpt , dt, eps ):

    ox,oy,opx,opy = original
    mx,my,mpx,mpy = midpt
    
    Da,Db,Dc,sqDa,sqDb,sqDc,divD,divsqD = fn_D(mx,my,eps)
    DivDx,DivDy = divsqD
    
    xrhs = ox + dt*(sqDa*mpx + sqDb*mpy)
    yrhs = oy + dt*(sqDb*mpx + sqDc*mpy)
    
    pxrhs = opx + dt*DivDx
    pyrhs = opy + dt*DivDy
    

    return xrhs,yrhs,pxrhs,pyrhs

def fn_Afield(x,y,px,py,eps):

    Da,Db,Dc,sqDa,sqDb,sqDc,divD,divsqD = fn_D(x,y,eps)
    DivDx,DivDy = divsqD
    
    xfield = (sqDa*px + sqDb*py)
    yfield = (sqDb*px + sqDc*py)

    pxfield = (DivDx)
    pyfield = DivDy

    return xfield,yfield,pxfield,pyfield

def fn_A_rk4(Q,P,dt,eps):
    x,y = Q
    px,py = P
    dt2 = dt*0.5
    dt6 = dt/6.0

    # RK4 integration

    f1x,f1y,f1px,f1py = fn_Afield(x,y,px,py,eps)
    f2x,f2y,f2px,f2py = fn_Afield(x+dt2*f1x,y+dt2*f1y,px+dt2*f1px,py+dt2*f1py,eps)
    f3x,f3y,f3px,f3py = fn_Afield(x+dt2*f2x,y+dt2*f2y,px+dt2*f2px,py+dt2*f2py,eps)
    f4x,f4y,f4px,f4py = fn_Afield(x+dt*f3x,y+dt*f3y,px+dt*f3px,py+dt*f3py,eps)

    xx = x + dt6*(f1x + 2*f2x + 2*f3x + f4x)
    yy = y + dt6*(f1y + 2*f2y + 2*f3y + f4y)
    ppxx = px + dt6*(f1px + 2*f2px + 2*f3px + f4px)
    ppyy = py + dt6*(f1py + 2*f2py + 2*f3py + f4py)

    return [xx,yy],[ppxx,ppyy]



def fn_InitMomenta_baoab(Q,R,odt,dt,eps,Afn):

        x,y=Q
        rx,ry=R
        
        nr1 = np.copy(rx)
        nr2 = np.copy(ry)

        x0 = np.copy(x)
        y0 = np.copy(y)
        Q0=[x0,y0]
        tol = 1e-5

        for ii in range(20):
            newQ,newP=Afn(Q0,R,dt,eps)
            Q1,P1=Afn(Q,newP,-dt,eps)
            dx = np.abs(Q0[0]-Q1[0]) + np.abs(Q0[1]-Q1[1])
            Q0 = Q1
            if (np.max(dx)<tol):
                break

        newQ,newP=Afn(Q0,[nr1,nr2],dt,eps)

        px,py = newP

        return px,py
















