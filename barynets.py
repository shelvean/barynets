"""
Contains routines for handling computations with
barycentric  coordinates and Bernstein-Bezier polynomials

by Shelvean Kapita, 2022

"""
import numpy as np
from scipy.special import gamma, comb, roots_jacobi
from ismember import ismember
from scipy.sparse import eye,\
    csr_matrix, kron, spdiags, identity
from smesh import meshdata, meshnodes,\
    areas, normals
from scipy.sparse.linalg import spsolve

def bary(v1,v2,v3,x,y):
# =============================================================================
#     # computes the barycentric coordinates of x,y wrt triangle [v1,v2,v3]
# =============================================================================
    n = x.size
    Ones = np.ones(shape=(1,n))
    a = [1,1,1]
    b = [v1[0],v2[0],v3[0]]
    c = [v1[1],v2[1],v3[1]]
    A = np.vstack((a,b,c))
    xx = x.ravel()
    yy = y.ravel()
    r = np.vstack((Ones,xx,yy))
    lam = np.linalg.solve(A,r)
    l1 = lam[0,:]
    l2 = lam[1,:]
    l3 = lam[2,:]
    lam1 = l1.reshape(x.shape)
    lam2 = l2.reshape(x.shape)
    lam3 = l3.reshape(x.shape)
    return lam1, lam2, lam3

def gradbary(v,t,d):
    
# =============================================================================
#     # gradients of the barycentric coordinates
#     # see Ainsworth, Andriamaro, Davydov SISC 2011 for formula
# =============================================================================
    a = np.abs(areas(v,t)) # areas of triangles
    n = a.size # number of triangles in mesh
    N,L = normals(v,t) # normals and lengths of edges of mesh
    u = np.ones((3,1))
    aT = np.multiply.outer(a,u).reshape(n,3) 
    vec = -L/(2*aT) 
    u1 = np.ones((2,1))
    bT = np.multiply.outer(vec,u1).reshape(n,3,2)
    g = bT*N
    return g # gradients of the barycentric coordinates
    

def domainpnts(v1,v2,v3,d):
# =============================================================================
#     # computes the domain points of triangle [v1,v2,v3]
# =============================================================================
    I, J, K = indices(d)
    x = (I*v1[0]+J*v2[0]+K*v3[0])/d
    y = (I*v1[1]+J*v2[1]+K*v3[1])/d
    return x, y

def indices(d):
# =============================================================================
#     # order of the domain point indices
# =============================================================================
    m = int((d+1)*(d+2)/2)
    I = np.zeros(shape=(m,1),dtype=np.int32).flatten()
    J = np.copy(I)
    K = np.copy(I)
    c = 0
    for j in range(d,-1,-1):
        I[c:(c+j+1)] = range(j,-1,-1)
        J[c:(c+j+1)] = range(0,j+1)
        K[c:(c+j+1)] = (d-j)*np.ones(shape=(j+1,1),dtype=np.int32).flatten()
        c = c+j+1
    return I, J, K


def matrix(d):
# =============================================================================
#     # computes inner product matrix
#     # this is a matrix of the Bernstein-Bezier basis functions
#                  evaluated at domain points
# =============================================================================
    m = int((d+1)*(d+2)/2)
    I,J,K = indices(d)
    One = np.ones(shape=(m,m),dtype=np.int32)
    IM = np.diag(I) @ One
    JM = np.diag(J) @ One
    KM = np.diag(K) @ One
    a = (IM/d)**(IM.T)
    b = (JM/d)**(JM.T)
    c = (KM/d)**(KM.T)
    Mat = a*b*c
    IF = gamma(I+1)
    JF = gamma(J+1)
    KF = gamma(K+1)
    r = np.math.factorial(d)*One
    s = np.diag(1/(IF*JF*KF))
    A = r @ s
    Mat = A*Mat
    return Mat

def bnet(v,t,d,f,*args):
    
# =============================================================================
#     # computes bnet coefficients of f 
#     # implementation uses tensor products
# =============================================================================
    
    n = t.shape[0]
    x,y,tr = meshnodes(v,t,d) 
    if len(args) > 0:
        a = args[0]
        b = f(x,y,a)
        if type(b)==tuple:
            if args[1]==0:
                b = b[0]  # function
            elif args[1]==1:
                b = b[1]  # x-derivative
            elif args[1]==2:
                b = b[2]  # y-derivative
    else:
        b = f(x,y)
    M = matrix(d)
    I = identity(n).tocsr()
    Mat = kron(I,M).tocsr()
    c = spsolve(Mat,b)
    return c.reshape(c.size,1)

def linecoeffs(d,g,v1,v2,*args):
# =============================================================================
#     # bnet coeffs of g over the line [v1,v2]
# =============================================================================
    Mat = buildl(d)
    I = np.arange(d+1)
    J = d - I
    x = (J*v1[0]+I*v2[0])/d
    y = (J*v1[1]+I*v2[1])/d
    if len(args)==1:
        a = args[0]
        b,gx,gy=g(x,y,a)
    else:
        b = g(x,y)
    c = np.linalg.solve(Mat,b)
    return c[:,np.newaxis]

def tcord(v1,v2,v3,v):
# =============================================================================
#     # computes the directional coordinates of a vector v
# =============================================================================
    z = np.array([0])
    l1, l2, l3 = bary(v1,v2,v3,v[0],v[1])
    a1, a2, a3 = bary(v1,v2,v3,z,z)
    l1 = l1 - a1
    l2 = l2 - a2
    l3 = l3 - a3
    return l1, l2, l3

def locate(i,j,k,I,J,K):
# =============================================================================
#     # finds positions of sub-indices
# =============================================================================
    ijk = np.vstack((i,j,k)).T
    IJK = np.vstack((I,J,K)).T
    iloc, idx = ismember(ijk,IJK,method='rows')
    return idx

def degraise(d):
# =============================================================================
#     # degree raising matrix for one element
# =============================================================================
    m = int((d+1)*(d+2)/2)
    n = int((d+2)*(d+3)/2)
    i,j,k = indices(d)
    I,J,K = indices(d+1)
    u = i+1
    v = j+1
    w = k+1
    idx1 = locate(u,j,k,I,J,K)
    idx2 = locate(i,v,k,I,J,K)
    idx3 = locate(i,j,w,I,J,K)
    Dg = np.zeros(shape=(n,m))
    Ind = np.vstack((idx1,idx2,idx3))
    for k in range(m):
        x = np.zeros(shape=(3,1)).flatten()
        x[0] = u[k]
        x[1] = v[k]
        x[2] = w[k]
        Dg[Ind[:,k],k]=x
    Dg = Dg/(d+1)
    return csr_matrix(Dg)

def globaldegraise(t,d):
# =============================================================================
#     # global degree raising matrix for triangulation
# =============================================================================
    Dg = degraise(d)
    n = t.shape[0]
    I = eye(n)
    return kron(I,Dg).tocsr()

def dirder(bcoeff,l1,l2,l3):
# =============================================================================
#     # computes the directional derivative of a polynomial
#              with bnet coefficients bcoeff in a direction
#     # given by directional coordinates l1, l2, l3
# =============================================================================
    m = bcoeff.shape[0]
    d = degree(m)
    dbcoeff = d*de_cast_step(l1,l2,l3,bcoeff)
    return dbcoeff

def degree(m):
# =============================================================================
#     # returns the degree of a polynomial
#      with m domain points
# =============================================================================
    d = int((-3+np.sqrt(8*m+1))/2)
    return d

def de_cast_step(l1,l2,l3,bcin):
# =============================================================================
#     # one step de Casteljau algorithm
# =============================================================================
    m = bcin.shape[0]
    d = degree(m)
    n = l1.size
    I,J,K = indices(d)
    i,j,k = indices(d-1)
    idx1 = locate(i+1,j,k,I,J,K)
    idx2 = locate(i,j+1,k,I,J,K)
    idx3 = locate(i,j,k+1,I,J,K)
    if bcin.shape[1]==1: # if bcoeffs form a column vector
        d = degree(m)
        a = np.outer(bcin[idx1],l1)
        b = np.outer(bcin[idx2],l2)
        c = np.outer(bcin[idx3],l3)
        bcout = a+b+c
    else:
        if n==1:
            dimn = bcin[idx1,:].shape
            a = np.outer(bcin[idx1,:],l1)
            b = np.outer(bcin[idx2,:],l2)
            c = np.outer(bcin[idx3,:],l3)
            a = a.reshape(dimn)
            b = b.reshape(dimn)
            c = c.reshape(dimn)
            bcout = a+b+c
        else:
            CL1 = spdiags(l1.flatten(),0,n,n)
            CL2 = spdiags(l2.flatten(),0,n,n)
            CL3 = spdiags(l3.flatten(),0,n,n)
            spdiags(l2.flatten(),0,n,n)
            spdiags(l3.flatten(),0,n,n)
            a = CL1.dot(bcin[idx1,:].T).T
            b = CL2.dot(bcin[idx2,:].T).T
            c = CL3.dot(bcin[idx3,:].T).T
            bcout = a+b+c
    return bcout

def globaldirder(c,v,t,dv):
# =============================================================================
#     # computes the global directional derivative of c in direction dv
#     # c is a vector of bnet coefficients over triangulation [v,t]
# =============================================================================
    k = t.shape[0]
    m = int((c.size)/k)
    d = degree(m)
    md = int((d*(d+1)/2))
    dc = np.zeros(shape=(k*md,1))
    for j in range(k):
        v1 = v[t[j,0],:]
        v2 = v[t[j,1],:]
        v3 = v[t[j,2],:]
        l1,l2,l3 = tcord(v1,v2,v3,dv)
        dc[j*md:(j+1)*md] = dirder(c[j*m:(j+1)*m],l1,l2,l3)
    return dc


def xder(s):
# =============================================================================
#     # returns bnet coeffs of the x-derivative of c over triangulation [v,t]
#     # s is a list: s[0]=v, s[1]=t, s[2]=c
#     # v ~ vertices, t ~ triangles, c ~ bnet coeffs
# =============================================================================
    v = s[0]
    t = s[1]
    c = s[2]
    u = np.array([1,0])
    if len(c)==1:
        cx = 0
    else:
        cx = globaldirder(c,v,t,u)
    return [v,t,cx]

def yder(s):
# =============================================================================
#     # returns bnet coeffs of the y-derivative of c over triangulation [v,t]
#     # s is a list: s[0]=v, s[1]=t, s[2]=c
#     # v ~ vertices, t ~ triangles, c ~ bnet coeffs
# =============================================================================
    v = s[0]
    t = s[1]
    c = s[2]
    u = np.array([0,1])
    if len(c)==1:
        cy = 0
    else:
        cy = globaldirder(c,v,t,u)
    return [v,t,cy]

def bnetval(s,x,y):
# =============================================================================
#     this is an older and slower implementation of gevalf below
#      evaluates bnet coeffs over the triangulation [v,t] at points [x,y]
#     # s[0]=v; s[1]=t; s[2]=c
#     # x, y arrays of shape (n,1)
#     # if x,y are of size (n,) redefine x = x[:,np.newaxis]
#     # also see gevalf
# =============================================================================
    v = s[0]
    t = s[1]
    c = s[2]
    k = t.shape[0]
    z = np.nan*np.ones(shape=x.shape)
    m = c.shape[0]//k
    tol = 100*np.finfo(float).eps
    for j in range(k):
        v1 = v[t[j,0],:]
        v2 = v[t[j,1],:]
        v3 = v[t[j,2],:]
        l1,l2,l3 = bary(v1,v2,v3,x,y)
        n = l1.size
        l1,l2,l3 = l1.reshape(n,1),l2.reshape(n,1),l3.reshape(n,1)
        I = np.argwhere((l1>=-tol)&(l2>=-tol)&(l3>=-tol))
        idx = I[:,0]
        z[idx] = loceval(l1[idx],l2[idx],l3[idx],c[j*m:(j+1)*m])
    return z

def gevalf(s,q):
# =============================================================================
#     * evaluates a polynomial in Bnet form
#     * faster alternative to bnetval(s,x,y)
#     * see Ainsworth, Andriamaro, Davydov SISC 2011 Algorithms 1, 2
#     * evaluation uses Duffy transformation from unit square to triangle
#     * n is the number of evaluation points in each triangle
#     * INPUTS: s = [v,t,c], a list
#               v=s[0] - vertices,
#               t=s[1] - triangles,
#               c=s[2] - bnet coefficients 
#               q is number of Stroud nodes on an interval, i.e.
#                q*q points in each triangle
#     * OUTPUT:
#                 u is (ntri*q*q,1) vector of values
#                                 of the solution at Stroud points on mesh
# =============================================================================
    # utilities for Gauss-Jacobi points
    t = s[1]
    c = s[2]
    ntri = t.shape[0]
    [t11,w11] = roots_jacobi(q,1,0)
    [t22,w22] = roots_jacobi(q,0,0)
    t1 = 0.5+0.5*t11
    t2 = 0.5+0.5*t22
    tx,ty = np.meshgrid(t1,t2)
    m = int(c.shape[0]/ntri) # number of domain points in a triangle
    d = int((-3+np.sqrt(8*m+1))/2) # polynomial degree
    I,J,K = indices(d)
    u = np.zeros(shape=(ntri,q,q)) # initialize the evaluation tensor
    combs = comb(d,I)*comb(d-I,J) # pre-compute the binomial coefficients
    nn = np.arange(ntri)
    for j in range(m): # loop over the BB indices
        u1 = (tx**I[j])*((1-tx)**(d-I[j]))
        u2 = (ty**J[j])*((1-ty)**(K[j]))
        u3 = u1*u2
        idx = j+m*nn
        cloc = combs[j]*c[idx].flatten() # computes on all triangles at once
        uloc = np.multiply.outer(cloc,u3)
        u += uloc
    return u.reshape(ntri*q*q,1) # function value at all Stroud nodes

def stroudnodes(v,t,q):
    
# =============================================================================
#     # tensorized computation of Stroud nodes on the mesh
#    # computes on all triangles at once as a tensor multiplication
# =============================================================================
    
    vt = v[t[:,:],:]
    [t11,w11] = roots_jacobi(q,1,0) # Gauss Jacobi nodes on [-1,1]
    [t22,w22] = roots_jacobi(q,0,0)
    t1 = 0.5 + 0.5*t11 # transform Gauss-Jacobi nodes to [0,1]
    t2 = 0.5 + 0.5*t22
    tx,ty = np.meshgrid(t1,t2)
    u = tx # barycentric coordinates at the Gauss-Jacobi nodes on square
    v = ty*(1-u)
    w = 1-u-v
    z = np.multiply.outer(vt[:,0,:],u)+np.multiply.outer(vt[:,1,:],v)+\
        np.multiply.outer(vt[:,2,:],w)
    x = z[:,0,:].flatten() # corresponding points on the mesh
    y = z[:,1,:].flatten()
    return x.reshape(x.size,1),y.reshape(y.size,1)

def bbint(v,t,d):
# =============================================================================
#     # vector of integrals of BB basis functions
#     # as scaling for the obstacle Lagrange multipliers
#     # int_{Omega} B^d_{ijk} dxdy, i+j+k=d
# =============================================================================
    m = int(0.5*(d+1)*(d+2))
    a = np.abs(areas(v,t)) # vector of areas for triangulation
    v = np.ones(shape=(m,1))
    u = (1/m)*np.multiply.outer(a,v).flatten()
    return u


def moment(v,t,d,f,q,*args):

# =============================================================================
#     # compute Bernstein-Bezier moments using Stroud conical quadrature
#     # see Ainsworth, Andriamaro and Davydov, SISC 2011
#     # computes int_T f(x,y)B^d_{ijk} dxdy for all i+j+k=d
#     # also f = M*c, where c = bnet(v,t,d,f) computes the moment
# =============================================================================

    a = np.abs(areas(v,t))
    ntri = t.shape[0]
    qs = q*q
    x,y = stroudnodes(v,t,q)
    if len(args) > 0:
        casenum = args[0]
        Fev = f(x,y,casenum)
        if type(Fev)==tuple:
            if args[1]==0:
                Fev = Fev[0]  # function
    else:
        Fev = f(x,y)
    m = int(0.5*(d+1)*(d+2))
    nn = np.arange(Fev.size)
    idx = np.floor(nn/qs).astype(int)
    a1 = a[idx]
    a1 = a1.reshape(a1.size,1)
    F = (Fev*a1).reshape(ntri,q,q) # tensorize the function evaluation
    I,J,K = indices(d)
    comb1 = comb(d,I) # pre-compute the binomial coefficients
    comb2 = comb(d-I,J)
    u = np.zeros(shape=(ntri,m))
    [t11,w11] = roots_jacobi(q,1,0)
    [t22,w22] = roots_jacobi(q,0,0)
    t1 = 0.5+0.5*t11
    t2 = 0.5+0.5*t22
    w1 = 0.5*w11
    w2 = 0.5*w22
    tx,ty = np.meshgrid(t1,t2)
    for k in np.arange(m): # loops through the domain point indices
        uloc = np.zeros(shape=(ntri,1))
        v1 = w1*comb1[k]*(t1**I[k])*((1-t1)**(d-I[k]))
        v21 = (t2**(J[k]))*((1-t2)**(K[k]))
        v2 = w2*comb2[k]*v21
        w = np.outer(v2,v1).reshape(1,q,q)
        uloc = np.tensordot(F,w,axes=([1,2],[1,2]))
        u[:,k] = uloc.flatten()
    return u.reshape(u.size,1)



def integrate(v,t,g,q):
# =============================================================================
# # computes the integral of a function g on the mesh [v,t]
#     int_{Omega} f(x,y) dxdy
# # uses Stroud conical quadrature at Gauss-Jacobi nodes
#  see Ainsworth, Andriamaro, Davydov, SISC 2011
# test:  g = lambda x,y: 0.*x + 1.0 
#           q = 5
#           val = integrate(v,t,g,q)
#          check that val = area of the region
# =============================================================================
    a = np.abs(areas(v,t))
    [t1,w1] = roots_jacobi(q,1,0) # Gauss Jacobi nodes/weights on [-1,1]
    [t2,w2] = roots_jacobi(q,0,0)
    wx = 0.5*w1  # Gauss Jacobi weights on [0,1]
    wy = 0.5*w2
    x,y = stroudnodes(v,t,q) # Stroud nodes on the mesh
    f = g(x,y) # evaluate g on the Stroud nodes
    qs = q*q # number of points per triangle
    w = np.outer(wy,wx).reshape(1,q,q)
    ntri = int(f.shape[0]/qs) # number of triangles
    nn = np.arange(f.size)
    idx = np.floor(nn/qs).astype(int) # find triangle containing Stroud point
    a1 = a[idx] # find area of triangle containing Stroud point
    a1 = a1.reshape(a1.size,1)
    ze = (a1*f).reshape(ntri,q,q)
    v = np.sum(np.tensordot(ze,w,axes=([1,2],[1,2]))) # Stroud quadrature
    return v 
    
def loceval(l1,l2,l3,bcoeff):
# =============================================================================
#     # evaluate polynomial with bnet coefficients bcoeff
#     # over points given in barycentric coordinates by l1,l2,l3
# =============================================================================
    m = bcoeff.size
    d = degree(m)
    for j in range(d):
        bcoeff = de_cast_step(l1,l2,l3,bcoeff)
    return bcoeff.T

def build(m,*args):
# =============================================================================
#     # inner product matrix over triangle
#     inner product matrix can be rectangular
# =============================================================================
    if len(args)==0:
        n = m
    else:
        n = args[0]
    I,J,K = indices(m)
    I1,J1,K1 = indices(n)
    r = int(0.5*(n+1)*(n+2))
    s = int(0.5*(m+1)*(m+2))
    A = np.zeros((s,r))
    a = comb(m+n,m)
    for j in range(r):
        w1 = comb(I+I1[j],I)/a
        w2 = w1*comb(J+J1[j],J)
        w3 = w2*comb(K+K1[j],K)
        A[:,j] = w3
    return A

def build2(d):
    
# =============================================================================
#  # construct inner product matrix needed for computing 
#                the stiffness matrix
# =============================================================================
    
    I,J,K = indices(d)
    m = int(0.5*(d+1)*(d+2))
    C = np.zeros((9,m,m))
    a = comb(2*d-2,d-1)*comb(2*d,2)

    for k in range(3):
        for l in range(3):
            idx = l+3*k
            for j in range(m):
                w1 = comb(I+I[j]-int(k==0)-int(l==0),\
                          I-int(k==0),exact=False)
                w2 = w1*comb(J+J[j]-int(k==1)-int(l==1),\
                             J-int(k==1),exact=False)
                w3 = w2*comb(K+K[j]-int(k==2)-int(l==2),\
                             K-int(k==2),exact=False)
                C[idx,:,j] = w3
    D = C/a
    return D

def build3(d3,d2,d1):
    
# =============================================================================
#     # inner product matrix required to compute the weighted mass matrix
#     # d3 is the degree associated with the weighting function
# =============================================================================
    b = comb(d1+d2+d3,d3)
    I,J,K = indices(d1)
    I1,J1,K1 = indices(d2)
    I2,J2,K2 = indices(d3)
    m = I.size
    n = I1.size
    s = I2.size
    C = np.zeros((s,m,n))
    for j in np.arange(m):
        for k in np.arange(n):
            w1 = comb(I2+I[j]+I1[k],I2)/b
            w2 = w1*comb(J2+J[j]+J1[k],J2)
            w3 = w2*comb(K2+K[j]+K1[k],K2)
            C[:,j,k] = w3
    B = build(d1,d2)
    u = np.ones((1,s))
    B = np.multiply.outer(u,B).reshape(s,m,n)
    C = B*C
    return C

def build4(d1,d2):
    
# =============================================================================
#     # inner product matrices required to compute
#     # the weighted stiffness matrix
# =============================================================================
    
    
    s = int(0.5*(d1+1)*(d1+2))  # domain points associated with the weight
    m = int(0.5*(d2+1)*(d2+2)) # domain points associated with test funcs
    I,J,K = indices(d2)
    I1,J1,K1 = indices(d1)
    B = np.zeros((9,s,m,m))
    for k in range(3):
        for l in range(3):
            idx = l+3*k
            for j in range(m):
                for i in range(m):
                    u1 = comb(I[j]+I[i]-int(k==0)-int(l==0),\
                              I[j]-int(k==0), exact=False)
                    u2 = u1*comb(J[j]+J[i]-int(k==1)-int(l==1),\
                              J[j]-int(k==1), exact=False)
                    u3 = u2*comb(K[j]+K[i]-int(k==2)-int(l==2),\
                              K[j]-int(k==2), exact=False) 
                    w1 = comb(I[i]+I[j]+I1-int(k==0)-int(l==0),\
                              I1,exact=False)
                    w2 = w1*comb(J[i]+J[j]+J1-int(k==1)-int(l==1),\
                              J1,exact=False)
                    w3 = w2*comb(K[i]+K[j]+K1-int(k==2)-int(l==2),\
                              K1,exact=False)
                    v3 = u3*w3
                    B[idx,:,j,i] = v3
    a = comb(2*d2-2,d2-1)
    b = comb(2*d2+d1-2,d1)
    c = comb(2*d2+d1,2)
    return B/(a*b*c) 
            


def buildl(d):
# =============================================================================
#     # inner product matrix over a line
# =============================================================================
    if d==0:
        Mat = 1.
    elif d==1:
        Mat = np.eye(2)
    else:
        I = np.arange(d+1)
        J = d - I
        m = (d+1)
        One = np.ones(shape=(m,m))
        IM = np.diag(I)@One
        JM = np.diag(J)@One
        dinv = 1/d
        r = (dinv*IM)**(IM.T)
        s = (dinv*JM)**(JM.T)
        Mat = r*s
        IF = gamma(I+1)
        JF = gamma(J+1)
        A = np.math.factorial(d)*One @ np.diag(1/(IF*JF))
        Mat = A*Mat
    return Mat


