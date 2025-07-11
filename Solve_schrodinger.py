import numpy as np
import scipy.linalg as la
from scipy.special import legendre as leg
from scipy.special import eval_legendre as legval
from scipy.special import gamma
from scipy.interpolate import make_interp_spline
from ReadPseudoXml import parse
from CoulombWF import coul90
#==============================================================================
class SolveSchrodingerAtomic(object):
    '''
    Author: Gian Parusa
    Solves Schrodinger equation for bound-state and scattering-state atomic problem 
    using finite-element method - discrete variable representation (FEM-DVR).
    '''
#==============================================================================
    def __init__(self,grid_type,xmin,xmax,N=30,Np=7,Nu=10,Ng=10,rc=15.,pp_xml=''):
        self.grid = grid_type
        self.Ng = Ng
        
        # Determine the spacing
        # -----------------------
        if grid_type == 'linear':
            self.Ne = N
            self.xp = np.linspace(xmin,xmax,self.Ne+1)
            
        elif grid_type == 'upm':
            self.Ne = Np * Nu
            self.xp = self.upm(xmin,xmax,Np,Nu)
            
        elif grid_type == 'mixed':
            self.Ne = Np*Nu + N
            x1 = self.upm(xmin,rc,Np,Nu)[:-1]
            x2 = np.linspace(rc,xmax,N+1)
            self.xp = np.concatenate((x1,x2))
            
            
        self.Xg, self.Wg = self.getLobattos()
        self.xgrid = self.getAxis()
        self.fw, self.bw = self.getWeights()
        self.Kin = self.Kinetic()
        
        self.nonloc = False
           
        if len(pp_xml) > 0:
            self.nonloc = True
            
            pp = parse(pp_xml)
            Dij = pp.get_data('pp_dij')
            Dii = np.diag(Dij)
            rmesh = pp.get_data('pp_rmesh')
            beta = pp.get_data('pp_beta')
            lbeta = pp.llbeta
            
            self.init_nonloc(Dii, rmesh, beta, lbeta)
#==============================================================================
    def upm(self,xmin,xmax,Np,Nu):
        '''
        Universal power-mesh.
        '''
        h = (xmax - xmin) / (2**Np - 1)
        idx = np.arange(Np+1)
        Points = xmin + h*(2**idx - 1)
        
        NextPoints = np.roll(Points,-1)
        Delta = NextPoints[:Np] - Points[:Np]
        Nd = np.arange(Nu) / Nu
        
        FinalPoints = Nd[None,:]*Delta[:,None] + Points[:Np,None]
        FinalPoints = np.append(FinalPoints,Points[-1])
        
        return FinalPoints
#==============================================================================
    def getLobattos(self):
        '''
        Returns the Lobatto points and weights for the Gauss-Lobatto quadrature.
        '''
        LobattoPoints = leg(self.Ng).deriv(1).r
        LobattoPoints = sorted(LobattoPoints)
        LobattoPoints = np.insert(LobattoPoints, 0, -1.)
        
        # Lobatto weights
        val = self.Ng * (self.Ng+1) * legval(self.Ng, LobattoPoints)**2
        LobattoWeights = 2/val
        
        return LobattoPoints, LobattoWeights
#==============================================================================
    def getAxis(self):
        xp1 = np.roll(self.xp,-1)
        del1 = xp1[:self.Ne] - self.xp[:self.Ne]
        del2 = xp1[:self.Ne] + self.xp[:self.Ne]
        
        pts = del1[:,None]*self.Xg[None,:] + del2[:,None]
        pts = 0.5 * np.reshape(pts,-1)
        pts = np.append(pts,self.xp[-1])
        
        return pts
#==============================================================================
    def getWeights(self):
        xp1 = np.roll(self.xp,-1)
        dlt = xp1[:self.Ne] - self.xp[:self.Ne]

        # Forward weights
        w1 = 0.5 * self.Wg[None,:] * dlt[:,None]
        # to avoid divide-by-zero
        w1 = np.append(w1,1.)

        # Backward weights
        dlt = np.insert(dlt,0,1.)
        w2 = dlt/ self.Ng/ (self.Ng + 1)

        return w1, w2
#==============================================================================
    def LobattoPoly(self,i,m,x):
        '''
        Lobatto polynomial.
        '''
        
        val = 0.
        if i>=0 and i<self.Ne-1:
            if i!=0 or m!=0:
                if x>=self.xgrid[i * self.Ng] and x<self.xgrid[(i+1) * self.Ng]:
                    val = 1.
                    for j in range(self.Ng+1):
                        if j!=m:
                            val *= x - self.xgrid[i*self.Ng + j]
                            val /= self.xgrid[i*self.Ng + m] - self.xgrid[i*self.Ng + j]
                        
        elif i==self.Ne-1 and (m>=0 and m<=self.Ng):
            if x>=self.xgrid[i * self.Ng]:
                val = 1.
                for j in range(self.Ng+1):
                    if j!=m:
                        val *= x - self.xgrid[i*self.Ng + j]
                        val /= self.xgrid[i*self.Ng + m] - self.xgrid[i*self.Ng + j]

        return val
#==============================================================================
    def LobattoBasis(self,i,m,x):
        val = 0.
        if m==0:
            val = self.LobattoPoly(i-1,self.Ng,x) + self.LobattoPoly(i,0,x)
            val /= np.sqrt(self.bw[i] + self.fw[i*self.Ng])
            
        elif m==self.Ng:
            val = self.LobattoPoly(i,self.Ng,x) + self.LobattoPoly(i+1,0,x)
            val /= np.sqrt(self.bw[i+1] + self.fw[(i+1)*self.Ng])
            
        elif m>0 and m<self.Ng:
            val = self.LobattoPoly(i,m,x)
            val /= np.sqrt(self.fw[i*self.Ng + m])
            
        return val
#==============================================================================
    def derLobatto(self,i,m1,m2):
        '''
        Calculating the first derivative of Lobatto polynomial.
        '''
        
        val = 0.
        if i>=0 and i<self.Ne:
            if (i!=0 or m1!=0) and (i!=self.Ne or m1!=0):
                if m1!=m2:
                    val = 1.
                    for j in range(self.Ng+1):
                        if j!=m1 and j!=m2:
                            val *= self.xgrid[i*self.Ng + m2] - self.xgrid[i*self.Ng + j]
                            val /= self.xgrid[i*self.Ng + m1] - self.xgrid[i*self.Ng + j]
                            
                    val /= self.xgrid[i*self.Ng + m1] - self.xgrid[i*self.Ng + m2]
                    
                else:
                    if m1==0:
                        val = -1./ (2 * self.fw[i*self.Ng])
                        
                    elif m1==self.Ng:
                        val = 1./ (2 * self.bw[i+1])
                        
        return val
#==============================================================================
    def derivmat(self,i):
        fderiv = np.vectorize(self.derLobatto)
        m1 = np.arange(self.Ng+1)
        m2 = np.copy(m1)
        
        mat = fderiv(i,m1[:,None],m2[None,:])
        
        return mat
#==============================================================================
    def ttilde(self,i):
        w = np.copy(self.fw[i*self.Ng: (i+1)*self.Ng])
        w = np.append(w,self.bw[i+1])
        D = self.derivmat(i)
        
        mat = np.einsum('ij,kj,j->ik',D,D,w)
        
        return mat
#==============================================================================
    def tderiv1(self,i):
        wp = np.copy(self.fw[i*self.Ng: (i+1)*self.Ng])
        wp = np.append(wp,self.bw[i+1])
        w = np.array( [wp for i in range(self.Ng+1)] )
        D = self.derivmat(i)

        return D.T * w.T
#==============================================================================
    def FmdvrBoundary(self):
        v = np.zeros(self.Ne * self.Ng + 1)
        tm = self.ttilde(self.Ne - 1)
        
        # i=ne-1, m=0
        t = np.copy(tm[0,-1])
        w1 = np.copy(self.bw[-1])
        w2 = np.copy(self.bw[-2] + self.fw[(self.Ne-1) * self.Ng])
        w = np.sqrt(w1*w2)
        v[(self.Ne-1) * self.Ng] = 0.5 * t / w
        
        # i=ne-1, m>0
        t = np.copy(tm[1:self.Ng, -1])
        w1 = np.copy(self.fw[(self.Ne-1)*self.Ng+1: -1])
        w2 = np.copy(self.bw[-1])
        w = np.sqrt(w1*w2)
        v[(self.Ne-1)*self.Ng+1: -1] = 0.5 * t / w
        
        return v[1:-1]
#==============================================================================
    def ProjectOntoLobattoBasis(self,spl):
        ' Calculate the projection onto the FEMDVR basis.     '
        ' spl: interpolator for the function to be projected. '

        w = np.copy(self.fw)
        w = np.sqrt(w)
        for i in range(self.Ne):
            w[i*self.Ng] = np.sqrt(self.bw[i] + self.fw[i*self.Ng])
            
        proj = spl(self.xgrid[1:-1]) * w[1:-1]

        return proj
#==============================================================================
    def GetFirstDeriv(self):
        '''
        The first derivative operator in the Lobatto basis.
        '''
        n = self.Ne * self.Ng
        T = np.zeros((n,n))

        for i in range(self.Ne):
            tm = self.tderiv1(i)
            tmm1 = self.tderiv1(i-1)

            # i1 = i2
            # --------------

            # m1>0, m2>0
            t = np.copy(tm[1: self.Ng, 1: self.Ng])
            w1 = np.copy(self.fw[i*self.Ng+1: (i+1)*self.Ng])
            w2 = np.copy(w1)
            w = np.sqrt(w1[:,None]*w2[None,:])
            T[i*self.Ng+1: (i+1)*self.Ng, i*self.Ng+1: (i+1)*self.Ng] = t / w

            # m1=0, m2>0
            t = np.copy(tm[0, 1: self.Ng])
            w1 = np.copy(self.fw[i*self.Ng+1: (i+1)*self.Ng])
            w2 = self.bw[i] + self.fw[i*self.Ng]
            w = np.sqrt(w1*w2)
            T[i*self.Ng, i*self.Ng+1: (i+1)*self.Ng] = t / w

            # m1>0, m2=0
            t = np.copy(tm[1: self.Ng, 0])
            T[i*self.Ng+1: (i+1)*self.Ng, i*self.Ng] = t / w

            # m1=m2=0
            t = np.copy(tm[0,0] + tmm1[-1,-1])
            w = self.bw[i] + self.fw[i*self.Ng]
            T[i*self.Ng, i*self.Ng] = t / w

            # i1 = i2+1
            # --------------------

            if i>0:

                # m1=0, m2>0
                t = np.copy(tmm1[-1, 1: self.Ng])
                w1 = np.copy(self.fw[(i-1)*self.Ng+1: i*self.Ng])
                w2 = np.copy(self.bw[i] + self.fw[i*self.Ng])
                w = np.sqrt(w1*w2)
                T[i*self.Ng, (i-1)*self.Ng+1: i*self.Ng] = t / w
                
                # m1=m2=0
                t = np.copy(tmm1[-1, 0])
                w1 = np.copy(self.bw[i] + self.fw[i*self.Ng])
                w2 = np.copy(self.bw[i-1] + self.fw[(i-1)*self.Ng])
                w = np.sqrt(w1*w2)
                T[i*self.Ng, (i-1)*self.Ng] = t / w

            # i1 = i2-1
            # --------------------
            
            if i<self.Ne-1:
                
                # m1>0, m2=0
                t = np.copy(tm[1:self.Ng, -1])
                w1 = np.copy(self.fw[i*self.Ng+1: (i+1)*self.Ng])
                w2 = np.copy(self.bw[i+1] + self.fw[(i+1)*self.Ng])
                w = np.sqrt(w1*w2)
                T[i*self.Ng+1: (i+1)*self.Ng, (i+1)*self.Ng] = t / w
                
                # m1=m2=0
                t = np.copy(tm[0,-1])
                w1 = np.copy(self.bw[i] + self.fw[i*self.Ng])
                w2 = np.copy(self.bw[i+1] + self.fw[(i+1)*self.Ng])
                w = np.sqrt(w1*w2)
                T[i*self.Ng, (i+1)*self.Ng] = t / w

        return T[1:, 1: ]
#==============================================================================
    def Kinetic(self):
        '''
        Calculate the kinetic energy matrix elements.
        '''
        n = self.Ne * self.Ng
        T = np.zeros((n,n))
        
        for i in range(self.Ne):
            tm = self.ttilde(i)
            tmm1 = self.ttilde(i-1)
            
            # i1 = i2
            # -------------------
            
            # m1>0, m2>0
            t = np.copy(tm[1: self.Ng, 1: self.Ng])
            w1 = np.copy(self.fw[i*self.Ng+1: (i+1)*self.Ng])
            w2 = np.copy(w1)
            w = np.sqrt(w1[:,None]*w2[None,:])
            T[i*self.Ng+1: (i+1)*self.Ng, i*self.Ng+1: (i+1)*self.Ng] = 0.5 * t / w
            
            # m1=0, m2>0
            t = np.copy(tm[0, 1: self.Ng])
            w1 = np.copy(self.fw[i*self.Ng+1: (i+1)*self.Ng])
            w2 = self.bw[i] + self.fw[i*self.Ng]
            w = np.sqrt(w1*w2)
            T[i*self.Ng, i*self.Ng+1: (i+1)*self.Ng] = 0.5 * t / w
            
            # m1>0, m2=0
            t = np.copy(tm[1: self.Ng, 0])
            T[i*self.Ng+1: (i+1)*self.Ng, i*self.Ng] = 0.5 * t / w
            
            # m1=m2=0
            t = np.copy(tm[0,0] + tmm1[-1,-1])
            w = self.bw[i] + self.fw[i*self.Ng]
            T[i*self.Ng, i*self.Ng] = 0.5 * t / w
            
            # i1 = i2+1
            # --------------------
            
            if i>0:
                
                # m1=0, m2>0
                t = np.copy(tmm1[-1, 1: self.Ng])
                w1 = np.copy(self.fw[(i-1)*self.Ng+1: i*self.Ng])
                w2 = np.copy(self.bw[i] + self.fw[i*self.Ng])
                w = np.sqrt(w1*w2)
                T[i*self.Ng, (i-1)*self.Ng+1: i*self.Ng] = 0.5 * t / w
                
                # m1=m2=0
                t = np.copy(tmm1[-1,0])
                w1 = np.copy(self.bw[i] + self.fw[i*self.Ng])
                w2 = np.copy(self.bw[i-1] + self.fw[(i-1)*self.Ng])
                w = np.sqrt(w1*w2)
                T[i*self.Ng, (i-1)*self.Ng] = 0.5 * t / w
                
            
            # i1 = i2-1
            # --------------------
            
            if i<self.Ne-1:
                
                # m1>0, m2=0
                t = np.copy(tm[1:self.Ng, -1])
                w1 = np.copy(self.fw[i*self.Ng+1: (i+1)*self.Ng])
                w2 = np.copy(self.bw[i+1] + self.fw[(i+1)*self.Ng])
                w = np.sqrt(w1*w2)
                T[i*self.Ng+1: (i+1)*self.Ng, (i+1)*self.Ng] = 0.5 * t / w
                
                # m1=m2=0
                t = np.copy(tm[0,-1])
                w1 = np.copy(self.bw[i] + self.fw[i*self.Ng])
                w2 = np.copy(self.bw[i+1] + self.fw[(i+1)*self.Ng])
                w = np.sqrt(w1*w2)
                T[i*self.Ng, (i+1)*self.Ng] = 0.5 * t / w
                
        return T[1:, 1:]
#==============================================================================
    def init_nonloc(self,D,rmesh,beta,lbeta):
        '''
        Initialize variables for using a non-local potential.
        '''
        self.D = D
        self.rmesh = rmesh
        self.beta = beta
        self.beta_is_complex = np.iscomplexobj(beta)
        self.lbeta = lbeta
#==============================================================================
    def get_Vnl(self,l):
        '''
        Calculate the non-local potential matrix elements.
        '''
        nproj = self.beta.shape[-1]
        N = len(self.xgrid[1:-1])
        
        if self.beta_is_complex:
            Vnl = np.zeros((N,N), dtype=complex)
        else:
            Vnl = np.zeros((N,N))
        
        for i in range(nproj):
            if self.lbeta[i] == l:
                spl_r = make_interp_spline(self.rmesh, np.real(self.beta[:,i]))
                beta = self.ProjectOntoLobattoBasis(spl_r)
                
                if self.beta_is_complex:
                    spl_i = make_interp_spline(self.rmesh, np.imag(self.beta[:,i]))
                    beta = beta + 1j*self.ProjectOntoLobattoBasis(spl_i)

                Vnl += self.D[i] * np.outer(beta, np.conj(beta))

        return Vnl
#==============================================================================
    def InitPotential(self,rr,Vloc):
        '''
        Initialize the bound-state potential.
        rr     : radial grid for Vloc
        Vloc   : local potential
        '''
        spl = make_interp_spline(rr, Vloc, k=3)
        xpoints = self.xgrid[1:-1]
        Npoints = len(xpoints)
        
        self.Z = rr[-1] * Vloc[-1]
        self.Vr = np.zeros(Npoints)

        # Interpolate the local potential onto the femdvr grid
        Ir, = np.where(xpoints <= rr[-1])
        self.Vr[Ir] = spl(xpoints[Ir])

        # Extrapolate the potential for points beyond the last point of the input
        Ir, = np.where(xpoints > rr[-1])
        self.Vr[Ir] = self.Z / xpoints[Ir]
#==============================================================================
    def InitScattering(self,E):
        '''
        Initialize the scattering state parameters.
        E      : energy of the scattering state
        '''
        self.E = E
        self.ke = np.sqrt(2. * E)
        self.eta = self.Z / self.ke
#==============================================================================
    def CoulombFunction(self,l,r,real=False):
        '''
        Calculate the Coulomb functions
        l       : angular momentum channel
        r       : radial point
        real    : real or complex Coulomb function
        '''
        
        Fc, Gc, _, _, _ = coul90(self.ke*r,self.eta,l+1,0)
        F = Fc[l]
        G = Gc[l]
        f = np.angle(gamma(l+1 + 1j*self.eta))
        r = np.exp(-1j * f)
            
        if real:
            Hp = F
            Hm = G
        else:
            Hp = r * (F + 1.0j*G)
            Hm = np.conj(Hp)
        
        return Hp, Hm
#==============================================================================    
    def GetCoulombBoundaryCondition(self,l,real):
        '''
        Calculate the Coulomb boundary condition in the FEM-DVR basis:
        l       : angular momentum channel
        real    : real or complex Coulomb functions
        '''
        r1 = self.xgrid[-1]
        r2 = self.xgrid[-2]
        H1 = self.CoulombFunction(l,r1,real)
        H2 = self.CoulombFunction(l,r2,real)
        
        a = H1[1] / H2[1]        
        b = H1[0] - a*H2[0]
        
        a *= np.sqrt(self.bw[-1] / self.fw[-2])
        b *= np.sqrt(self.bw[-1])
        
        return a, b
#==============================================================================
    def AddScatteringBoundary(self,x,l,const,real):
        '''
        Add the scattering boundary condition to the wave function.
        x       : the grid point
        l       : angular momentum channel 
        const   : phase factor between the Coulomb functions
        real    : real or complex Coulomb functions
        '''
        R = self.GetCoulombBoundaryCondition(l,real)
        c = R[0]*const + R[1]
        
        val = c * self.LobattoPoly(self.Ne-1,self.Ng,x) / np.sqrt(self.bw[-1])
        
        return val
#==============================================================================
    def Vlr(self,l):
        Vl = 0.5 * l*(l+1) / self.xgrid[1:-1]**2
        Vlr = self.Vr + Vl
        
        return Vlr 
#==============================================================================
    def GetBound(self,l,n=None):
        '''
        Solves the bound-state problem.
        l   : angular momentum quantum number
        n   : principal quantum number
        '''
        Vlr = np.diag(self.Vlr(l))
        Ham = self.Kin + Vlr
        
        if self.nonloc == True:
            Ham = Ham + self.get_Vnl(l)

        E, V = la.eigh(Ham)
        
        if n != None:
            k = n - l - 1
            if k >= 0:
                E = E[k]
                V = V[:,k]
            else: 
                raise ValueError('wrong n value')
        
        return E, V
#==============================================================================
    def GetHamMinEn(self,l,c):
        Em = self.E * np.eye(self.Ne*self.Ng - 1)
        Vlr = np.diag(self.Vlr(l))
        v = self.FmdvrBoundary()
        
        mat = self.Kin + Vlr - Em
            
        if self.nonloc == True:
            mat = mat + self.get_Vnl(l)
            
        if np.iscomplexobj(c):
            mat = mat.astype(complex)
            
        mat[:, -1] = np.copy(mat[:, -1] + c*v)
        
        return mat
#==============================================================================    
    def GetScatt(self,E,l,real=False):
        '''
        Solves the scattering state.
        E       : scattering energy [Ha]
        l       : angular momentum quantum number
        real    : real or complex Coulomb function as the boundary condition
        '''
        self.InitScattering(E)
        
        K = self.GetCoulombBoundaryCondition(l,real)
        v = self.FmdvrBoundary()
        A = self.GetHamMinEn(l,K[0])
        b = -K[1] * v
        
        x = la.solve(A,b)
        
        return x
#==============================================================================
    def Psix(self,eigv,x,bound=True,l=0,real=False):
        '''
        Calculate the wave function at a given point x.
        eigv    : eigenvalue of the Hamiltonian
        x       : the grid point
        bound   : whether to calculate the bound-state or scattering state
        l       : if bound=True, angular momentum channel
        real    : if bound=True, real or complex Coulomb function as the boundary condition
        '''
        
        basis = [[self.LobattoBasis(i,m,x) for m in range(self.Ng)] for i in range(self.Ne)]
        basis = np.array(basis).reshape(-1)

        psi = np.dot(eigv, basis[1:])
        
        if not bound:
            psi += self.AddScatteringBoundary(x,l,eigv[-1],real)

        return psi
#==============================================================================
    def GetWavefunc(self,eigv,rr,bound=True,l=0,real=False):
        '''
        Calculate the wave function on a given radial grid.
        eigv    : eigenvalue of the Hamiltonian
        rr      : array of grid points
        '''
        
        psi = [self.Psix(eigv,x,bound,l,real) for x in rr]
        psi = np.array(psi)
        
        if bound:
            irmax = np.argmax(np.abs(psi))
            psi *= np.sign(psi[irmax])

        return psi
#==============================================================================
    def GetScatteringPhase(self,KinE,l,smooth=False,real=False):
        '''
        Calculate the scattering phase at infinity.
        KinE    : kinetic (scattering) energy [Ha]
        l       : angular momentum channel
        '''
        
        # Infinity is taken to be the last point in the grid
        R = self.xgrid[-1]
        phase = []
        
        if real:
            f = np.angle(gamma(l+1 + 1j*self.eta))
            f = np.exp(-2j * f)
        
        for E in KinE:
            v = self.GetScatt(E,l,real)
            yR = self.Psix(v,R,bound=False,l=l,real=real)
            H = self.CoulombFunction(l,R,real)

            S = (yR - H[0]) / H[1]

            if real:
                S = (1. + 1.0j*S) / (1. - 1.0j*S)
                S *= f

            phase.append(S)

        phase = np.array(phase)
        Sl = np.angle(phase) / np.pi

        if smooth:
            Sl = self.SmoothPhase(Sl)

        return Sl
#==============================================================================
    def SmoothPhase(self,phase):
        '''
        Smoothen the phase so that there is no jumps due to the phase unwrapping.
        phase : array of phases in units of pi
        '''

        phase_smooth = np.copy(phase)

        for i in range(1, len(phase)):
            if phase[i] - phase_smooth[i-1] > 1.5:
                phase_smooth[i] -= 2.0
            elif phase[i] - phase_smooth[i-1] < -1.5:
                phase_smooth[i] += 2.0
        
        return phase_smooth
#==============================================================================
    def GetRadialIntegral(self,KinE,n,l,verbose='low',store_type='real-imag',smooth_phase=False,Boundwfc=[]):
        '''
        Calculate the photo-emission radial integral.
        KinE         : kinetic (scattering) energy [Ha]
        n            : principal quantum number
        l            : angular momentum channel
        verbose      : verbosity level ('low', 'high')
        store-type   : type of the output ('real-imag' or 'abs-angle')
        smooth_phase : if store_type is 'abs-angle', smoothen the phase
        Boundwfc     : bound-state wave function for the radial integral
        '''
        
        if len(Boundwfc) == 0:
            _, b = self.GetBound(l, n)
            psi_inf = self.Psix(b, self.xgrid[-1])
            
            if psi_inf < 0.0:
                b *= -1.0
        else:
            bound_spl = make_interp_spline(Boundwfc[:,0], Boundwfc[:,1])
            b = self.ProjectOntoLobattoBasis(bound_spl)
            
        rb = self.xgrid[1:-1] * b
        
        if l == 0:
            l_final = [1]
        else:
            l_final = [l-1, l+1]
        
        if store_type == 'real-imag':
            radint = np.zeros((len(KinE), len(l_final)), dtype=complex)

        elif store_type == 'abs-angle':
            radint_abs = np.zeros((len(KinE), len(l_final)), dtype=float)
            radint_angle = np.zeros((len(KinE), len(l_final)), dtype=float)
        
        for i,l1 in enumerate(l_final):
            
            print('Calculating for scattering channel l =' + str(l1))
            
            for j,E in enumerate(KinE):
                if verbose == 'high':
                    print('Calculating for energy E = ' + str(E) + ' Ha')
                    
                v = self.GetScatt(E, l1)
                dot = np.dot(np.conj(v), rb)
                
                if store_type == 'real-imag':
                    radint[j,i] = dot
                elif store_type == 'abs-angle':
                    radint_abs[j,i] = np.abs(dot)
                    radint_angle[j,i] = np.angle(dot) / np.pi

            
        if store_type == 'real-imag':
            return radint
        
        elif store_type == 'abs-angle':
            if smooth_phase:
                for ich in range(len(l_final)):
                    radint_angle[:,ich] = self.SmoothPhase(radint_angle[:,ich])

            return radint_abs, radint_angle
#==============================================================================
