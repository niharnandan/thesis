import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt

class doeclim:
    def __init__ (self, deltat=1, nsteps=1):
        self.ak = 0.31
        self.bk   = 1.59
        self.bsi  = 1.3
        self.cal  = 0.52
        self.cas  = 7.80
        self.csw  = 0.13
        self.flnd = 0.29
        self.fso  = 0.95
        self.kcon = 3155.8
        self.q2co = 3.7
        self.rlam = 1.43
        self.zbot = 4000
        self.deltat = deltat#???
        self.nsteps = nsteps #???
        self.earth_area = 5100656#???.D8      #[m^2]
        self.secs_per_Year = 3.154e+7
        self.temp_landair = np.array([0.]*self.nsteps)
        self.temp_sst = np.array([0.]*self.nsteps)
        self.heat_mixed = np.array([0.]*self.nsteps)
        self.heat_interior = np.array([0.]*self.nsteps)
        self.heatflux_mixed = np.array([0.]*self.nsteps)
        self.heatflux_interior = np.array([0.]*self.nsteps)
        self.IBaux, self.Baux, self.Cdoe = ([np.ndarray(shape=(2,2), dtype=float)]*3)
        self.QL, self.Q0 = np.array([0.]*self.nsteps), np.array([0.]*self.nsteps)
        self.Ker = np.array([0.]*self.nsteps)
        self.ocean_area = (1.-self.flnd)*self.earth_area
        self.KT0,self.KTA1,self.KTB1,self.KTA2,self.KTB2,self.KTA3,self.KTB3 = np.array([0.]*self.nsteps),np.array([0.]*self.nsteps), \
		np.array([0.]*self.nsteps),np.array([0.]*self.nsteps),np.array([0.]*self.nsteps),np.array([0.]*self.nsteps),np.array([0.]*self.nsteps)
        self.ocean_area = (1.-self.flnd)*self.earth_area
        self.powtoheat = self.ocean_area*self.secs_per_Year / 1.2
        self.cnum= self.rlam*self.flnd + self.bsi * (1.0-self.flnd)
        self.cden = self.rlam * self.flnd - self.ak *(self.rlam-self.bsi)
        self.taucfl, self.taukls, self.taucfs, self.tauksl, self.taudif, self.taubot = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.IB, self.Adoe = np.ndarray(shape=(2,2), dtype=float), np.ndarray(shape=(2,2), dtype=float)

        
    def init_doeclim_parameters(self, t2co, kappa):
        self.keff = self.kcon * kappa
        self.cfl = self.flnd *self.cnum/self.cden*self.q2co/t2co-self.bk*(self.rlam-self.bsi)/self.cden
        
        self.cfs = (self.rlam * self.flnd - self.ak / (1.0-self.flnd) * (self.rlam-self.bsi))                  \
        * self.cnum / self.cden * self.q2co / t2co + self.rlam * self.flnd / (1.0-self.flnd) * self.bk *       \
        (self.rlam - self.bsi) / self.cden
        
        self.kls = self.bk * self.rlam * self.flnd / self.cden - self.ak * self.flnd * self.cnum             \
        / self.cden * self.q2co / t2co
        
        self.taubot = self.zbot**2 /self. keff
        self.taudif = self.cas**2 / self.csw**2 * np.pi / self.keff
        self.taucfs = self.cas / self.cfs
        self.taucfl = self.cal / self.cfl
        self.tauksl  = (1.0-self.flnd) * self.cas / self.kls
        self.taukls  = self.flnd * self.cal / self.kls

        self.KT0[self.nsteps-1] = 4.0 - 2.0*np.sqrt(2.0) #4-2*SQRT(2.)
        self.KTA1[self.nsteps-1] = -8.0*np.exp(-self.taubot/self.deltat) + \
                    4.0*np.sqrt(2.0)*np.exp(-0.5*self.taubot/self.deltat)
        self.KTB1[self.nsteps-1] = 4*np.sqrt(np.pi*self.taubot/self.deltat) * \
            (1+special.erf(np.sqrt(0.5*self.taubot/self.deltat)) - 2*special.erf(np.sqrt(self.taubot/self.deltat)))
        self.KTA2[self.nsteps-1] =  8*np.exp(-4.*self.taubot/self.deltat) - \
                4*np.sqrt(2.0)*np.exp(-2.0*self.taubot/self.deltat)

        self.KTB2[self.nsteps-1] = -8*np.sqrt(np.pi*self.taubot/self.deltat) *  \
            (1.0+ special.erf(np.sqrt(2.0*self.taubot/self.deltat)) - 2.0*special.erf(2.*np.sqrt(self.taubot/self.deltat)) )


        self.KTA3[self.nsteps-1] = -8.*np.exp(-9.0*self.taubot/self.deltat) + \
            4*np.sqrt(2.0)*np.exp(-4.5*self.taubot/self.deltat)

        self.KTB3[self.nsteps-1] = 12.0*np.sqrt(np.pi*self.taubot/self.deltat) * \
          (1 +special.erf(np.sqrt(4.5*self.taubot/self.deltat)) - 2.0*special.erf(3.0*np.sqrt(self.taubot/self.deltat)) )

#-----------------------------------------------------------------
#!%Hammer and Hollingsworth correction (Equation 2.3.27, TK07):
#!%Switched on (To switch off, comment out lines below)
#!     do i=1,N_samples_lambda_star
        self.Cdoe[0,0] = 1./self.taucfl**2+1./self.taukls**2                      \
          +2./self.taucfl/self.taukls+self.bsi/self.taukls/self.tauksl
        self.Cdoe[0,1] = -self.bsi/self.taukls**2-self.bsi/self.taucfl/self.taukls            \
        -self.bsi/self.taucfs/self.taukls-self.bsi**2/self.taukls/self.tauksl
        self.Cdoe[1,0] = -self.bsi/self.tauksl**2-1./self.taucfs/self.tauksl             \
        -1./self.taucfl/self.tauksl-1./self.taukls/self.tauksl
        self.Cdoe[1,1] =  1./self.taucfs**2+self.bsi**2/self.tauksl**2                \
        +2.*self.bsi/self.taucfs/self.tauksl+self.bsi/self.taukls/self.tauksl
        self.Cdoe=self.Cdoe*(self.deltat**2/12.)

        self.Baux[0,0] = 1. + self.deltat/(2.*self.taucfl) + self.deltat/(2.*self.taukls)
        self.Baux[0,1] = -self.deltat/(2.*self.taukls)*self.bsi
        self.Baux[1,0] = -self.deltat/(2.*self.tauksl)
        self.Baux[1,1] = 1. + self.deltat/(2.*self.taucfs) + self.deltat/(2.*self.tauksl)*self.bsi +    \
        2.*self.fso*np.sqrt(self.deltat/self.taudif)
        self.Baux=self.Baux+self.Cdoe
        self.MIGS(self.Baux, 2, self.IBaux)  #!,indx)
        self.IB = self.IBaux
        for i in range(1,self.nsteps):
            self.KT0[i-1] = 4.0*np.sqrt((self.nsteps+1-i)) - 2.*np.sqrt((self.nsteps+2-i)) - 2.0*np.sqrt((self.nsteps-i))


            self.KTA1[i-1] = -8.0*np.sqrt((self.nsteps+1-i)) * \
            np.exp(-self.taubot/self.deltat/(self.nsteps+1-i)) + \
            4.0*np.sqrt((self.nsteps+2-i)) *np.exp(-self.taubot/self.deltat/(self.nsteps+2-i)) + \
            4.0*np.sqrt((self.nsteps-i)) *np.exp(-self.taubot/self.deltat/(self.nsteps-i))

            self.KTB1[i-1] =  4.0*np.sqrt(np.pi*self.taubot/self.deltat) * ( \
            special.erf(np.sqrt(self.taubot/self.deltat/(self.nsteps-i))) + \
            special.erf(np.sqrt(self.taubot/self.deltat/(self.nsteps+2-i))) - \
            2.0*special.erf(np.sqrt(self.taubot/self.deltat/(self.nsteps+1-i))) )



            self.KTA2[i-1] =  8.*np.sqrt((self.nsteps+1-i)) *                               \
            np.exp(-4.*self.taubot/self.deltat/(self.nsteps+1-i))- 4.*np.sqrt((self.nsteps+2-i))*           \
            np.exp(-4.*self.taubot/self.deltat/(self.nsteps+2-i))- 4.*np.sqrt((self.nsteps-i)) *            \
            np.exp(-4.*self.taubot/self.deltat/(self.nsteps-i))

            self.KTB2[i-1] = -8.*np.sqrt(np.pi*self.taubot/self.deltat) * (                         \
            special.erf(2.*np.sqrt(self.taubot/self.deltat/((self.nsteps-i)))) +                          \
            special.erf(2.*np.sqrt(self.taubot/self.deltat/(self.nsteps+2-i))) -                        \
            2.*special.erf(2.*np.sqrt(self.taubot/self.deltat/(self.nsteps+1-i))) )



            self.KTA3[i-1] = -8.*np.sqrt((self.nsteps+1-i)) *                              \
            np.exp(-9.*self.taubot/self.deltat/(self.nsteps+1.-i)) + 4.*np.sqrt((self.nsteps+2-i))*        \
            np.exp(-9.*self.taubot/self.deltat/(self.nsteps+2.-i)) + 4.*np.sqrt((self.nsteps-i))*          \
            np.exp(-9.*self.taubot/self.deltat/(self.nsteps-i))

            self.KTB3[i-1] = 12.*np.sqrt(np.pi*self.taubot/self.deltat) * (\
            special.erf(3.*np.sqrt(self.taubot/self.deltat/(self.nsteps-i))) + \
            special.erf(3.*np.sqrt(self.taubot/self.deltat/(self.nsteps+2-i))) - \
            2.*special.erf(3.*np.sqrt(self.taubot/self.deltat/(self.nsteps+1-i))))

        self.Ker = self.KT0+self.KTA1+self.KTB1+self.KTA2+self.KTB2+self.KTA3+self.KTB3

        self.Adoe[0,0] = 1.0 - self.deltat/(2.*self.taucfl) - self.deltat/(2.*self.taukls)
        self.Adoe[0,1] =  self.deltat/(2.*self.taukls)*self.bsi
        self.Adoe[1,0] =  self.deltat/(2.*self.tauksl)
        self.Adoe[1,1] = 1.0 - self.deltat/(2.*self.taucfs) - self.deltat/(2.*self.tauksl)*self.bsi +   \
        self.Ker[self.nsteps-1]*self.fso*np.sqrt(self.deltat/self.taudif)
        self.Adoe=self.Adoe+self.Cdoe
        
        return
        
    
    def doeclimtimestep_simple(self, n, forcing, temp):
        DQ, DPAST, QC, DTEAUX = np.array([0.,0.]), np.array([0.,0.]), np.array([0.,0.]), np.array([0.,0.])
        DTE = np.ndarray(shape=(2,self.nsteps))
        DelQL, DelQ0 = 0.,0.
        DTE[0,:] = self.temp_landair
        DTE[1,:] = self.temp_sst
        
        self.QL[n] = forcing # Timestep n-1 !-forcing(1)!-1.4D-1!-FORC(1)    !forcing 
        self.Q0[n] = forcing #!-forcing(1)!-1.4D-1!-FORC(1)    !forcing

        if (n > 0):
            DelQL = self.QL[n] - self.QL[n-1]
            DelQ0 = self.Q0[n] - self.Q0[n-1]
            QC[0] = (DelQL/self.cal*(1./self.taucfl+1./self.taukls)-self.bsi*DelQ0/self.cas/self.taukls)
            QC[1] = (DelQ0/self.cas*(1./self.taucfs+self.bsi/self.tauksl)-DelQL/self.cal/self.tauksl)
            QC = QC* self.deltat**2/12.
            DQ[0] = 0.5*self.deltat/self.cal*(self.QL[n]+self.QL[n-1])
            DQ[1] = 0.5*self.deltat/self.cas*(self.Q0[n]+self.Q0[n-1])
            DQ = DQ + QC


            for i in range(n):
                DPAST[1] = DPAST[1]+DTE[1,i]*self.Ker[self.nsteps-n+i-1]
            DPAST[1] = DPAST[1]*self.fso * np.sqrt(self.deltat/self.taudif)
            DTEAUX[0] = self.Adoe[0,0]*DTE[0,n-1]+self.Adoe[0,1]*DTE[1,n-1]
            DTEAUX[1] = self.Adoe[1,0]*DTE[0,n-1]+self.Adoe[1,1]*DTE[1,n-1]

            DTE[0,n] = self.IB[0,0]*(DQ[0]+DPAST[0]+DTEAUX[0])+ self.IB[0,1]*(DQ[1]+DPAST[1]+DTEAUX[1])
            DTE[1,n] = self.IB[1,0]*(DQ[0]+DPAST[0]+DTEAUX[0])+ self.IB[1,1]*(DQ[1]+DPAST[1]+DTEAUX[1])
            self.temp_sst[n] = DTE[1,n]
            self.temp_landair[n] = DTE[0,n]
			
            print(DTE[0,n], DTE[1,n])
            self.heatflux_mixed[n] = self.cas*( DTE[1,n]-DTE[1,n-1] )
            for i in range(n):
                self.heatflux_interior[n] = self.heatflux_interior[n]+DTE[1,i]*self.Ker[self.nsteps-n+i]

            self.heatflux_interior[n] = self.cas*self.fso/np.sqrt(self.taudif*self.deltat)*(2.*DTE[1,n] - self.heatflux_interior[n])
            self.heat_mixed[n] = self.heat_mixed[n-1] +self.heatflux_mixed[n] *(self.powtoheat*self.deltat)

            self.heat_interior[n] = self.heat_interior[n-1] + self.heatflux_interior[n] * (self.fso*self.powtoheat*self.deltat)

        temp[0] = self.flnd*self.temp_landair[n] + (1.-self.flnd)*self.bsi*self.temp_sst[n]

        return

    def MIGS(self, FV,N,X):
	
        INDX = np.array([0]*N)
        VF, self.B = np.ndarray(shape=(N,N), dtype=float), np.ndarray(shape=(N,N), dtype=float)
        VF = np.copy(FV)

        for I in range (N):
            self.B[I,I] = 1.0


        self.ELGS(VF,N,INDX)
        for I in range (N-1):
            for J in range(I+1,N):
                for K in range (N):
                    self.B[INDX[J],K] = self.B[INDX[J],K]-VF[INDX[J],I]*self.B[INDX[I],K]

        for I in range(0,N):
            X[N-1,I] = self.B[INDX[N-1],I]/(VF[INDX[N-1],N-1])
            for J in range(N-2,-1,-1):
                X[J,I] = self.B[INDX[J],I]
                for K in range(J+1,N):
                    X[J,I] = X[J,I]-VF[INDX[J],K]*X[K,I]
                X[J,I] =  X[J,I]/VF[INDX[J],J]
        return
    
    def ELGS(self, VF,N,INDX):
        C = np.array([0]*N)
        for I in range(0,N):
            INDX[I] = I
        for I in range(N):
            C1= 0.
            for J in range(N):
                C1 = max(C1,abs(VF[I,J]))
            C[I] = C1

        for J in range(N-1):
            PI1 = 0.
            for I in range(J,N):
                PI = abs(VF[INDX[I],J])/C[INDX[I]]
                if (PI > PI1):
                    PI1 = PI
                    K   = I

            ITMP    = INDX[J]
            INDX[J] = INDX[K]
            INDX[K] = ITMP
            for I in range(J+1,N):
                PJ  = VF[INDX[I],J]/VF[INDX[J],J]
                VF[INDX[I],J] = PJ
                for K in range(J+1,N):
                    VF[INDX[I],K] = VF[INDX[I],K]-PJ*VF[INDX[J],K]
        return 