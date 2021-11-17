import numpy as np

class Particle:
    def __init__(self,pid):
        self.pid=pid
    
    def UpdatePosition(self,x,y,cellid):
        self.x=x
        self.y=y
        self.cellid=cellid

    def UpdateVelocity(self,u,v):
        self.u=u
        self.v=v
    
    def UpadateLayer(self,layer):   #0:surface 1:deposite -1:unused
        self.layer=layer

class Cell:
    def __init__(self,Area,M,N,z):
        self.M=M
        self.N=N
        self.dx=Area.lx/M
        self.dy=Area.ly/N
        self.cellid=np.arange(M*N)
        self.x=self.dx/2+self.cellid%M*self.dx
        self.y=self.dy/2+(self.cellid/M).astype(int)*self.dy
        self.z=np.zeros(M*N)+z

    def InitHydroCondition(self,u,v,h):
        self.u=np.zeros((self.M*self.N))+u
        self.v=np.zeros((self.M*self.N))+v
        self.h=np.zeros((self.M*self.N))+h
        self.qx=self.u*self.h
        self.qy=self.v*self.h

    def UpdatePollutionMass(self,Ms,Mg):
        self.Ms=Ms
        self.Mg=Mg

    def InitParticleInCell(self,p_mass):
        self.Ms_num=(self.Ms/p_mass).astype(int)
        self.Mg_num=(self.Mg/p_mass).astype(int)
        self.TolParticle_num=self.Mg_num+self.Ms_num
        self.Mrs=self.Ms%p_mass
        self.Mrg=self.Mg%p_mass

    def UpdateParticleLayer(self,SurfaceParticle,DepositionParticle):
        self.SurfaceParticle=SurfaceParticle
        self.DepositionParticle=DepositionParticle
    
    def UpdateParticle(self,p_mass,Ms_num,Mg_num,SurfaceParticle,DepositionParticle):
        self.Ms_num=Ms_num
        self.Mg_num=Mg_num
        self.TolParticle_num=Mg_num+Ms_num
        self.Mrs=self.Ms-self.Ms_num*p_mass
        self.Mrg=self.Mg-self.Mg_num*p_mass
        self.SurfaceParticle=SurfaceParticle
        self.DepositionParticle=DepositionParticle

    def UpdatePollution(self,Ms_num,p_mass):
        self.Ms_num=Ms_num
        self.Ms=Ms_num*p_mass+self.Mrs
        self.TolParticle_num=self.Mg_num+Ms_num

class CellParameters:
    def __init__(self,lx,ly):
        self.lx=lx
        self.ly=ly

class PollutionParameters:
    def __init__(self,c0,Dxx,Dyy,PNN):
        self.c0=c0
        self.Dxx=Dxx
        self.Dyy=Dyy
        self.PNN=PNN
        self.p_mass=c0/PNN

class PollutionAttributes:
    def __init__(self,ad0,DR,P,b,Sf,F,omega0,vs,rho_s):
        # Rainfall driven
        self.ad0=ad0
        self.DR=DR
        self.P=P
        self.b=b
        self.h0=0.33*DR
        # Flow driven
        self.Sf=Sf
        self.F=F
        self.omega0=omega0
        # Deposition
        self.vs=vs
        self.rho_s=rho_s
