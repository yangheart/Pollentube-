import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/Users/sichenchen/Desktop/research/cnn/PDE-Net-PDE-Net-2.0/aTEAM/")
import aTEAM. pdetools as pdetools
from aTEAM.pdetools.stepper import TimeStepper
from aTEAM.pdetools.upwind1 import UpWind2dRHI, _pad
from aTEAM.pdetools.initb import initgen



class Murrayll28(nn.Module, TimeStepper):
    """
    2d convection diffusion equation with reactive source
    \partial_t u+ u u_x+v u_y = nu\laplace u+(1-A) u+\beta A v
    \partial_t v+ u v_x+v v_y = nu\laplace v-\beta A u+(1-A) v
    where A=u^2+v^2
    """
    @property
    def timescheme(self):
        return self._timescheme
    @property
    def spatialscheme(self):
        return self._spatialscheme
    def RightHandItems(self, u, **kw):
        """
        u[...,0,y,x],u[...,1,y,x]
        """

        upad = _pad(u, [1,1,1,1], mode='wrap')


        rhi = (1/self.dx**2)*(upad[...,2:,1:-1]+upad[...,:-2,1:-1]
                +upad[...,1:-1,2:]+upad[...,1:-1,:-2]-4*upad[...,1:-1,1:-1])
        a=1
        b=2
        pho=2
        k=4
        k1=0.5
        alpha=0.2
      
        rhi[...,:1,:,:]=0.3*rhi[...,:1,:,:]
        rhi[...,1:,:,:]=0.4*rhi[...,1:,:,:]
        A = u[...,:1,:,:]
        B=u[...,1:,:,:]
        h=(pho*A*B)/(1+A+k*A**2)
        h1=(pho*A*B)/(1+torch.exp(-A))

        
        rhi[...,:1,:,:] += a-A-h  #ex2
        rhi[...,1:,:,:] += 0.2*(b-B)-h #ex2
        
        
        return rhi
    def __init__(self, max_dt, mesh_size, mesh_bound=((-5,-5),(5,5)), timescheme='rk2', spatialscheme='uw2'):
        super(Murrayll28, self).__init__()
        self.max_dt = max_dt
        self.mesh_size = np.array(mesh_size).copy()
        self.mesh_bound = np.array(mesh_bound).copy()
        dx0,dx1 = (self.mesh_bound[1]-self.mesh_bound[0])/self.mesh_size
        
        assert abs(dx0-dx1)<1e-10
        self.dx = dx0
        
        self._timescheme = timescheme
        
        self._spatialscheme = spatialscheme
        
    def forward(self, inputs, T, **kw):
        
        return self.predict(inputs, T, **kw)


