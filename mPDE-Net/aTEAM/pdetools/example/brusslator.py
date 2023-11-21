import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/Users/sichenchen/Desktop/research/cnn/PDE-Net-PDE-Net-2.0/aTEAM/")
import aTEAM. pdetools as pdetools
from aTEAM.pdetools.stepper import TimeStepper
#from aTEAM.pdetools.upwind import UpWind2dRHI,_pad
#from aTEAM.pdetools.init import initgen
from aTEAM.pdetools.upwind1 import UpWind2dRHI,_pad
from aTEAM.pdetools.initb import initgen

from aTEAM.pdetools.spectral import *


class Brusselator(nn.Module, TimeStepper):
    """
    2d convection diffusion equation with reactive source
    \partial_t u = 1\laplace u+(3-7*u+u^2*v)
    \partial_t v= 5\laplace v + 6*u-u^2*v
    where A=u^2
    """
    print("Brusselator")
    @property
    def timescheme(self):
        return self._timescheme
    @property
    def spatialscheme(self):
        return self._spatialscheme
    def RightHandItems(self, u, viscosity=None, beta=None, force=None, **kw):
        """
        u[...,0,y,x],u[...,1,y,x]
        """
        #print("u", u)
        #print("ushape", u.shape)
        upad = _pad(u, [1,1,1,1], mode='wrap')
        #print("upad", upad.shape)
        #print("upshape", upad.shape)
        #print("dx",self.dx)
        rhi = (1/self.dx**2)*(upad[...,2:,1:-1]+upad[...,:-2,1:-1]
                +upad[...,1:-1,2:]+upad[...,1:-1,:-2]-4*upad[...,1:-1,1:-1])
        #print("rhi", rhi)
        rhi[...,:1,:,:]=1*rhi[...,:1,:,:]+3
        rhi[...,1:,:,:]=5*rhi[...,1:,:,:]
        A = u[...,:1,:,:]**2

        rhi[...,:1,:,:] += -7*u[...,:1,:,:]+A*u[...,1:,:,:]
        rhi[...,1:,:,:] += -A*u[...,1:,:,:]+6*u[...,:1,:,:] #-A*u[...,1:,:,:]

        if not force is None:
            #print("yes")
            rhi = rhi+force
        elif not self.force is None:
            rhi = rhi+self.force
        return rhi
    def __init__(self, max_dt, mesh_size, mesh_bound=((-5,-5),(5,5)), viscosity=0.01, beta=1, timescheme='rk2', spatialscheme='uw2', force=None):
        super(Brusselator, self).__init__()
        self.max_dt = max_dt
        self.mesh_size = np.array(mesh_size).copy()
        print("mesh_size", self.mesh_size)
        self.mesh_bound = np.array(mesh_bound).copy()
        print("mesh_bound", self.mesh_bound)
        dx0,dx1 = (self.mesh_bound[1]-self.mesh_bound[0])/self.mesh_size
        print("dx", dx0)
        # print("upbound", self.mesh_bound[1])
        # print("lbound", self.mesh_bound[0])
        # print("mesh", self.mesh_size)
        # print("dx0", dx0)
        assert abs(dx0-dx1)<1e-10
        self.dx = dx0
        self.viscosity = viscosity
        self.beta = beta
        self._timescheme = timescheme
        self._spatialscheme = spatialscheme
        self.force = force
        # print("force 1", self.force)
    def forward(self, inputs, T, **kw):
        return self.predict(inputs, T, **kw)


# viscosity=0.1
# beta=1
# max_dt=1e-5

# import matplotlib.pyplot as plt
# import time

# if torch.cuda.is_available():
#     device = 'cuda:0'
# else:
#     device = None
# mesh_size = [200,200]
# T = 1e-2
# batch_size = 2

# init = pdetools.initb.initgen(mesh_size=mesh_size, device=device, batch_size=batch_size)
# # init += init.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]*\
# #     torch.randn(2*batch_size,1,1, dtype=torch.float64, device=device)*\
# #     torch.rand(2*batch_size,1,1, dtype=torch.float64, device=device)*2
# cdr0 = Brusselator(max_dt=max_dt, mesh_size=mesh_size, mesh_bound=((-20,-20),(20,20)), viscosity=viscosity, beta=beta, timescheme='rk2')
# h = plt.figure()
# u0 = h.add_subplot(2,3,1,aspect='equal')
# uA0 = h.add_subplot(2,3,2,aspect='equal')
# uDelta0 = h.add_subplot(2,3,3,aspect='equal')
# v0 = h.add_subplot(2,3,4,aspect='equal')
# vA0 = h.add_subplot(2,3,5,aspect='equal')
# vDelta0 = h.add_subplot(2,3,6,aspect='equal')
# def resetticks(*argv):
#     for par in argv:
#         par.set_xticks([]); par.set_yticks([])

# resetticks(u0,uA0,uDelta0,v0,vA0,vDelta0)

# x0 = init.view([batch_size,2,]+mesh_size)

# Y,X = np.mgrid[0:1:(mesh_size[0]+1)*1j,0:1:(mesh_size[1]+1)*1j]
# Y,X = Y[:-1,:-1],X[:-1,:-1]
# for i in range(20):
#     u0.clear();uA0.clear();uDelta0.clear();v0.clear();vA0.clear();vDelta0.clear();

#     x0pad = _pad(x0, [1,1,1,1], mode='wrap')
#     deltax0 = (1/cdr0.dx**2)*(x0pad[...,2:,1:-1]+x0pad[...,:-2,1:-1])
#     print(x0[0,0].data.cpu().numpy()[::-1])
#     print(x0[0,1].data.cpu().numpy()[::-1])

#     A = x0[...,:1,:,:]**2
#     uA0rhi = -7*x0[...,:1,:,:]+A*x0[...,1:,:,:]+3
#     vA0rhi = 6*x0[...,:1,:,:]+A*x0[...,1:,:,:]
#     timeu0 = u0.imshow(x0[0,0].data.cpu().numpy()[::-1], cmap='jet')
#     timev0 = v0.imshow(x0[0,1].data.cpu().numpy()[::-1], cmap='jet')
#     timeuA0 = uA0.imshow(uA0rhi[0,0].data.cpu().numpy()[::-1],cmap='jet')
#     timevA0 = vA0.imshow(vA0rhi[0,0].data.cpu().numpy()[::-1],cmap='jet')
#     timeuDelta0 = uDelta0.imshow(deltax0[0,0].data.cpu().numpy()[::-1],cmap='jet')
#     timevDelta0 = vDelta0.imshow(deltax0[0,1].data.cpu().numpy()[::-1],cmap='jet')
#     colorbars = []
#     colorbars.append(h.colorbar(timeu0, ax=u0))
#     colorbars.append(h.colorbar(timev0, ax=v0))
#     colorbars.append(h.colorbar(timeuA0, ax=uA0))
#     colorbars.append(h.colorbar(timevA0, ax=vA0))
#     colorbars.append(h.colorbar(timeuDelta0, ax=uDelta0))
#     colorbars.append(h.colorbar(timevDelta0, ax=vDelta0))
#     resetticks(u0,uA0,uDelta0,v0,vA0,vDelta0)

#     h.suptitle('t={:.1e}'.format(i*T))

#     startt = time.time()
#     with torch.no_grad():
#         x0 = cdr0.predict(x0, T=T)
#     stopt = time.time()
#     speedrange = max(x0[0,0].max().item()-x0[0,0].min().item(),x0[0,1].max().item()-x0[0,1].min().item())
#     print('elapsed-time:{:.1f}'.format(stopt-startt)+
#             ', speedrange:{:.0f}'.format(speedrange))
#     if i > 0:
#         for colorbartmp in colorbars:
#             colorbartmp.remove()
#     plt.pause(1e-3)
