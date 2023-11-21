import numpy as np
import torch
import torch.nn as nn
from ..stepper import TimeStepper
from ..upwind import _pad
from ..init import initgen
from ..spectral import *

class RDTime2d(nn.Module, TimeStepper):
    """
    2d Reaction Diffusion equation
    \partial_t u = 0.1\Delta u+(1-A) u+\beta A v
    \partial_t v = 0.1\Delta v-\beta A u+(1-A) v
    where A=u^2+v^2
    """
    @property
    def timescheme(self):
        return self._timescheme
    def RightHandItems(self, u, viscosity=None, beta=None, force=None, **kw):
        """
        u[...,0,y,x],u[...,1,y,x]
        """
        viscosity = (self.viscosity if viscosity is None else viscosity)
        beta = (self.beta if beta is None else beta)
        A = u[...,:1,:,:]**2+u[...,1:,:,:]**2
        print("u",u)
        upad = _pad(u, [1,1,1,1], mode='wrap')
        print("upad", upad)
        rhi = (viscosity/self.dx**2)*(upad[...,2:,1:-1]+upad[...,:-2,1:-1]
                +upad[...,1:-1,2:]+upad[...,1:-1,:-2]-4*upad[...,1:-1,1:-1])
        rhi[...,:1,:,:] += (1-A)*u[...,:1,:,:]+beta*A*u[...,1:,:,:]
        rhi[...,1:,:,:] -= beta*A*u[...,:1,:,:]-(1-A)*u[...,1:,:,:]
        if not force is None:
            rhi = rhi+force
        elif not self.force is None:
            rhi = rhi+self.force
        return rhi
    def __init__(self, max_dt, mesh_size, mesh_bound=((0,0),(1,1)), viscosity=0.1, beta=1, timescheme='rk2', force=None):
        super(RDTime2d, self).__init__()
        self.max_dt = max_dt
        self.mesh_size = np.array(mesh_size).copy()
        self.mesh_bound = np.array(mesh_bound).copy()
        dx0,dx1 = (self.mesh_bound[1]-self.mesh_bound[0])/self.mesh_size
        assert abs(dx0-dx1)<1e-10
        self.dx = dx0
        self.viscosity = viscosity
        self.beta = beta
        self._timescheme = timescheme
        self.force = force
    def forward(self, inputs, T, **kw):
        return self.predict(inputs, T, **kw)

def test_RD2d(viscosity=0.1, beta=1, max_dt=1e-5):
    import aTEAM.pdetools as pdetools
    import aTEAM.pdetools.example.rd2d as rd2d
    import torch
    import matplotlib.pyplot as plt
    import time
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = None
    mesh_size = [64,64]
    T = 1e-2
    batch_size = 2

    init = pdetools.init.initgen(mesh_size=mesh_size, freq=4, device=device, batch_size=2*batch_size)*2
    init += init.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]*\
            torch.randn(2*batch_size,1,1, dtype=torch.float64, device=device)*\
            torch.rand(2*batch_size,1,1, dtype=torch.float64, device=device)*2
    rd0 = rd2d.RDTime2d(max_dt=max_dt, mesh_size=mesh_size, mesh_bound=((0,0),(2*np.pi,2*np.pi)), viscosity=viscosity, beta=beta, timescheme='rk2')
    h = plt.figure()
    u0 = h.add_subplot(2,3,1,aspect='equal')
    uA0 = h.add_subplot(2,3,2,aspect='equal')
    uDelta0 = h.add_subplot(2,3,3,aspect='equal')
    v0 = h.add_subplot(2,3,4,aspect='equal')
    vA0 = h.add_subplot(2,3,5,aspect='equal')
    vDelta0 = h.add_subplot(2,3,6,aspect='equal')
    def resetticks(*argv):
        for par in argv:
            par.set_xticks([]); par.set_yticks([])
    resetticks(u0,uA0,uDelta0,v0,vA0,vDelta0)
    x0 = init.view([batch_size,2,]+mesh_size)

    Y,X = np.mgrid[0:1:(mesh_size[0]+1)*1j,0:1:(mesh_size[1]+1)*1j]
    Y,X = Y[:-1,:-1],X[:-1,:-1]
    for i in range(20):
        u0.clear();uA0.clear();uDelta0.clear();v0.clear();vA0.clear();vDelta0.clear();

        x0pad = _pad(x0, [1,1,1,1], mode='wrap')
        deltax0 = (viscosity/rd0.dx**2)*(x0pad[...,2:,1:-1]+x0pad[...,:-2,1:-1]
                +x0pad[...,1:-1,2:]+x0pad[...,1:-1,:-2]-4*x0pad[...,1:-1,1:-1])
        A = x0[...,:1,:,:]**2+x0[...,1:,:,:]**2
        uA0rhi = (1-A)*x0[...,:1,:,:]+beta*A*x0[...,1:,:,:]
        vA0rhi = -beta*A*x0[...,:1,:,:]+(1-A)*x0[...,1:,:,:]
        timeu0 = u0.imshow(x0[0,0].data.cpu().numpy()[::-1], cmap='jet')
        timev0 = v0.imshow(x0[0,1].data.cpu().numpy()[::-1], cmap='jet')
        timeuA0 = uA0.imshow(uA0rhi[0,0].data.cpu().numpy()[::-1],cmap='jet')
        timevA0 = vA0.imshow(vA0rhi[0,0].data.cpu().numpy()[::-1],cmap='jet')
        timeuDelta0 = uDelta0.imshow(deltax0[0,0].data.cpu().numpy()[::-1],cmap='jet')
        timevDelta0 = vDelta0.imshow(deltax0[0,1].data.cpu().numpy()[::-1],cmap='jet')
        colorbars = []
        colorbars.append(h.colorbar(timeu0, ax=u0))
        colorbars.append(h.colorbar(timev0, ax=v0))
        colorbars.append(h.colorbar(timeuA0, ax=uA0))
        colorbars.append(h.colorbar(timevA0, ax=vA0))
        colorbars.append(h.colorbar(timeuDelta0, ax=uDelta0))
        colorbars.append(h.colorbar(timevDelta0, ax=vDelta0))
        resetticks(u0,uA0,uDelta0,v0,vA0,vDelta0)

        h.suptitle('t={:.1e}'.format(i*T))

        startt = time.time()
        with torch.no_grad():
            x0 = rd0.predict(x0, T=T)
        stopt = time.time()
        speedrange = max(x0[0,0].max().item()-x0[0,0].min().item(),x0[0,1].max().item()-x0[0,1].min().item())
        print('elapsed-time:{:.1f}'.format(stopt-startt)+
                ', speedrange:{:.0f}'.format(speedrange))
        if i > 0:
            for colorbartmp in colorbars:
                colorbartmp.remove()
        plt.pause(1e-3)
