import numpy as np
from numpy import *
import torch

__all__ = ['initgen']

def _initgen_Neumann(mesh_size, freq=3):
    print("ms", mesh_size)
    def newpoint(lb,up,n):
        dx=(up-lb)/n
        newx=[]
        for i in range(n):
            x1=lb+(0.5+i)*dx
            newx.append(x1)
        newx=np.array(newx)
        return newx

    x=newpoint(-5, 5,mesh_size[0])
    y=newpoint(-5, 5,mesh_size[1])
    # x=np.linspace(-5, 5,mesh_size[0])
    # y=np.linspace(-5, 5,mesh_size[1])
    # x=newpoint(-20, 20,mesh_size[0])
    # y=newpoint(-20, 20,mesh_size[1])
    
    #return np.ones((mesh_size[0], mesh_size[1]))*3, np.cos(np.pi/5*x.reshape((mesh_size[0],1)))*np.cos(np.pi/5*y.reshape((1,mesh_size[1])))
    x1=np.ones((mesh_size[0], mesh_size[1]))*3
    #x1=np.sin(np.pi/10*x.reshape((mesh_size[0],1)))*np.cos(np.pi/5*y.reshape((1,mesh_size[1])))+0.1*np.cos(np.pi/2.5*y.reshape((1,mesh_size[1])))+0.5*np.cos(3*np.pi/10*x.reshape((mesh_size[0],1)))
    #x1=1.5*np.sin(np.pi/10*x.reshape((mesh_size[0],1)))*np.cos(np.pi/5*y.reshape((1,mesh_size[1])))+0.01*(y.reshape((1,mesh_size[1]))**3/3-25*y.reshape((1,mesh_size[1])))
    #x1=np.sin(np.pi/10*x.reshape((mesh_size[0],1)))*np.cos(np.pi/5*y.reshape((1,mesh_size[1])))+0.3*np.cos(np.pi/5*y.reshape((1,mesh_size[1])))+0.01*(y.reshape((1,mesh_size[1]))**3/3-25*y.reshape((1,mesh_size[1])))
    #x1=np.sin(np.pi/10*x.reshape((mesh_size[0],1)))*np.sin(3*np.pi/10*y.reshape((1,mesh_size[1])))+0.01*(y.reshape((1,mesh_size[1]))**3/3-25*y.reshape((1,mesh_size[1])))+0.5
    #x1=np.cos(np.pi/5*x.reshape((mesh_size[0],1)))*np.cos(6*np.pi/10*y.reshape((1,mesh_size[1])))+0.01*(y.reshape((1,mesh_size[1]))**3/3-25*y.reshape((1,mesh_size[1])))
    #print(np.abs(x1).max())
    #x1 = x1/np.abs(x1).max()
    #print(x1)
    
    #y=np.cos(20/np.pi*x.reshape((mesh_size[0],1)))*np.cos(20/np.pi*y.reshape((1,mesh_size[1])))
    #y = y/np.abs(y).max()
    y=np.cos(np.pi/5*x.reshape((mesh_size[0],1)))*np.cos(np.pi/5*y.reshape((1,mesh_size[1])))
    
    #y=np.cos(np.pi/5*x.reshape((mesh_size[0],1)))*np.sin(np.pi/10*y.reshape((1,mesh_size[1])))+0.01*(x.reshape((mesh_size[0],1))**3/3-25*x.reshape((mesh_size[0],1)))#+0.3*np.sin(5*np.pi/10*y.reshape((1,mesh_size[1])))
    #print(np.abs(y).max())
    #y = y/np.abs(y).max()
    #print(y)
    #y=np.cos(np.pi/5*x.reshape((mesh_size[0],1)))*np.sin(np.pi/10*y.reshape((1,mesh_size[1])))+0.3*np.cos(np.pi/2.5*y.reshape((1,mesh_size[1])))+0.5*np.cos(3*np.pi/10*x.reshape((mesh_size[0],1)))
    #y=np.cos(np.pi/5*x.reshape((mesh_size[0],1)))*np.sin(np.pi/10*y.reshape((1,mesh_size[1])))+0.3*np.cos(np.pi/2.5*y.reshape((1,mesh_size[1])))
    #y=np.sin(np.pi/10*y.reshape((1,mesh_size[1])))+0.3*np.sin(np.pi/10*x.reshape((mesh_size[0],1)))*np.cos(np.pi/2.5*y.reshape((1,mesh_size[1])))+0.3

    return x1, y

def _initgen(mesh_size, freq=3,boundary='Neumann', dtype=None, device=None):
    #print("boundary", boundary)

    x, y = _initgen_Neumann(mesh_size)
    return torch.from_numpy(x).to(dtype=dtype, device=device), torch.from_numpy(y).to(dtype=dtype, device=device)

def initgen(mesh_size, freq=3,boundary='Neumann', dtype=None, device=None, batch_size=1):
    #print("initgenb")
    xs = []
    #print("batch_size", batch_size)
    for k in range(int(batch_size/2)):
        result=_initgen(mesh_size, boundary=boundary, dtype=dtype, device=device)
        for i in range(2):
            xs.append(result[i])
    x = torch.stack(xs, dim=0)
    
    
    return x


# mesh_size = [20,20]
# T = 1e-2
# batch_size = 2
#
# if torch.cuda.is_available():
#     device = 'cuda:0'
# else:
#     device = None
# k=initgen(mesh_size=mesh_size, device=device, batch_size=batch_size)
# print(k)