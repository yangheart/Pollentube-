import numpy as np
from numpy import *
import torch

__all__ = ['initgen']

def _initgen_Neumann(mesh_size, freq=3):
    #print("ms", mesh_size)
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
    x=x.reshape((mesh_size[0],1))
    y=y.reshape((1,mesh_size[1]))
    x1=0
    y1=0
    for i in range(-freq,freq+1):
      for j in range(-freq,freq+1):
        coe = random.randn(1,8)
        coe1 = random.randn(1,16)
        #print("coe", coe)
        x1+=\
        coe[0][0]*cos(2*i*x)*cos(2*j*y) + coe[0][1]*sin((2*i+1)*x)*sin((2*j+1)*y) + coe[0][2]*sin((2*i+1)*x)*cos(2*j*y) + coe[0][3]*sin((2*j+1)*y)*cos(2*i*x)\
        #+ coe1[0][0]*4*np.log(4+(y**3/3-25*y)/80) + coe1[0][1]*4*np.log(4+(x**3/3-25*x)/80)\
        #+ coe1[0][2]*np.exp(cos(2*x)) + coe[0][3]*np.exp(sin((2+1)*x))\
        #+coe1[0][4]*4*(np.sin(3*np.pi*x/10)+2)/np.log(4+(y**3/3-25*y)/100)+coe1[0][5]*4*(np.cos(2*np.pi*x/5)+2)/np.log(4+(y**3/3-25*y)/100)\
        #+coe1[0][6]*(np.sin(3*np.pi*y/10)+2)*np.log(3+(x**3/3-25*x)/80)+coe1[0][7]*(np.cos(2*np.pi*y/5)+2)/np.log(3+(x**3/3-25*x)/80)\
        #+coe[0][23]*(np.sin(3*np.pi*x/10)+5)/(-np.cos(np.pi/5*y)+4)+coe[0][24]*(np.sin(3*np.pi*y/10)+5)/(-np.cos(np.pi/5*x)+4)


        y1+=\
        coe[0][4]*cos(2*i*x)*cos(2*j*y) + coe[0][5]*sin((2*i+1)*x)*sin((2*j+1)*y) + coe[0][6]*sin((2*i+1)*x)*cos(2*j*y) + coe[0][7]*sin((2*j+1)*y)*cos(2*i*x)\
        # +coe1[0][8]*1*np.log(4+(y**3/3-25*y)/80) + coe1[0][9]*1*np.log(4+(x**3/3-25*x)/80)\
        # +coe1[0][10]*np.exp(cos(2*x))+coe1[0][11]*np.exp(sin((2+1)*x))\
      #  +coe1[0][12]*1*(np.sin(3*np.pi*x/10)+2)/np.log(4+(y**3/3-25*y)/100)+coe1[0][13]*1*(np.cos(2*np.pi*x/5)+2)/np.log(4+(y**3/3-25*y)/100)\
        # +coe1[0][14]*(np.sin(3*np.pi*y/10)+2)*np.log(3+(x**3/3-25*x)/80)+coe1[0][15]*(np.cos(2*np.pi*y/5)+2)/np.log(3+(x**3/3-25*x)/80)\
        #+coe[0][25]*(np.sin(3*np.pi*x/10)+5)/(-np.cos(np.pi/5*y)+4)+coe[0][26]*(np.sin(3*np.pi*y/10)+5)/(-np.cos(np.pi/5*x)+4)

    return x1/(np.abs(x1).max())-1.5,y1/(np.abs(y1).max())+1.5

def _initgen(mesh_size, freq=3,boundary='Neumann', dtype=None, device=None):
    #print("boundary", boundary)

    x, y = _initgen_Neumann(mesh_size)
    return torch.from_numpy(x).to(dtype=dtype, device=device), torch.from_numpy(y).to(dtype=dtype, device=device)

def initgen(mesh_size, freq=3,boundary='Neumann', dtype=None, device=None, batch_size=1):
    print("initgenb")
    xs = []
    #print("batch_size", batch_size)
    for k in range(int(batch_size/2)):
        result=_initgen(mesh_size, boundary=boundary, dtype=dtype, device=device)
        #print("result", result)
        for i in range(2):
            xs.append(result[i])
    x = torch.stack(xs, dim=0)


    return x
