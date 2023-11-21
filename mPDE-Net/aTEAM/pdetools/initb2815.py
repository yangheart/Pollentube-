import numpy as np
from numpy import *
import torch

__all__ = ['initgen']

def _initgen_Neumann(mesh_size, ind):
   
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
    
    if ind=="0":
     x1=(-np.sin(np.pi*x/10)+3)*np.log(4+(y**3/3-25*y)/80)
     y1=(np.cos(np.pi*x/5)+3)/(2-np.sin(np.pi/10*y))
    elif ind=="1":
      x1=(np.sin(np.pi*x/10)+1.5)*(np.cos(np.pi/5*y)+3)
      y1=(np.sin(3*np.pi*y/10)+3)/np.log(4+(x**3/3-25*x)/80)
    elif ind=="2":
      x1=(-np.sin(np.pi*x/10)+2)*np.log(4+(y**3/3-25*y)/100)
      y1=(np.sin(np.pi*y/10)+1)*(np.cos(np.pi/5*x)+1.5)
    elif ind=="3":
        x1=(50*y**2-y**4+4)/(800*(1.2-np.cos(pi/5*x)))+4
        y1=1/800*(50*y**2-y**4+4)*(2+np.cos(pi/5*x))+1
    elif ind=="4":
        x1=(np.cos(np.pi*x/5)+1.5)*(-np.sin(np.pi/10*y)+2)
        y1=(np.sin(np.pi*y/10)+3)/np.log(5+(x**3/3-25*x)/60)
    elif ind=="5":
        x1=0.2*np.log((x**3/3-25*x)/100+4)+0.5*cos(pi*y/5)*sin(3*pi*y/10)
        y1=cos(pi*x/5)*sin(pi*y/10)+2
    



  
    

    
    return x1, y1

def _initgen(mesh_size, ind,boundary='Neumann', dtype=None, device=None):
 

    x, y = _initgen_Neumann(mesh_size,ind)
    return torch.from_numpy(x).to(dtype=dtype, device=device), torch.from_numpy(y).to(dtype=dtype, device=device)

def initgen(mesh_size, ind,boundary='Neumann', dtype=None, device=None, batch_size=1):
    xs = []

    for k in range(int(batch_size/2)):
        result=_initgen(mesh_size, ind,boundary=boundary, dtype=dtype, device=device)
        for i in range(2):
            xs.append(result[i])
    x = torch.stack(xs, dim=0)


    return x



