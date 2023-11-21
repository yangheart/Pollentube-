"""numpy function interface for torch"""
import torch
from functools import reduce
import warnings
import numpy as np

from .PGManager import ParamGroupsManager

__all__ = ['NumpyFunctionInterface',]

class NumpyFunctionInterface(ParamGroupsManager):
    """
    Interfaces class for representing torch forward & backward procedures 
           as Numpy functions. 

    .. warning::
    If you are going to change options of one of self.param_groups with 
        always_refresh=False, please use self.set_options. This is because, for 
        example, any changes on 'grad_proj's will have impact on self.fprime(x), 
        even for the same input x; so do 'isfrozen's, 'x_proj's. So do changes 
        on 'x_proj's, 'isfrozen's.

    .. warning::
    Right now all parameters have to be dense Variable and their dtype 
        (float or double) have to be the same. This will be improved in the 
        future.

    Arguments:
        params (iterable): See ParamGroupsManager.__doc__
        forward (callable): callable forward(**kw)
            torch forward procedure, return a :class:`torch.Tensor`
        isfrozen (bool): whether parameters should be frozen, if you set 
            isfrozen=True, as a result, grad of this param_group would be 
            set to be 0 after calling self.fprime(x).
        x_proj (callable): callable x_proj(param_group['params']). 
            It is similar to nn.module.register_forward_pre_hook(x_proj) but 
            they are not have to be the same. Each time you call 
            self.set_options(idx,{'x_proj':x_proj}), self._x_cache will be 
            set to be None.
            It can be used to make parameters to satisfied linear constraint. 
            Wether isfrozen or not, x_proj&grad_proj will go their own way.
        grad_proj (callable): callable grad_proj(param_group['params']).
            It is similar to nn.module.register_backward_hook(grad_proj).
            grad_proj(param_group['params']) should project gradients of 
            param_group['params'] to the constrained linear space if needed.
        always_refresh (bool): If always_refresh=True, then any changes on 
            forward & backward procedure is OK. We recommand you to set 
            always_refresh=True unless you are familiar with 
            :class:`NumpyFunctionInterface`.
            When always_refresh=False, NumpyFunctionInterface will cache 
            parameters for fast forward & backward.
        **kw (keyword args): other options for parameter groups
    """

    def __init__(self, params, forward, *, 
            isfrozen=False, x_proj=None, grad_proj=None, always_refresh=True, 
            **kw):
 
        defaults = dict(isfrozen=isfrozen, 
                x_proj=x_proj, grad_proj=grad_proj, **kw)
            
        super(NumpyFunctionInterface, self).__init__(params, defaults)
        self.dtype = next(self.params).data.cpu().numpy().dtype
        self._forward = forward
        self.options_refresh()
        self.always_refresh = always_refresh

    def options_refresh(self):
        """
        Any changes on 'isfrozen's, 'x_proj's, 'grad_proj's, self._forward will 
        have impact on self.f, self.fprime. Call this function to keep them 
        safe when you apply any changes on options.
        """
        self._need_backward = True
        self._grad_cache = None
        self._x_cache = None
        self._loss = None
        self._numel = None

    @staticmethod
    def _proj_check(kw):
        if not kw['isfrozen'] and None in set([kw['x_proj'],kw['grad_proj']]):
            if not (kw['x_proj'] is None and kw['grad_proj'] is None):
                print(kw)
                warnings.warn("Exactly one of {x_proj,grad_proj} is not None, "
                        "and the parameters are not set to be frozen, "
                        "make sure what you are doing now.")
        return None
    def set_options(self, idx, **kw):
        """
        A safe way to update idx_th param_group's options.
        """
        self.param_groups[idx].update(**kw)
        NumpyFunctionInterface._proj_check(self.param_groups[idx])
        self.options_refresh()
    def add_param_group(self, param_group):
        super(NumpyFunctionInterface, self).add_param_group(param_group)
        param_group_tmp = self.param_groups[-1]
        # check consistency of x_proj,grad_proj,isfrozen
        NumpyFunctionInterface._proj_check(param_group_tmp)
        # check is_leaf,requires_grad
        for _,p in param_group_tmp['params'].items():
            if not p.is_leaf:
                raise ValueError("can't manage a non-leaf Tensor")
            if not p.requires_grad:
                raise ValueError("managing a Tensor that does not "
                        "require gradients")
        self.options_refresh()

    @property
    def forward(self):
        """
        A safe way to get access of self._forward.
        When you use property NumpyFunctionInterface.forward, I expect you are 
        going to do some modifications on self._forward, like: 
            self.forward.property = value
        in this case, we should call self.options_refresh() to keep self.f and 
        self.fprime safe. 
        """
        print("NFI forward1")
        self.options_refresh()
        return self._forward
    @forward.setter
    def forward(self, v):
        print("NFI forward2")
        self.options_refresh()
        self._forward = v

    def numel(self):
        if not self._numel is None:
            return self._numel
        return reduce(lambda a,p: a+p.numel(), self.params, 0)

    def _all_x_proj(self):
        for param_group in self.param_groups:
            x_proj = param_group['x_proj']
            if not x_proj is None:
                x_proj(param_group['params'])

    def _all_grad_proj(self):
        for param_group in self.param_groups:
            grad_proj = param_group['grad_proj']
            if not grad_proj is None:
                grad_proj(param_group['params'])

    # if you do self.flat_param = x; y = self.flat_param; 
    # np.array_equal(x,y) may not be True.
    # because of 'x_proj's,'grad_proj's may have impact on self.flat_param
    @property
    def flat_param(self):
        views = []
        self._all_x_proj()
        for p in self.params:
            view = p.data.view(-1).cpu()
            views.append(view)
        return torch.cat(views,0).numpy()
    @flat_param.setter
    def flat_param(self, x):
        assert isinstance(x, np.ndarray)
        assert x.size == self.numel()
        x = x.astype(dtype=self.dtype,copy=False)
        offset = 0
        for isfrozen,p in self.params_with_info('isfrozen'):
            numel = p.numel()
            if not isfrozen:
                p_tmp = torch.from_numpy(x[offset:offset+numel])\
                        .view_as(p)
                p.data.copy_(p_tmp)
            offset += numel
        self._all_x_proj()

    
    def _flat_grad(self):
        views = []
        self._all_grad_proj()
        order=0
        for isfrozen, p in self.params_with_info('isfrozen'):
            if isfrozen or p.grad is None:
                view = torch.zeros(p.numel(), dtype=p.dtype)
            else:
                size=p.size()
                
                if len(size)==2 and size[0]==2:
                    
                    if size[1]==12:
                      p.grad.data[0,0:] = torch.zeros(*p[0,0:].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,0:] = torch.zeros(*p[1,0:].shape,dtype=p.dtype,device=p.device)                     
                      view = p.grad.data.view(-1).cpu() #u^2

                    elif size[1]==13:
                      p.grad.data[0,0:] = torch.zeros(*p[0,0:].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,0:] = torch.zeros(*p[1,0:].shape,dtype=p.dtype,device=p.device)                      
                      view = p.grad.data.view(-1).cpu() #v^2

                    
                    elif size[1]==14:
                      p.grad.data[0,1:] = torch.zeros(*p[0,1:].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,1:] = torch.zeros(*p[1,1:].shape,dtype=p.dtype,device=p.device)
                      view = p.grad.data.view(-1).cpu()
                    
                    elif size[1]==15:
                      p.grad.data[0,0:6] = torch.zeros(*p[0,0:6].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[0,7:] = torch.zeros(*p[0,7:].shape,dtype=p.dtype,device=p.device)   
                      p.grad.data[1,0:6] = torch.zeros(*p[1,0:6].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,7:] = torch.zeros(*p[1,7:].shape,dtype=p.dtype,device=p.device)
                      view = p.grad.data.view(-1).cpu() #frac1
                      
                    elif size[1]==16:
                      p.grad.data[0,1:6] = torch.zeros(*p[0,1:6].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[0,7:] = torch.zeros(*p[0,7:].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,1:6] = torch.zeros(*p[0,1:6].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,7:] = torch.zeros(*p[0,7:].shape,dtype=p.dtype,device=p.device)
                      view = p.grad.data.view(-1).cpu() #frac2

                    elif size[1]==17:
                      p.grad.data[0,1:6] = torch.zeros(*p[0,1:6].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[0,7:15] = torch.zeros(*p[0,7:15].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,1:6] = torch.zeros(*p[0,1:6].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,7:15] = torch.zeros(*p[0,7:15].shape,dtype=p.dtype,device=p.device)                      
                      view = p.grad.data.view(-1).cpu() #nonlinear*linear

                    elif size[1]==18:
                      p.grad.data[0,1:] = torch.zeros(*p[0,1:].shape,dtype=p.dtype,device=p.device)
                      #p.grad.data[0,13:] = torch.zeros(*p[0,13:].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,0:15] = torch.zeros(*p[0,0:15].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,16:] = torch.zeros(*p[0,16:].shape,dtype=p.dtype,device=p.device) 
                      view = p.grad.data.view(-1).cpu() #nonlinear*linear

                    elif size[1]==19:
                      p.grad.data[0,0:6] = torch.zeros(*p[0,0:6].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[0,7:] = torch.zeros(*p[0,7:].shape,dtype=p.dtype,device=p.device)
                      #p.grad.data[0,14:] = torch.zeros(*p[0,14:].shape,dtype=p.dtype,device=p.device)                      
                      p.grad.data[1,0:14] = torch.zeros(*p[0,0:14].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,15:] = torch.zeros(*p[0,15:].shape,dtype=p.dtype,device=p.device) 
                      view = p.grad.data.view(-1).cpu()
     
                    else:

                      p.grad.data[0,1:6] = torch.zeros(*p[0,1:6].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[0,7:16] = torch.zeros(*p[0,7:16].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,1:6] = torch.zeros(*p[1,1:6].shape,dtype=p.dtype,device=p.device)
                      p.grad.data[1,7:] = torch.zeros(*p[1,7:].shape,dtype=p.dtype,device=p.device)
                      #p.grad.data[1,16:] = torch.zeros(*p[1,16:].shape,dtype=p.dtype,device=p.device)
                      view = p.grad.data.view(-1).cpu()
                    
                elif len(size)==2 and size[0]==1:
                  if order==22:                   
                    p.grad.data[0,5] = p.grad.data[0,3] #diffusion u
                    p.grad.data[0,9]=0
                    p.grad.data[0,11] = p.grad.data[0,9] #diffusion v
                    p.grad.data[0,1] = 0
                    p.grad.data[0,2] = 0
                    p.grad.data[0,4] = 0
                    p.grad.data[0,7] = 0
                    p.grad.data[0,8] = 0
                    p.grad.data[0,10] = 0
                    p.grad.data[0,12] = 0
                    p.grad.data[0,13] = 0
                    p.grad.data[0,14] = 0
                    p.grad.data[0,15] = 0                  
                    p.grad.data[0,17] = 0
                    view = p.grad.data.view(-1).cpu()

                  else:
                    p.grad.data[0,3]=0
                    p.grad.data[0,5] = p.grad.data[0,3]                    
                    p.grad.data[0,11] = p.grad.data[0,9]
                    p.grad.data[0,1] = 0
                    p.grad.data[0,2] = 0
                    p.grad.data[0,4] = 0
                    p.grad.data[0,7] = 0
                    p.grad.data[0,8] = 0
                    p.grad.data[0,10] = 0
                    p.grad.data[0,12] = 0
                    p.grad.data[0,13] = 0
                    p.grad.data[0,14] = 0
                    p.grad.data[0,15] = 0                    
                    p.grad.data[0,17] = 0
                    view = p.grad.data.view(-1).cpu()

                   
                else:
                  if order==7 or order==25 or order==9 or order==27:         
                      p.grad.data[0] = 0
                      p.grad.data[1] = 0
                      view = p.grad.data.view(-1).cpu()
                  
                  elif order==19 or order==37 or order==21 or order==39:
                      p.grad.data[0] = 0
                      p.grad.data[1] = 0
                      view = p.grad.data.view(-1).cpu()
                  
                  else:
                      view = p.grad.data.view(-1).cpu()
                     

            views.append(view)
            order+=1
        return torch.cat(views, 0).numpy()

    def f(self, x, *args, **kw):
        """
        self.f(x) depends on self.flat_param and self.forward
        """
        if self.always_refresh:
            self.options_refresh()
        self.flat_param = x
        _x_cache = self.flat_param
        if self._loss is None or not np.array_equal(_x_cache, self._x_cache):
            self._x_cache = _x_cache
            self._loss = self._forward()
            self._need_backward = True
        return self._loss.item()
    def fprime(self, x, always_double=True, *args, **kw):
      
        self.f(x)
        if self._need_backward:
            self.zero_grad()
            self._loss.backward()
     
            self._grad_cache = self._flat_grad()
        self._need_backward = False
        if always_double:
            return self._grad_cache.astype(np.float64)
        else:
            return self._grad_cache
