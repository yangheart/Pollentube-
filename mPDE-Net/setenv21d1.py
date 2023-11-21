import warnings
import numpy as np
import torch
import aTEAM.pdetools as pdetools
import aTEAM.pdetools.example.burgers2d as burgers2d
import aTEAM.pdetools.example.cde2d as cde2d
import aTEAM.pdetools.example.rd2d as rd2d
import aTEAM.pdetools.example.cdr2d as cdr2d
import aTEAM.pdetools.example.brusslator as brusslator
import aTEAM.pdetools.example.Murrayll28 as Murrayll28
#import aTEAM.pdetools.example.test1 as test1


import polypde_ex21d1 as polypde
import conf,transform
import setcallback1 as setcallback

__all__ = ['setenv',]

def _set_model(options):
  

    globalnames = {} # variables need to be exported to training&testing script
    for k in options:
        globalnames[k[2:]] = options[k]

    globalnames['dtype'] = torch.float if globalnames['dtype'] == 'float' else torch.float64
    bound = options['--eps']*options['--cell_num']
    s = bound/options['--dx']

    if abs(s-round(s))>1e-6:
        warnings.warn('cell_num*eps/dx should be an integer but got'+str(s))
    if not globalnames['constraint'].upper() in ['FROZEN','MOMENT','FREE']:
        # using moment matrix with globalnames['constraint']-order approximation
        globalnames['constraint'] = int(globalnames['constraint'])
    ub=options["--upper_bound"]
    lb=options["--lower_bound"]

    globalnames['mesh_size'] = [round(s),]*2
    globalnames['mesh_bound'] = [[lb,]*2,[ub,]*2]
    globalnames['kernel_size'] = [options['--kernel_size'],]*2
    

    model = polypde.POLYPDE2D(
            dt=globalnames['dt'],
            dx=globalnames['dx'],
            kernel_size=globalnames['kernel_size'],
            max_order=globalnames['max_order'],
            constraint=globalnames['constraint'],
            channel_names=globalnames['channel_names'],
            hidden_layers=globalnames['hidden_layers'],
            scheme=globalnames['scheme']
            ) # build pde-net: a PyTorch module/forward network

    if globalnames['dtype'] == torch.float64:
        model.double()
    else:
        model.float()
    model.to(globalnames['device'])


    if globalnames['npseed'] < 0:
        npseed=np.random.randint(1e8)
        globalnames['npseed'] = npseed

    if globalnames['torchseed'] < 0:
        torchseed=np.random.randint(1e8)
        globalnames['torchseed'] = torchseed

    callback = setcallback.setcallback(options) 
    # some useful interface, see callback.record, callback.save
    callback.module = model

    return globalnames, callback, model

def setenv(options):
    """
    set training & testing environment
    Returns:
        globalnames(dict): variables need to be exported to training & testing script
        callback(function class): callback function for optimizer
        model(torch.nn.Module): PDE-Net, a torch forward neural network
        data_model(torch.nn.Module): a torch module for data generation
        sampling,addnoise(callable function): data down sample and add noise to data
    """
    globalnames, callback, model = _set_model(options)
    if options['--dataname'] == 'None':
        dataoptions = conf.setoptions(configfile="checkpoint-"+
                  options["--para_size"]+"/"+options['--name'], 
                  isload=True)
        dataoptions['--start_from'] = 80
        assert options['--cell_num']%dataoptions['--cell_num'] == 0
        dataoptions['--device'] = options['--device']
        dataoptions['--dtype'] = options['--dtype']
        _,_,data_model = _set_model(dataoptions)
        data_model.tiling = options['--cell_num']//dataoptions['--cell_num']
    mesh_size = list(m*globalnames['zoom'] for m in globalnames['mesh_size'])
    mesh_bound = globalnames['mesh_bound']
    viscosity = globalnames['viscosity']
    dx = globalnames['cell_num']*globalnames['eps']/mesh_size[0]
    
    if options['--dataname'].upper() == 'BURGERS':
        max_dt = globalnames['max_dt']
        data_model = burgers2d.BurgersTime2d(max_dt=max_dt,
                mesh_size=mesh_size,
                mesh_bound=mesh_bound,
                viscosity=viscosity,
                timescheme=globalnames['data_timescheme'],
                )
    elif options['--dataname'].upper() == 'HEAT':
        max_dt = globalnames['max_dt']
        data_model = cde2d.Heat(max_dt=max_dt,
                mesh_size=mesh_size,
                mesh_bound=mesh_bound,
                timescheme=globalnames['data_timescheme']
                )
        data_model.coe[0,2] = data_model.coe[2,0] = viscosity
    elif options['--dataname'].upper() == 'REACTIONDIFFUSION':
        max_dt = globalnames['max_dt']
        data_model = rd2d.RDTime2d(max_dt=max_dt,
                mesh_size=mesh_size,
                mesh_bound=mesh_bound,
                viscosity=viscosity,
                beta=1,
                timescheme=globalnames['data_timescheme']
                )
    elif options['--dataname'].upper() == 'CDR':
        max_dt = globalnames['max_dt']
        data_model = cdr2d.CDRTime2d(max_dt=max_dt,
                mesh_size=mesh_size,
                mesh_bound=mesh_bound,
                viscosity=viscosity,
                beta=1,
                timescheme=globalnames['data_timescheme'],
                )
    elif options['--dataname'].upper() == 'BRUSSLATOR':
        max_dt = globalnames['max_dt']
        data_model = brusslator.Brusselator(max_dt=max_dt,#brusslator.Brusselator(max_dt=max_dt,
                mesh_size=mesh_size,
                mesh_bound=mesh_bound,
                viscosity=viscosity,
                beta=1,
                timescheme=globalnames['data_timescheme'],
                )
    elif options['--dataname'].upper() == "MURRAYLL28":
        max_dt = globalnames['max_dt']
        data_model = Murrayll28.Murrayll28(max_dt=max_dt,
                mesh_size=mesh_size,
                mesh_bound=mesh_bound,
                timescheme=globalnames['data_timescheme'],
                )
      
    data_model.to(device=model.device)
    
    if globalnames['dtype'] == torch.float64:
        data_model.double()
    else:
        data_model.float()
    sampling = transform.Compose(
            transform.DownSample(mesh_size=globalnames['mesh_size']),
            )
    

    addnoise = transform.AddNoise(start_noise=options['--start_noise'], end_noise=options['--end_noise'])
    

    ##test

    addnoiset = transform.AddNoise(start_noise=options['--start_noise'], end_noise=options['--end_noise'])
    
    

    return globalnames, callback, model, data_model, sampling, addnoise, addnoiset

def data(model, data_model, globalnames, sampling, addnoise, block, data_start_time=1):
    """
    generate data 
    Returns:
        u_obs(list of torch.tensor): observed data
        u_true(list of torch.tensor): underlying data
        u(list of torch.tensor): underlying high resolution data
    """
    freq, batch_size, device, dtype, dt = \
            globalnames['freq'], globalnames['batch_size'], \
            globalnames['device'], globalnames['dtype'], globalnames['dt']
    
    
    initrange = 2
    
    u0 = pdetools.initcsc2d.initgen(mesh_size=data_model.mesh_size, 
            freq=freq, 
            batch_size=model.channel_num*batch_size, 
            device=device, 
            dtype=dtype)
    
    u0 += 2*(torch.rand(model.channel_num*batch_size,1,1,dtype=dtype,device=device)+0.5)
    u0 = u0.view([batch_size, model.channel_num]+data_model.mesh_size.tolist())
    u0 = u0.view([batch_size, model.channel_num]+data_model.mesh_size.tolist())
    

    with torch.no_grad():
        if batch_size>1:
            B = batch_size//2
            u0[B:] = data_model(u0[:B], T=0.2)
            
              
    u = [u0,]
    u0_true = sampling(u0)
    u_true = [u0_true,]
    u0_obs = addnoise(u0_true)
    u_obs = [u0_obs,]

    stepnum = block if block>=1 else 1
    ut = u0
  
    with torch.no_grad():
        for k in range(stepnum):
            ut = data_model(ut, T=dt)
            u.append(ut)
            ut_true = sampling(ut)
            u_true.append(ut_true)
            _, ut_obs = addnoise(u0_true, ut_true)
            u_obs.append(ut_obs)

    return u_obs, u_true, u


def testdata(model, data_model, globalnames, sampling, addnoiset, block, data_start_time=1):
    """
    generate test data 
    Returns:
        u_obs(list of torch.tensor): observed data
        u_true(list of torch.tensor): underlying data
        u(list of torch.tensor): underlying high resolution data
    """

    freq, device, dtype, dt = \
            globalnames['freq'], \
            globalnames['device'], globalnames['dtype'], globalnames['dt']
    

    batch_size = 8
    initshift = (1 if (globalnames['dataname']=='reactiondiffusion') else 2)
    
    
    # 
 
    
    
    u1 = pdetools.initb2815.initgen(mesh_size=data_model.mesh_size,ind="0", 
            batch_size=model.channel_num*batch_size, 
            device=device, 
            dtype=dtype)
    u2 = pdetools.initb2815.initgen(mesh_size=data_model.mesh_size,ind="1", 
            batch_size=model.channel_num*batch_size, 
            device=device, 
            dtype=dtype)
    u3 = pdetools.initb2815.initgen(mesh_size=data_model.mesh_size,ind="5", 
            batch_size=model.channel_num*batch_size, 
            device=device, 
            dtype=dtype)
    
    u0 = u1.view([batch_size, model.channel_num]+data_model.mesh_size.tolist())
    u1 = u2.view([batch_size, model.channel_num]+data_model.mesh_size.tolist())
    u2 = u3.view([batch_size, model.channel_num]+data_model.mesh_size.tolist())
    
    with torch.no_grad():
        if batch_size>1:
            B = batch_size//3
            u0[B:2*B] = data_model(u1[:B], T=0)
            u0[2*B:B*3] = data_model(u2[:B], T=0)
             
    u = [u0,]
    u0_true = sampling(u0)
    u_true = [u0_true,]
    u0_obs = addnoiset(u0_true)
    u_obs = [u0_obs,]
    stepnum = block if block>=1 else 1
    ut = u0
 
    with torch.no_grad():
        for k in range(stepnum):
          ut = data_model(ut, T=dt)
          u.append(ut)
          ut_true = sampling(ut)
          u_true.append(ut_true)
          _, ut_obs = addnoiset(u0_true, ut_true)
          u_obs.append(ut_obs)

    return u_obs, u_true, u

def cvtestdata(model, data_model, globalnames, sampling, addnoiset, block, data_start_time=1):
    """
    generate cross validate data 
    Returns:
        u_obs(list of torch.tensor): observed data
        u_true(list of torch.tensor): underlying data
        u(list of torch.tensor): underlying high resolution data
    """
    freq, device, dtype, dt = \
            globalnames['freq'], \
            globalnames['device'], globalnames['dtype'], globalnames['dt']
    

    batch_size = 3
    initshift = (1 if (globalnames['dataname']=='reactiondiffusion') else 2)
    
    u1 = pdetools.initb2815.initgen(mesh_size=data_model.mesh_size,ind="3", 
            batch_size=model.channel_num*batch_size, 
            device=device, 
            dtype=dtype)
    u2 = pdetools.initb2815.initgen(mesh_size=data_model.mesh_size,ind="4", 
            batch_size=model.channel_num*batch_size, 
            device=device, 
            dtype=dtype)
    u3 = pdetools.initb2815.initgen(mesh_size=data_model.mesh_size,ind="2", 
            batch_size=model.channel_num*batch_size, 
            device=device, 
            dtype=dtype)
    


    
    u0 = u1.view([batch_size, model.channel_num]+data_model.mesh_size.tolist())
    u1 = u2.view([batch_size, model.channel_num]+data_model.mesh_size.tolist())
    u2 = u3.view([batch_size, model.channel_num]+data_model.mesh_size.tolist())
    
    
    with torch.no_grad():
        if batch_size>1:
            B = batch_size//3
            u0[B:B*2] = data_model(u1[:B], T=0)
            u0[B*2:B*3] = data_model(u2[:B], T=0)
          
             
    u = [u0,]
    u0_true = sampling(u0)
    u_true = [u0_true,]
    u0_obs = addnoiset(u0_true)
    u_obs = [u0_obs,]
    stepnum = block if block>=1 else 1
    ut = u0
 
    with torch.no_grad():
        for k in range(stepnum):
            ut = data_model(ut, T=dt)
            u.append(ut)
            ut_true = sampling(ut)
            u_true.append(ut_true)
            _, ut_obs = addnoiset(u0_true, ut_true)
            u_obs.append(ut_obs)

    return u_obs, u_true, u

def _sparse_loss(model):
    """
    SymNet regularization
    """
    loss = 0
    s = 1e-3
    for p in model.expr_params():
        p = p.abs()
        loss = loss + ((p<s).to(p)*0.5/s*p**2).sum() + ((p>=s).to(p)*(p-s/2)).sum()

    return loss


def _moment_loss(model):
    """
    Moment regularization
    """
    loss = 0
    s = 1e-2
    for p in model.diff_params():
        p = p.abs()
        loss = loss + ((p<s).to(p)*0.5/s*p**2).sum() + ((p>=s).to(p)*(p-s/2)).sum()

    return loss

def loss(model, u_obs, utest_obs, globalnames, block,layerweight=None):
    if layerweight is None:
        layerweight = [0,]*stepnum
        layerweight[-1] = 1

    dt = globalnames['dt']
    dx = globalnames['dx']
    stableloss = 0
    dataloss = 0
    sparseloss = _sparse_loss(model)
    momentloss = _moment_loss(model)
    
    stepnum = block if block>=1 else 1
    

    ut = u_obs[0]
    for steps in range(1,stepnum+1):
        uttmp = model(ut, T=dt)
        upad = model.fd00.pad(ut)

        stableloss = stableloss+(model.relu(uttmp-model.maxpool(upad))**2+
                model.relu(-uttmp-model.maxpool(-upad))**2).sum()
        
        dataloss = dataloss+\
                layerweight[steps-1]*torch.mean(((uttmp-u_obs[steps])/dt)**2)
       
        ut = uttmp
    
    with torch.no_grad(): 
        testloss=0
        utpre=utest_obs[0]
        for steps in range(1,stepnum+1):
          uttmpre = model(utpre, T=dt)
          
          testloss = testloss+\
                    (layerweight[steps-1]*torch.mean(((uttmpre-utest_obs[steps])/dt)**2)).item()
       
          utpre = uttmpre
            
    return stableloss, dataloss, sparseloss, momentloss, testloss