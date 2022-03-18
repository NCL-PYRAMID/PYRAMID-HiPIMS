import torch
import math
import sys
import os
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

try:
    from SWE_CUDA import Godunov
    from MULTI_POL_CUDA import Pollution_multi,Particle
except ImportError:
    from .SWE_CUDA import Godunov
    from .POL_CUDA import Pollution
    from .MULTI_POL_CUDA import Pollution_multi,Particle

def run(paraDict):
    # ===============================================
    # set the device
    # ===============================================
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    # ===============================================
    # get initial data
    # ===============================================
    # Case 2: init dem,h,qx,qy data by code
    tensorsize = paraDict['tensorsize']
    h_host = np.zeros(tensorsize)
    qx_host = np.zeros(tensorsize)
    qy_host = np.zeros(tensorsize)

    # ===============================================
    # set the tensors
    # ===============================================
    row = np.size(h_host,0)
    col = np.size(h_host,1)
    tensorsize = (row + 2, col + 2)

    mask = torch.ones(tensorsize, dtype=torch.int32, device=device) 
    mask *= 10
    mask[:,-2] = 30 
    mask[:,1] = 30
    mask[-2,:] = 30
    mask[1,:] = 30

    mask[0, :] = -9999
    mask[-1, :] = -9999
    mask[:, 0] = -9999
    mask[:, -1] = -9999

    z_host = torch.zeros(tensorsize,dtype=torch.float64)
    z = torch.zeros(tensorsize, device=device)
    h = torch.zeros(tensorsize, device=device)
    qx = torch.zeros(tensorsize, device=device)
    qy = torch.zeros(tensorsize, device=device)
    wl = torch.zeros(tensorsize, device=device)

    h[1:row+1,1:col+1] = torch.from_numpy(h_host)
    qx[1:row+1,1:col+1] = torch.from_numpy(qx_host)
    qy[1:row+1,1:col+1] = torch.from_numpy(qy_host)
    wl = z + h

    # ===============================================
    # set x, y 
    # ===============================================
    row = torch.arange(row+2)
    col = torch.arange(col+2)
    y, x = torch.meshgrid(row, col)
    x, y = x.type(torch.float64), y.type(torch.float64)
    dx = paraDict['dx']
    x = (x - 0.5) * dx
    y = (y - 0.5) * dx  
    z_host[x<2.5]=2+y[x<2.5]*0.1-0.2-0.2-0.2
    z_host[y<0.6]=2+y[y<0.6]*0.1-0.2-0.2
    z_host[y>0.9]=2+y[y>0.9]*0.1-0.2-0.2
    z_host[x<2]=2-x[x<2]*0.1+y[x<2]*0.1
    z_host[x>2.5]=2-(4.5-x[x>2.5])*0.1+y[x>2.5]*0.1

    z[1:row+1,1:col+1] = torch.from_numpy(z_host)
    
    # ===============================================
    # set manning
    # ===============================================
    Manning = 0.0

    # ===============================================
    # gauge data
    # ===============================================
    gauge_index_1D = torch.tensor(paraDict['gauge_position'])
    gauge_index_1D = gauge_index_1D.to(device)

    rainfallMatrix = np.array([[0., 0.00003694444444], [6000., 0.00003694444444]])

    # ===============================================
    # set hydro field data
    # ===============================================
    numerical = Godunov(device,
                    paraDict['dx'],
                    paraDict['CFL'],
                    paraDict['Export_timeStep'],
                    0.0,
                    0,
                    secondOrder=paraDict['secondOrder'],
                    tensorType=torch.float64)
    numerical.setOutPutPath(paraDict['outputPath'])
    numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, device)
    numerical.set__frictionField_tensor(Manning, device)
    numerical.set_uniform_rainfall_time_index()
    numerical.exportField()

    # ===============================================
    # set pollutant field data
    # ===============================================
    x = x[mask > 0]
    y = y[mask > 0]
    x_host = x.numpy()
    y_host = y.numpy()
    x = torch.as_tensor(x.type(torch.float64),device=device)
    y = torch.as_tensor(y.type(torch.float64),device=device)
    
    del h_host,qx_host,qy_host,z_host
    
    # ===============================================
    # set pollution 
    # ===============================================
    ## Test1 : net deposition
    n = 3
    Mg = torch.zeros(size = (3,numerical._h_internal.shape[0]),device=device)
    Ms = torch.zeros(size = (3,numerical._h_internal.shape[0]),device=device)
    manning = torch.zeros(size = (3,numerical._h_internal.shape[0]),dtype=torch.float64, device=device)
    Mg[0,] = 1
    Mg[1,] = 4
    Mg[2,] = 5
    P = torch.zeros(size = (1,numerical._h_internal.shape[0]),dtype=torch.float64, device=device)+0.00003694444444
    p_mass = paraDict['p_mass']
    vs = np.array([0.002, 0.192, 1.78])
    PNN = ((torch.sum(Ms+Mg) / p_mass).ceil()).type(torch.int32)
    pollutant = Pollution_multi(device=device, ad0=3000, DR=2*10e-3, b=1.0, 
                            F=0.01,omega0=0.186,
                            vs=vs,rho_s=2.6e3,p_mass=p_mass)

    pollutant.init_pollutionField_tensor(Mg,Ms)
    del mask, h, qx, qy, wl, z, manning
    del Ms, Mg
    # ===============================================
    # set particle field data
    # ===============================================
    particles = Particle(PNN, p_mass, device)
    particles.init_particle_tensor()
    particles.init_particles(x , y, pollutant._Ms_cum, pollutant._Mg_cum, numerical.dx)

    dt = torch.tensor(1., dtype=torch.float64, device=device)

    

    start = time.clock()
    t = 0
    dt_washoff = 2
    dt_tol = 0
 
    while numerical.t.item() < paraDict['EndTime']:
        numerical.rungeKutta_update(rainfallMatrix, device)
        numerical.time_update_cuda(device)

        particles.transport(numerical._index, x, y, numerical.get_h(), numerical.get_qx(), numerical.get_qy(), numerical.dx, numerical.dt)
        
        if dt_tol >= dt_washoff:
            pollutant._Ms_num, pollutant._Ms = particles.update_particles_after_transport(x, pollutant._Ms_num, pollutant._Ms, pollutant._Mrs)
            pollutant.washoff_HR(P=P, h=numerical.get_h(), qx=numerical.get_qx(), qy=numerical.get_qy(), manning=manning, dx=numerical.dx, dt=dt)
            particles.update_after_washoff(pollutant._Ms_num, pollutant._Mg_num, x, y)

            # particles.update_particles_after_transport(pollutant._Ms_num, pollutant._Ms, pollutant._Mrs)
            # pollutant.washoff_HR(numerical.get_h(),numerical.get_qx(),numerical.get_qy(),dt_tol)
            # particles.update_after_washoff(pollutant._Ms_num, pollutant._Mg_num, x_host, y_host)
            dt_tol = 0.
        else:
            dt_tol += numerical.dt
        
        print(numerical.t.item())

    end = time.clock()
    print('Running time: %s Seconds'%(end-start))
    
    

if __name__ == "__main__":
    paraDict = {
        'deviceID' : 0,
        'outputPath' : '/home/tongxue/SWE_TEST/output/',
        'filePath' : '/home/tongxue/SWE_TEST/',
        'gauge_position' : [0,0],
        'dx' : 0.01,
        'CFL' : 0.5,
        'secondOrder' : True,
        'Export_timeStep' : 20.,
        'EndTime' : 20,
        'tensorsize' : [150,450],
        'p_mass' : 0.01
    }
    run(paraDict)