import torch
import math
import sys
import os
import numpy as np
import time

import matplotlib.pyplot as plt

try:
    from SWE_CUDA import Godunov
    from POL_CUDA import Pollution
    from POL_CUDA import Particle
except ImportError:
    from .SWE_CUDA import Godunov
    from .POL_CUDA import Pollution
    from .POL_CUDA import Particle

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
    z_host = np.zeros(tensorsize)
    h_host = np.ones(tensorsize)
    qx_host = np.ones(tensorsize)
    qy_host = np.zeros(tensorsize)

    # ===============================================
    # set the tensors
    # ===============================================
    row = np.size(z_host,0)
    col = np.size(z_host,1)
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

    z = torch.zeros(tensorsize, device=device)
    h = torch.zeros(tensorsize, device=device)
    qx = torch.zeros(tensorsize, device=device)
    qy = torch.zeros(tensorsize, device=device)
    wl = torch.zeros(tensorsize, device=device)

    z[1:row+1,1:col+1] = torch.from_numpy(z_host)
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
    x = x - 0.5 * dx
    y = y - 0.5 * dx

    # ===============================================
    # set pollution 
    # ===============================================
    p_mass = paraDict['p_mass']
    pollutant = Pollution(device=device,
                            ad0=3000,DR=2*10e-3, b=1.0,
                            Sf=0.00,F=0.02,omega0=0.20,
                            vs=0.01,rho_s=1.5e3,p_mass=p_mass) 
    
    
    # ===============================================
    # set manning
    # ===============================================
    Manning = 0.0

    # ===============================================
    # gauge data
    # ===============================================
    gauge_index_1D = torch.tensor(paraDict['gauge_position'])
    gauge_index_1D = gauge_index_1D.to(device)

    rainfallMatrix = np.array([[0., 0.0], [6000., 0.0]])

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

    ### Test1 : net deposition
    # Mg=torch.zeros(tensorsize,device=device)
    # Mg = torch.as_tensor(Mg[mask > 0].type(torch.float64),device=device)
    # Ms=233.06/numerical.get_h()/240./3.14/math.sqrt(1.02*0.094)*torch.exp(-(x-60.)**2.0/240.0/1.02-(y-100.)**2.0/240.0/0.094)
    ### Test2 : washoff
    Ms = torch.zeros(tensorsize,device=device)
    Ms = torch.as_tensor(Ms[mask > 0].type(torch.float64),device=device)
    Mg = 233.06/numerical.get_h()/240./3.14/math.sqrt(1.02*0.094)*torch.exp(-(x-60.)**2.0/240.0/1.02-(y-100.)**2.0/240.0/0.094)
    # ###
    pollutant.init_pollutionField_tensor(Mg,Ms)

    Ms_cum = (torch.cumsum(pollutant._Ms_num,dim=0)).type(torch.int32)
    Mg_cum = (torch.cumsum(pollutant._Mg_num,dim=0)).type(torch.int32)
    PNN = ((torch.sum(Ms+Mg) / p_mass).ceil()).type(torch.int32)

    # ===============================================
    # set particle field data
    # ===============================================
    particles = Particle(PNN, p_mass, device)
    particles.init_particle_tensor()
    particles.init_particles(x , y, Ms_cum, Mg_cum, numerical.dx)

    dt = torch.tensor(1., dtype=torch.float64, device=device)

    del mask, h, qx, qy, wl, z, Manning
    del Ms, Mg

    start = time.clock()
    t = 0
    while t <  paraDict['EndTime']:
        pollutant.washoff_HR(numerical.get_h(),numerical.get_qx(),numerical.get_qy(),dx, dt)
        particles.update_after_washoff(pollutant._Ms_num, pollutant._Mg_num, x_host, y_host)
        particles.transport(numerical._index, x, y, numerical.get_h(), numerical.get_qx(), numerical.get_qy(), numerical.dx, dt)
        pollutant._Ms_num, pollutant._Ms = particles.update_particles_after_transport(pollutant._Ms_num, pollutant._Ms, pollutant._Mrs)
        t += dt
        print(t.item())

    end = time.clock()
    print('Running time: %s Seconds'%(end-start))
    np.savetxt('Ms.txt',pollutant._Ms.cpu().numpy())
    np.savetxt('Mg.txt',pollutant._Mg.cpu().numpy())

if __name__ == "__main__":
    paraDict = {
        'deviceID' : 0,
        'outputPath' : '/home/tongxue/SWE_TEST/output/',
        'filePath' : '/home/tongxue/SWE_TEST/',
        'gauge_position' : [0,0],
        'dx' : 1.,
        'CFL' : 0.5,
        'secondOrder' : True,
        'Export_timeStep' : 20.,
        'EndTime' : 400,
        'tensorsize' : [200,800],
        'p_mass' : 0.0001
    }
    run(paraDict)