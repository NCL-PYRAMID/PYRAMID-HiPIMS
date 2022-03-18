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

    P = torch.zeros(size = (1,numerical._h_internal.shape[0]),dtype=torch.float64, device=device)

    # ===============================================
    # set pollution 
    # ===============================================
    ## Test1 : net deposition
    n = 3
    Mg = torch.zeros(size = (3,numerical._h_internal.shape[0]),device=device)
    Ms = torch.zeros(size = (3,numerical._h_internal.shape[0]),device=device)
    manning = torch.zeros(size = (3,numerical._h_internal.shape[0]),dtype=torch.float64, device=device)
    for i in range(3):
        Ms[i,] = 233.06/numerical.get_h()/240./3.14/math.sqrt(1.02*0.094)*torch.exp(-(x-60.)**2.0/240.0/1.02-(y-100.)**2.0/240.0/0.094)
    
    p_mass = paraDict['p_mass']
    vs = np.array([0.005, 0.01, 0.02])
    PNN = ((torch.sum(Ms+Mg) / p_mass).ceil()).type(torch.int32)
    pollutant = Pollution_multi(device=device, ad0=3000, DR=2*10e-3, b=1.0, 
                            F=0.01,omega0=0.186,
                            vs=vs,rho_s=2.6e3,p_mass=p_mass)

    pollutant.init_pollutionField_tensor(Mg,Ms)

    # ===============================================
    # set particle field data
    # ===============================================
    particles = Particle(PNN, p_mass, device)
    particles.init_particle_tensor()
    particles.init_particles(x , y, pollutant._Ms_cum, pollutant._Mg_cum, numerical.dx)

    dt = torch.tensor(1., dtype=torch.float64, device=device)

    del mask, h, qx, qy, wl, z, Manning
    del Ms, Mg

    start = time.clock()
    t = 0
    t_host = 0
    dt_host = 1
    ims = []
    fig = plt.figure()
    while t <  paraDict['EndTime']:
        pollutant.washoff_HR(P=P, h=numerical.get_h(), qx=numerical.get_qx(), qy=numerical.get_qy(), manning=manning, dx=numerical.dx, dt=dt)
        particles.update_after_washoff(pollutant._Ms_num, pollutant._Mg_num, x, y)
        particles.transport(numerical._index, x, y, numerical.get_h(), numerical.get_qx(), numerical.get_qy(), numerical.dx, dt)
        pollutant._Ms_num, pollutant._Ms = particles.update_particles_after_transport(x, pollutant._Ms_num, pollutant._Ms, pollutant._Mrs)
        
        im = plt.scatter(x_host[100*400:101*400-1],pollutant._Ms[0,100*400:101*400-1].cpu(),color='none', marker='o', edgecolors='b', s=50)
        im2 = plt.scatter(x_host[100*400:101*400-1],pollutant._Ms[1,100*400:101*400-1].cpu(),color='none', marker='o', edgecolors='r', s=50)
        im3 = plt.scatter(x_host[100*400:101*400-1],pollutant._Ms[2,100*400:101*400-1].cpu(),color='none', marker='o', edgecolors='darkcyan', s=50)
        t_host=t_host+dt_host
        Ms_ana1 = 233.06/1/240./3.14/np.sqrt(1.02*0.094)*np.exp(-(x_host[100*400:101*400-1]-60.-t_host)**2.0/240.0/1.02-(100.5-100.)**2.0/240.0/0.094) * math.exp(-0.005*t_host)
        Ms_ana2 = 233.06/1/240./3.14/np.sqrt(1.02*0.094)*np.exp(-(x_host[100*400:101*400-1]-60.-t_host)**2.0/240.0/1.02-(100.5-100.)**2.0/240.0/0.094) * math.exp(-0.01*t_host)
        Ms_ana3 = 233.06/1/240./3.14/np.sqrt(1.02*0.094)*np.exp(-(x_host[100*400:101*400-1]-60.-t_host)**2.0/240.0/1.02-(100.5-100.)**2.0/240.0/0.094) * math.exp(-0.02*t_host)
        im4, = plt.plot(x_host[100*400:101*400-1],Ms_ana1,'k',linewidth=1.5)
        im5, = plt.plot(x_host[100*400:101*400-1],Ms_ana2,'k',linewidth=1.5)
        im6, = plt.plot(x_host[100*400:101*400-1],Ms_ana3,'k',linewidth=1.5)
        ims.append([im,im2,im3,im4,im5,im6])

        t += dt
        print(t.item())

    plt.xlabel('x(m)',weight='bold')
    plt.ylabel('pollutant mass (g)',weight='bold')
    plt.legend((im,im2,im3,im4),('fine particle','medium particle','coarse particle','analytic solution'))
    ani=animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=1000)
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Snow'), bitrate=1800)
    ani.save("/home/tongxue/multi_pollution.mp4",writer=writer)
    end = time.clock()
    print('Running time: %s Seconds'%(end-start))
    # np.savetxt('Ms.txt',pollutant._Ms.cpu().numpy())
    # np.savetxt('Mg.txt',pollutant._Mg.cpu().numpy())

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
        'EndTime' : 200,
        'tensorsize' : [200,400],
        'p_mass' : 0.0001
    }
    run(paraDict)