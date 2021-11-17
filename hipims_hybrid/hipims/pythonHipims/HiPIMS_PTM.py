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
    import postProcessing as post
    import preProcessing as pre
    from SWE_CUDA import Godunov
    from POL_CUDA_NEW import Pollution
except ImportError:
    from . import postProcessing as post
    from . import preProcessing as pre
    from .SWE_CUDA import Godunov
    from .POL_CUDA_NEW import Pollution
    
def run(paraDict):

    # ===============================================
    # set the device
    # ===============================================
    deviceID = 0
    torch.cuda.set_device(deviceID)
    device = torch.device("cuda", deviceID)

    tensorsize = paraDict['tensorsize']
    mask = torch.ones(tensorsize, dtype=torch.int32, device=device)
    mask *= 10
    mask[1, :] = 30
    mask[-2, :] = 30
    mask[:, 1] = 30
    mask[:, -2] = 30
    mask[0, :] = -9999
    mask[-1, :] = -9999
    mask[:, 0] = -9999
    mask[:, -1] = -9999

    h = torch.ones(tensorsize, dtype=torch.float64, device=device)
    h[:,int(tensorsize[1]/2):int(tensorsize[1])] = 0.5
    qx = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    qy = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    u = torch.zeros(tensorsize, dtype = torch.float64, device=device)
    u[h>=10e-6] = qx[h>=10e-6]/h[h>=10e-6]
    v = torch.zeros(tensorsize, dtype = torch.float64, device=device)
    v[h>=10e-6] = qy[h>=10e-6]/h[h>=10e-6]
    z = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    wl = z+h

    x = torch.zeros(tensorsize, device=device)
    y = torch.zeros(tensorsize, device=device)
    dx = paraDict['dx']
    for i in range(tensorsize[1]):
        x[:, i] = (i - 0.5) * dx
    for i in range(tensorsize[0]):
        y[i, :] = (i - 0.5) * dx

    # ===============================================
    # set the pollutant tensors
    # ===============================================
    p_mass = paraDict['p_mass']
    c = torch.ones(tensorsize, dtype=torch.float64, device=device) * 0.7
    c[:,int(tensorsize[1]/2):int(tensorsize[1])] = 0.5 
    Ms = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    Mg = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    Ms = c * h * dx * dx
    # ===============================================
    # set the initial pollutant
    # ===============================================   
    pollutant = Pollution(device=device,
                            ad0=3000,DR=2*10e-3, b=1.0,
                            Sf=0.00,F=0.02,omega0=0.20,
                            vs=0.01,rho_s=1.5e3) 

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

    # # ===============================================
    # # HQ_GIVEN data
    # # ===============================================
    # given_depth = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64, device=device)
    # given_discharge = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64, device=device)

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
    pollutant.init_pollutionField_tensor(p_mass=0.1, Mg=Mg, Ms=Ms, x=x, y=y, u=u, v=v, dx=dx, mask=mask, index=numerical._index) 

    numerical.set__frictionField_tensor(Manning, device)
    # numerical.set_boundary_tensor(given_depth, given_discharge)
    numerical.set_uniform_rainfall_time_index()
    numerical.exportField()

    del mask, h, qx, qy, wl, z, Manning
    # print(Ms_num)

    ims = []
    fig=plt.figure()
    axh = fig.add_subplot(3,1,1)
    axu = fig.add_subplot(3,1,2)
    axMs = fig.add_subplot(3,1,3)

    x_plt = x[0,1:-1].cpu().numpy()
    if paraDict['secondOrder']:
        while numerical.t.item() < paraDict['EndTime']:
            numerical.rungeKutta_update(rainfallMatrix, device)
            numerical.time_update_cuda(device)
            pollutant.transport(numerical.get_h(), numerical.get_qx(), numerical.get_qy(), numerical.dt)
            # pollutant.transport(numerical.get_h(), numerical.get_qx(), numerical.get_qy(), numerical.dt)

            # c = pollutant._Ms/numerical._h_internal
            # np.savetxt('/home/tongxue/dambreak_output/xp.txt', pollutant._xp.cpu().numpy())
            # np.savetxt('/home/tongxue/dambreak_output/cellid.txt', pollutant._cellid.cpu().numpy())
            # np.savetxt('/home/tongxue/dambreak_output/Ms_num.txt', pollutant._Ms_num.cpu().numpy())
            im1, = axh.plot(x_plt,numerical.get_h().cpu().numpy(),'k')
            im2, = axu.plot(x_plt,(numerical.get_qx()/numerical.get_h()).cpu().numpy(),'k')
            im3, = axMs.plot(x_plt,pollutant._Ms_num.cpu().numpy(),'k')
            ims.append([im1,im2,im3])
            
            print(numerical.t.item())
    else :
        while numerical.t.item() <  paraDict['EndTime']:
            numerical.addFlux()
            numerical.time_friction_euler_update_cuda(device)
            pollutant.transport(numerical.get_h(), numerical.get_qx(), numerical.get_qy, numerical.dt)
            print(numerical.t.item())
    axh.set_ylabel('h (m)')
    axu.set_ylabel('u (m/s)')
    axMs.set_ylabel('pollutant mass (g)')
    ani=animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=1000)
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Snow'), bitrate=1800)
    ani.save("/home/tongxue/dambreak_output/pollution.mp4",writer=writer)



if __name__ == "__main__":
    paraDict = {
        'deviceID' : 0,
        'outputPath' : '/home/tongxue/SWE_TEST/output/',
        'filePath' : '/home/tongxue/SWE_TEST/',
        'gauge_position' : [0,0],
        'dx' : 2.,
        'CFL' : 0.5,
        'secondOrder' : True,
        'Export_timeStep' : 20.,
        'EndTime' : 20,
        'tensorsize' : [3,1002],
        'p_mass' : 0.001
    }
    
    run(paraDict)