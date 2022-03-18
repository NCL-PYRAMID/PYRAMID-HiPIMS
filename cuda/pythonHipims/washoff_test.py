import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from SWE_CUDA import Godunov
    from POL_CUDA import Particle
except ImportError:
    from .SWE_CUDA import Godunov
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
    qx_host = np.zeros(tensorsize)
    qy_host = np.zeros(tensorsize)
    # qy_host[0:2] = np.arange(2)+1

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
    # set Ms_num, Mg_num
    # ===============================================
    Ms_num = torch.zeros(tensorsize, device=device)
    Mg_num = torch.zeros(tensorsize, device=device)
    Ms_cum = torch.zeros(tensorsize, device=device)
    Mg_cum = torch.zeros(tensorsize, device=device)

    # ===============================================
    # set field data
    # ===============================================
    numerical = Godunov(device,
                    dx,
                    paraDict['CFL'],
                    paraDict['Export_timeStep'],
                    0.0,
                    0,
                    secondOrder=False,
                    tensorType=torch.float64)
    numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, device)

    x = torch.as_tensor(x[mask > 0].type(torch.float64),device=device)
    y = torch.as_tensor(y[mask > 0].type(torch.float64),device=device)

    x_host = x.to(device='cpu').numpy()
    y_host = y.to(device='cpu').numpy()

    dt = torch.tensor(0.1, dtype=torch.float64, device=device)
    dx = torch.tensor(1.0, dtype=torch.float64, device=device)

    # ===============================================
    # set pollutant field data
    # ===============================================
    Ms_num = torch.as_tensor(Ms_num[mask > 0].type(torch.int32),device=device)
    Ms_num[0:10] = 4
    Ms_cum = torch.as_tensor(Ms_cum[mask > 0].type(torch.int32),device=device)
    Ms_cum += 40
    Ms_cum[0:10] = (torch.arange(10) + 1) * 4
    Mg_num = torch.as_tensor(Mg_num[mask > 0].type(torch.int32),device=device)
    Mg_cum = torch.as_tensor(Mg_cum[mask > 0].type(torch.int32),device=device)

    # ===============================================
    # set particles field data
    # ===============================================
    particles = Particle(50,0.1,device)
    particles.init_particle_tensor()
    particles.init_particles(x , y, Ms_cum, Mg_cum, dx)
    print(particles._cellid)

    for i in range(4):
        Ms_num[0:10] -= 1
        Mg_num[0:10] += 1
        particles.update_after_washoff(Ms_num, Mg_num, x_host, y_host)
        print(particles._layer)
        # print(particle._cellid)
        # print(particles._yp)


if __name__ == "__main__":
    paraDict = {
        'deviceID' : 0,
        'outputPath' : '/home/tongxue/SWE_TEST/output/',
        'filePath' : '/home/tongxue/SWE_TEST/',
        'Manning' : 0.003,
        'gauge_position' : [0,0],
        'dx' : 1.0,
        'CFL' : 1.0,
        'Export_timeStep' : 1.0,
        'EndTime' : 10,
        'tensorsize' : [3,5]
    }
    
    run(paraDict)