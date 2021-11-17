import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import transport

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
    qx_host[0:5] = np.arange(5)+1
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

    dt = torch.tensor(0.1, dtype=torch.float64, device=device)
    dx = torch.tensor(1.0, dtype=torch.float64, device=device)

    # ===============================================
    # set pollutant field data
    # ===============================================
    Ms_num = torch.as_tensor(Ms_num[mask > 0].type(torch.int32),device=device)
    Ms_num[0:5] = 4
    Ms_cum = torch.as_tensor(Ms_cum[mask > 0].type(torch.int32),device=device)
    Ms_cum += 20
    Ms_cum[0:5] = (torch.arange(5) + 1) * 4
    Mg_cum = torch.as_tensor(Mg_cum[mask > 0].type(torch.int32),device=device)

    Mrs = torch.zeros_like(Ms_num.type(torch.float64),device=device)
    Ms = Ms_num.type(torch.float64) * 0.1 + Mrs

    # ===============================================
    # set particles field data
    # ===============================================
    particles = Particle(20,0.1,device)
    particles.init_particle_tensor()
    particles.init_particles(x , y, Ms_cum, Mg_cum, dx)

    index = numerical._index
    u = numerical.get_qx()
    v = numerical.get_qy()

    SurfaceParticleId=[[] for i in range(Ms_num.numel())]
    for j in range(Ms_num.numel()):
         if Ms_num[j] != 0 :
            if j == 0 :
                SurfaceParticleId[j] = list(range(0,Ms_cum[0]))
            else :
                SurfaceParticleId[j] = list(range(Ms_cum[j-1],Ms_cum[j]))
    
    # print(index)
    t=0
    # print(particles._xp)
    # print(particles._yp)
    print(particles._cellid)
    # print(Ms_num)
    while t<4.0:
        particles.transport(index, x, y, numerical.get_h(), numerical.get_qx(), numerical.get_qy(), dx, dt)
        particles.update_particles_after_transport(Ms_num, Ms, Mrs)
        # print(particles._xp)
        # print(particles._yp)
        # print(Ms_num)
        Ms = Ms_num.type(torch.float64) * 0.1 + Mrs
        print(Ms)
        # print(particles._layer)
        t=t+0.1
        # print(Ms_num)
    # print(particles._layer)


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