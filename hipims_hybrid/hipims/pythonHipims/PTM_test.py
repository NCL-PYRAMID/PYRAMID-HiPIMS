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

    t = torch.zeros(tensorsize, device=device)

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
    x = (x - 0.5) * dx
    y = (y - 0.5) * dx

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

    t = torch.as_tensor(t[mask > 0].type(torch.float64),device=device)

    dt = torch.tensor(1.0, dtype=torch.float64, device=device)
    dx = torch.tensor(dx, dtype=torch.float64, device=device)

    # ===============================================
    # set pollutant field data
    # ===============================================
    Ms_num = torch.as_tensor(Ms_num[mask > 0].type(torch.int32),device=device)
    Ms_num[0] = 1
    Ms_cum = torch.as_tensor(Ms_cum[mask > 0].type(torch.int32),device=device)
    Ms_cum += 1
    # Ms_cum[0:10] = (torch.arange(10) + 1) * 1
    Mg_cum = torch.as_tensor(Mg_cum[mask > 0].type(torch.int32),device=device)

    # ===============================================
    # set particles field data
    # ===============================================
    particles = Particle(1,0.1,device)
    particles.init_particle_tensor()
    particles.init_particles(x , y, Ms_cum, Mg_cum, dx)

    index = numerical._index
    u = numerical.get_qx()
    v = numerical.get_qy()

    # SurfaceParticleId=[[] for i in range(Ms_num.numel())]
    # for j in range(Ms_num.numel()):
    #      if Ms_num[j] != 0 :
    #         if j == 0 :
    #             SurfaceParticleId[j] = list(range(0,Ms_cum[0]))
    #         else :
    #             SurfaceParticleId[j] = list(range(Ms_cum[j-1],Ms_cum[j]))
    

    xp = []
    yp = []
    xp.append(particles._xp.item())
    yp.append(particles._yp.item())

    while t[0]<36000:
        u = 4 * torch.sin(2*3.14/12.25/3600 * t) + 0.32 * torch.sin(2*3.14/24.5/3600 * t + 3.14/18)
        v = 3 * torch.sin(2*3.14/12.25/3600 * t + 3.14/8) + 0.12 * torch.sin(2*3.14/24.5/3600 * t + 3.14/10)
        particles.transport(index, x, y, u, v, dx, dt)
        t = t + dt
        xp.append(particles._xp.item())
        yp.append(particles._yp.item())
    # print(xp,'\t', yp)
    file = open('particle.txt','w')
    for i in range(len(xp)):
        file.write(str(xp[i])+'\t')
        file.write(str(yp[i])+'\n')
    file.close()
        # print(u)
        # print(v)
    # print(particles._layer)


if __name__ == "__main__":
    paraDict = {
        'deviceID' : 0,
        'outputPath' : '/home/tongxue/SWE_TEST/output/',
        'filePath' : '/home/tongxue/SWE_TEST/',
        'Manning' : 0.003,
        'gauge_position' : [0,0],
        'dx' : 1000,
        'CFL' : 1.0,
        'Export_timeStep' : 1.0,
        'EndTime' : 10,
        'tensorsize' : [100,100]
    }
    
    run(paraDict)