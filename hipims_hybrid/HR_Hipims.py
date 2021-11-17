import torch
import math
import sys
import os
import numpy as np
import time

from hipims.pythonHipims.SWE_CUDA import Godunov
import hipims.pythonHipims.preProcessing as pre
import hipims.pythonHipims.postProcessing as post

from POL_CUDA import Pollution
from ClassDefine import PollutionParameters

def run():

    deviceID = 0

    torch.cuda.set_device(deviceID)

    device = torch.device("cuda", deviceID)

    print(torch.cuda.current_device())

    t = torch.tensor([0.0], device=device)
    n = torch.tensor([0], device=device)
    # mask, h, qx, qy, wl, z, manning, device, gravity, dx, CFL, h_SMALL, Export_timeStep, t, export_n
    tensorsize = (800, 200)
    cell_size=1.0

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

    qx = torch.zeros(tensorsize, device=device)
    qy = torch.zeros(tensorsize, device=device)
    z = torch.zeros(tensorsize, device=device)
    h = torch.zeros(tensorsize, device=device)

#   initializing pollution
    PolParameters=PollutionParameters(c0=233.06,Dxx=1.02,Dyy=0.094,PNN=100000)  # initialize pollution parameters
    PNN = PolParameters.PNN
    pollutant=Pollution(device=device,ad0=3000,DR=2*10e-3,b=1.0,P=2.0/1000/3600,Sf=0.00,F=0.02,omega0=0.20,vs=0.01,rho_s=1.5e3,p_mass=233.06/PNN)   # initialize pollution attributes
    
    row = torch.arange(tensorsize[0])
    col = torch.arange(tensorsize[1])
    y, x = torch.meshgrid(row, col)
    x, y = x.type(torch.DoubleTensor), y.type(torch.DoubleTensor)
    x = (x - 0.5) * cell_size
    y = (y - 0.5) * cell_size

    Mg=torch.zeros(tensorsize,device=device)
    Ms=233.06/h/240./math.pi/math.sqrt(1.02*0.094)*torch.exp(-(x-60.)**2/240/1.02-(y-100.)**2/240/0.094)
    Ms = Ms.to(device)    # to(device) :将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行

    pollutant.init_pollutionField_tensor(mask,Mg,Ms,device)

    mesh.InitParticleInCell(PolParameters.p_mass)
    particles,mesh,ParticlePool=Init.ParticleDistribute(mesh,PolParameters)  # initialize particles distribution

#   initializing fluidField
    h += 1.0
    qx += 1.0*h
    wl = z+h

    numerical = Godunov(device,
                        0.1,    #dx
                        0.5,    #CFL
                        0.5,   #Export_timeStep
                        0.0,    #t
                        0,  #export_n
                        secondOrder=False,
                        tensorType=torch.float64)

    numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, device)

    h_mesh=numerical.get_h()
    qx_mesh=numerical.get_qx() 
    qy_mesh=numerical.get_qy()