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

rhow = 1e3
g = 9.81
kv = 1.006e-6 # kinematic viscosity

def run(paraDict, paraDictPol):
    tensorsize = paraDict['tensorsize']
    # initial condition
    h = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    qx = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    qy = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    z = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    wl = z+h

    x = torch.zeros(tensorsize, device=device)
    y = torch.zeros(tensorsize, device=device)
    for i in range(tensorsize[1]):
        x[:, i] = (i - 0.5) * dx
    for i in range(tensorsize[0]):
        y[i, :] = (i - 0.5) * dx

    Sf = paraDictPol['Sf']
    z = x * Sf    

    # ===============================================
    # set the pollutant tensors
    # ===============================================
    p_mass = paraDictPol['p_mass']
    total_mass = paraDictPol['total_mass']
    N = (tensorsize[0]-2) * (tensorsize[1]-2)
    Ms = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    Mg = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    Mg += total_mass/N

    # ===============================================
    # set the initial pollutant
    # =============================================== 
    rhos = paraDictPol['rhos']
    pd = paraDictPol['particleDiameter']
    theta = (rhos - rhow)/ rhow
    d_star = pow((theta * g / pow(kv,2.)), 1./3.) * pd
    vs = pow((math.sqrt(25 + 1.2 * pow(d_star,2)) - 5.), 1.5) * kv / pd

    pollutant = Pollution(device=device,
                            ad0=paraDictPol['ad0'], m0=paraDictPol['m0'],
                            DR=paraDictPol['DR'], b=paraDictPol['b'],
                            Sf=Sf,F=paraDictPol['F'],omega0=paraDictPol['omega0'],
                            vs=vs,rho_s=paraDictPol['rhos']) 

    # ===============================================
    # set manning
    # ===============================================
    Manning = 0.0

    # ===============================================
    # gauge data
    # ===============================================
    gauge_index_1D = torch.tensor(paraDict['gauge_position'])
    gauge_index_1D = gauge_index_1D.to(device)

    # ======================================================================
    # uniform rainfall test
    # ======================================================================
    rainfall_intensive = paraDict['rainfall_intensive']
    rainfallMatrix = np.array([[0., rainfall_intensive], [60., rainfall_intensive]])

    # ===============================================
    # set hydro field data
    # ===============================================
    mask = paraDict['mask']
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
    # numerical.set_boundary_tensor(given_depth, given_discharge)
    numerical.set_uniform_rainfall_time_index()
    numerical.exportField()
    del h, qx, qy, wl, z, Manning

    pollutant.init_pollutionField_tensor(p_mass=p_mass, Mg=Mg, Ms=Ms, x=x, y=y, dx=dx, mask=mask, manning = numerical._manning,index=numerical._index) 
    del mask, Ms, Mg, x, y

    simulate_start = time.time()
    if paraDict['secondOrder']:
        while numerical.t.item() < paraDict['EndTime']:
            numerical.addFlux()
            numerical.add_uniform_PrecipitationSource(rainfallMatrix, device)
            numerical.addFriction()
            # transporting ....
            pollutant.transport(numerical.get_h(), numerical.get_qx(), numerical.get_qy(), numerical.dt)  
            pollutant.update_after_transport()  
            # washoff and modifying the CFL condition
            pollutant.washoff_HR( numerical._rainStationData, numerical.get_h(), numerical.get_qx(), numerical.get_qy(), numerical._wetMask, numerical.dt)
            # CFL
            numerical.time_friction_euler_update_cuda(device) 
            #     pollutant.washoff_fraction()
            #     dt_tol = 0.
            # else:
            #     dt_tol += numerical.dt
            
            # fw_list.append(fw)
            # t_list.append(numerical.t.item())
            print(numerical.t.item())
    else :
        while numerical.t.item() <  paraDict['EndTime']:
            numerical.addFlux()
            numerical.time_friction_euler_update_cuda(device)
            pollutant.transport(numerical.get_h(), numerical.get_qx(), numerical.get_qy, numerical.dt)
            # pollutant.washoff_fraction()
            
            # fw_list.append(fw)
            # t_list.append(numerical.t.item())
            print(numerical.t.item())
    simulate_end = time.time()


    # def postprocess(simulate):




if __name__ == "__main__":
    ################################ common parameters #####################################
    # set the device
    deviceID = 0
    torch.cuda.set_device(deviceID)
    device = torch.device("cuda", deviceID)

    # rainfall data
    rainfall = np.array([20, 40, 65, 86, 115, 133]) /3600. /1000.
    rainDuration = np.array([40, 35, 30, 25, 25, 20]) * 60.

    # DEM data
    dx = 0.01
    tensorsize = [150+2, 200+2]
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
    
    # pollutant parameters
    b = 1.
    F = 0.02
    omega0 = 0.25
    DR = 2.

    # hydrodynamic paradict
    paraDict = {
        'deviceID' : 0,
        'outputPath' : '/home/tongxue/SWE_TEST/output/',
        'filePath' : '/home/tongxue/SWE_TEST/',
        'gauge_position' : [0,0],
        'dx' : dx,
        'CFL' : 0.5,
        'secondOrder' : True,
        'Export_timeStep' : 20.,
        'EndTime' : 200,
        'tensorsize' : tensorsize,
        'rainfall_intensive' : rainfall[0],
        'mask' : mask
    }
    
    ################################ Test Case 1 #####################################
    # Gumbeel Ct
    s_Gu = 0.072
    
    mass_Gu = 32.6  
    ad0_Gu = np.array([3800, 2300, 1600, 1600, 1500])
    m0_Gu = np.array([0.09, 0.08, 0.08, 0.06, 0.06])
    rhos = 2000
    particleDiameter = 310e-6
    # measured data
    fw_Gu = [[[0.0, 0.0], [10*60., 0.0411], [20*60, 0.0575], [30*60, 0.0629], [40*60, 0.0684]],
                            [[0.0, 0.0], [10*60, 0.112], [15*60, 0.148], [25*60, 0.230], [35*60, 0.250]],
                            [[0.0, 0.0], [10*60, 0.088], [15*60, 0.178], [20*60, 0.263], [30*60, 0.352]],
                            [[0.0, 0.0], [10*60, 0.120], [15*60, 0.134], [20*60, 0.271], [27*60, 0.339]],
                            [[0.0, 0.0], [5.*60, 0.099], [10*60, 0.292], [15*60, 0.454], [22*60, 0.542]],
                            [[0.0, 0.0], [5.*60, 0.186], [9*60, 0.4486], [13*60, 0.678], [17*60, 0.842], [20.5*60, 0.899]]]
    
    paraDictPol_Gu = {
        'p_mass' : 0.0001,
        'b' : b,
        'F' : F,
        'omega0' : omega0,
        'DR' : DR,
        'total_mass' : mass_Gu,        
        'ad0' : ad0_Gu[0],
        'm0' : m0_Gu[0],
        'rhos' : rhos,
        'Sf' : s_Gu,
        'particleDiameter' : particleDiameter
    }
    run(paraDict, paraDictPol_Gu)

    ################################ Test Case 2 #####################################
    # Lauder Ct
    # pollutant
    s_La = 0.072
    z_La = x * s_La
    mass_La = 9.3
    ad0_La = [3200, 2100, 1600, 1500, 1500]
    m0_La = [0.1, 0.12, 0.12, 0.09, 0.06]
    # measured data
    fw_La = [[[0.0, 0.0], [10, 0.173], [15, 0.209], [25, 0.258], [35, 0.303]],
                [[0.0, 0.0], [10, 0.173], [15, 0.209], [25, 0.258], [35, 0.303]],
                [[0.0, 0.0], [10, 0.273], [15, 0.328], [20, 0.399], [25, 0.438]],
                [[0.0, 0.0], [5., 0.313], [10, 0.484], [15, 0.663], [20, 0.672]],
                [[0.0, 0.0], [5., 0.357], [10, 0.529], [15, 0.677], [18.5, 0.769], [20, 0.784]]]
    paraDict_La = {
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
        'p_mass' : 0.001,
        'total_mass' : mass_La
    }
    run(paraDict_La)

    ################################ Test Case 3 #####################################
    # Piccadilly P1
    s_Pi = 0.108
    z_Pi = x * s_Pi
    ad0_Pi = [3500, 2200, 1600, 1600, 1500]
    mass_Pi = 10.6
    m0_Pi = [0.09, 0.09, 0.09, 0.08, 0.05]
    # measured data
    fw_Pi = [[[0.0, 0.0], [10, 0.090], [20, 0.128], [30, 0.140], [40, 0.149]],
                [[0.0, 0.0], [10, 0.147], [15, 0.174], [25, 0.245], [35, 0.291]],
                [[0.0, 0.0], [10, 0.248], [15, 0.288], [20, 0.330], [30, 0.382]],
                [[0.0, 0.0], [10, 0.246], [15, 0.279], [20, 0.323], [25, 0.352]],
                [[0.0, 0.0], [5, 0.253], [10, 0.381], [15, 0.486], [20, 0.551]],
                [[0.0, 0.0], [5, 0.319], [8.5, 0.459], [13, 0.721], [17, 0.818], [20, 0.849]]]
    paraDict_Pi = {
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
    run(paraDict_Pi)