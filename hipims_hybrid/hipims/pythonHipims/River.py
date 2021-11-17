import torch
import math
import sys
import os
import numpy as np
import time

try:
    import postProcessing as post
    import preProcessing as pre
    from SWE_CUDA import Godunov
except ImportError:
    from . import postProcessing as post
    from . import preProcessing as pre
    from .SWE_CUDA import Godunov
    
def run(paraDict):

    # ===============================================
    # set the device
    # ===============================================
    deviceID = 0
    torch.cuda.set_device(deviceID)
    device = torch.device("cuda", deviceID)

    z_host = paraDict['z']
    h_host = paraDict['h']

    # ===============================================
    # set the tensors
    # ===============================================
    row = np.size(z_host,0)
    col = np.size(z_host,1)
    tensorsize = (row + 2, col + 2)

    z = torch.zeros(tensorsize, device=device)
    h = torch.zeros(tensorsize, device=device)
    qx = torch.zeros(tensorsize, device=device)
    qy = torch.zeros(tensorsize, device=device)
    wl = torch.zeros(tensorsize, device=device)

    z[1:row+1,1:col+1] = torch.from_numpy(z_host)
    h[1:row+1,1:col+1] = torch.from_numpy(h_host)
    wl = z + h

    mask = paraDict['mask']
    mask[np.isnan(mask)] = -9999
    mask = torch.as_tensor(mask, dtype=torch.int32, device=device)

    Manning = paraDict['manning']

    # ===============================================
    # gauge data
    # ===============================================
    gauge_index_1D = torch.tensor(paraDict['gauge_position'])
    gauge_index_1D = gauge_index_1D.to(device)

    rainfallMatrix = np.array([[0., 0.0], [6000., 0.0]])

    # ===============================================
    # HQ_GIVEN data
    # ===============================================
    h_bound = paraDict['h_bound']
    q_bound = paraDict['q_bound']
    given_depth = torch.as_tensor(h_bound, dtype=torch.float64, device=device)
    given_discharge = torch.as_tensor(q_bound, dtype=torch.float64, device=device)

    # ===============================================
    # set field data
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
    numerical.set_boundary_tensor(given_depth, given_discharge)
    numerical.set_uniform_rainfall_time_index()
    numerical.exportField()

    # simulation_start = time.time()
    del mask, h, qx, qy, wl, z, Manning
    if paraDict['secondOrder']:
        while numerical.t.item() < paraDict['EndTime']:
            numerical.rungeKutta_update(rainfallMatrix, device)
            numerical.time_update_cuda(device)
            print(numerical.t.item())
    else :
        while numerical.t.item() <  paraDict['EndTime']:
            numerical.addFlux()
            numerical.time_friction_euler_update_cuda(device)
            print(numerical.t.item())
    h_output = numerical.get_h().cpu().numpy()
    np.savetxt('h_output.txt',h_output)
