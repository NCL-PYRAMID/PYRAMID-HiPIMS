from numpy.lib.npyio import savetxt
import torch
import math
import sys
import os
import numpy as np
import time

from SWE_CUDA_DEM import Godunov

cell_size = 0.001
paraDict = {'deviceID': 6,
            'dx': float(cell_size),
            'CFL': 0.5,
            'Export_timeStep': 30,
            't': 0.0,
            'export_n': 0,
            'secondOrder': True,
            'firstTimeStep': 0.005,
            'tensorType': torch.float64,
            'EndTime': 30,
        }

OUTPUT_PATH = '/home/lunet/cvxt2/HMS_JH/TurbCase'

# ===============================================
# set the device
# ===============================================
torch.cuda.set_device(paraDict['deviceID'])
device = torch.device("cuda", paraDict['deviceID'])

# ===============================================
# set the tensors
# ===============================================

# prepare the dem
lx = 0.75 * 2. + 0.75 * math.cos(math.pi/30) * 2.
ly = 0.75 * 2.
tensorsize = (int(ly/cell_size) + 2, int(lx/cell_size) + 2)

z = torch.zeros(tensorsize, device=device)
h = torch.zeros(tensorsize, device=device) + 0.1
qx = torch.zeros(tensorsize, device=device) 
qy = torch.zeros(tensorsize, device=device)
wl = torch.zeros(tensorsize, device=device)
wl = z + h

rainfall_station_Mask = torch.zeros(tensorsize, device=device)

row = torch.arange(tensorsize[0])
col = torch.arange(tensorsize[1])
y, x = torch.meshgrid(row, col)
x, y = x.type(torch.DoubleTensor), y.type(torch.DoubleTensor)
x = (x - 0.5) * cell_size
y = (y - 0.5) * cell_size
x = x.to(device)
y = y.to(device)

mask = np.loadtxt('/home/lunet/cvxt2/HMS_JH/TurbCase/mask.txt')
mask = torch.as_tensor(mask, device=device)
landuse = torch.zeros(tensorsize, device=device)
Manning = 0.001
# Manning = 0.0

# ===============================================
# rainfall data
# ===============================================

gauge_index_1D = torch.tensor([])

gauge_index_1D = gauge_index_1D.to(device)

rainfallMatrix = np.array([[0., 0.0], [200., 0.0]])

# ===============================================
# Turbulence coefficient
# ===============================================
molecule_viscousity = 1.0016e-6
pvt = 0.85

turbCoef = np.array([molecule_viscousity, pvt])

# ===============================================
# H_GIVEN & Q_GIVEN data
# ===============================================
given_depth = torch.tensor([[0.0, 0.1, 0.1], [60, 0.1, 0.1], [120, 0.1, 0.1]], dtype=torch.float64, device=device)
# given_velosity = torch.tensor([[0.0, 0.1, 0.0], [60, 0.1, 0.0]], dtype=torch.float64, device=device)
given_discharge = torch.tensor([[0.0, 0.001, 0.0, 0.0, 0.0], [60, 0.001, 0.0, 0.0, 0.0], [120, 0.001, 0.0, 0.0, 0.0]], dtype=torch.float64, device=device)

# ===============================================
# set field data
# ===============================================
numerical = Godunov(device,
                    paraDict['dx'],
                    paraDict['CFL'],
                    paraDict['Export_timeStep'],
                    t=paraDict['t'],
                    export_n=paraDict['export_n'],
                    firstTimeStep=paraDict['firstTimeStep'],
                    secondOrder=paraDict['secondOrder'],
                    tensorType=paraDict['tensorType'])

numerical.setOutPutPath(OUTPUT_PATH)
numerical.init__fluidField_tensor_Turb(mask, h, qx, qy, wl, z, turbCoef, device)
numerical.set__frictionField_tensor(Manning, device)
numerical.set_boundary_tensor(given_depth, given_discharge)
numerical.set_landuse(mask, landuse, device)

# ======================================================================
numerical.set_distributed_rainfall_station_Mask(mask, rainfall_station_Mask,
                                                    device)
# ======================================================================

del landuse, h, qx, qy, wl, z
torch.cuda.empty_cache()

simulation_start = time.time()

index_mask = torch.zeros_like(mask, dtype=torch.int32,
                                      device=device)
index_mask = torch.tensor(
    [i for i in range((mask.flatten()).size()[0])],
    dtype=torch.int32,
    device=device,
)

xx = index_mask[mask.flatten()>0]

while numerical.t.item() < paraDict['EndTime']:
    # numerical.addFlux()
    numerical.addFluxAndTurb()
    numerical.addStation_PrecipitationSource(rainfallMatrix, device)
    numerical.time_friction_euler_update_cuda(device)
    # numerical.time_friction_turb_euler_update_cuda(device)

    # print(torch.max(numerical._qx_internal), torch.argmax(numerical._qx_internal))
    # print(torch.max(numerical._qy_internal), torch.argmax(numerical._qy_internal))

    print(numerical.t.item())

simulation_end = time.time()
print('total time', simulation_end-simulation_start)

hh = torch.zeros(tensorsize, dtype=torch.float64, device=device) 
qxx = torch.zeros(tensorsize, dtype=torch.float64, device=device) 
qyy = torch.zeros(tensorsize, dtype=torch.float64, device=device) 

hh[mask>0] = numerical.get_h()
qxx[mask>0] = numerical.get_qx()
qyy[mask>0] = numerical.get_qy()

np.savetxt('/home/lunet/cvxt2/HMS_JH/TurbCase/h.txt', hh.cpu().numpy())
np.savetxt('/home/lunet/cvxt2/HMS_JH/TurbCase/qx.txt', qxx.cpu().numpy())
np.savetxt('/home/lunet/cvxt2/HMS_JH/TurbCase/qy.txt', qyy.cpu().numpy())