import os
import torch
import numpy as np
from hipims.pythonHipims import CatchFlood_main as catchFlood

if __name__ == "__main__":
    CASE_PATH = os.path.join(os.environ['HOME'], 'NewcastleCivilCentre')
    RASTER_PATH = os.path.join(CASE_PATH, 'input')
    OUTPUT_PATH = os.path.join(CASE_PATH, 'output')
    Rainfall_data_Path = os.path.join(CASE_PATH, 'input/rain_source_2523.txt')
    Manning = np.array([0.02,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.03])
    Degree = False
    gauges_position = np.array([])
    boundBox = np.array([])
    bc_type = np.array([])
    rasterPath = {
        'DEM_path': os.path.join(RASTER_PATH, 'DEM.tif'),
        'Landuse_path': os.path.join(RASTER_PATH, 'Landuse.tif'),
        'Rainfall_path': os.path.join(RASTER_PATH, 'RainMask.tif')
    }
    landLevel = 0

    paraDict = {
        'deviceID': 0,
        'dx': 2.,
        'CFL': 0.5,
        'Manning': Manning,
        'Export_timeStep': 3. * 3600.,        
        't': 0.0,
        'export_n': 0,
        'secondOrder': False,
        'firstTimeStep': 1.0,
        'tensorType': torch.float64,
        'EndTime': 24. * 3600.,        
        'Degree': Degree,
        'OUTPUT_PATH': OUTPUT_PATH,
        'rasterPath': rasterPath,
        'gauges_position': gauges_position,
        'boundBox': boundBox,
        'bc_type': bc_type,
        'landLevel': landLevel,
        'Rainfall_data_Path': Rainfall_data_Path
    }

    catchFlood.run(paraDict)