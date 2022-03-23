###############################################################################
# HiPIMS application
# Xue Tong, Robin Wardle, February 2022
###############################################################################
import os, sys
import torch
import numpy as np
from pythonHipims import CatchFlood_main as catchFlood

# Add to module path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Main function
def main():
    CASE_PATH = os.path.join(os.getcwd(), 'data')
    RASTER_PATH = os.path.join(CASE_PATH, 'input')
    OUTPUT_PATH = os.path.join(CASE_PATH, 'output')
    Rainfall_data_Path = os.path.join(CASE_PATH, 'input/rain_source_2523.txt')
    
    Manning = np.array([0.02,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.03])
    hydraulic_conductivity = 0.0
    capillary_head = 0.0
    water_content_diff = 0.0

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
    default_BC = 30

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
        'Rainfall_data_Path': Rainfall_data_Path,
        'hydraulic_conductivity': hydraulic_conductivity,
        'capillary_head': capillary_head,
        'water_content_diff': water_content_diff, 
        'default_BC':default_BC
    }

    catchFlood.run(paraDict)

if __name__ == "__main__":
    main()
