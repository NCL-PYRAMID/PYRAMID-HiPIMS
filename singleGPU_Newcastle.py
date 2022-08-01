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
    CASE_PATH = os.path.join(os.environ['HOME'], '/home/lunet/cvxt2/PYRAMID-HiPIMS')
    RASTER_PATH = os.path.join(CASE_PATH, 'input')
    OUTPUT_PATH = os.path.join(CASE_PATH, 'output')
    Rainfall_data_Path = os.path.join(CASE_PATH, 'input/rain_source.txt')
    
    Manning = np.array([0.02,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.03])
    hydraulic_conductivity = 0.0
    capillary_head = 0.0
    water_content_diff = 0.0

    Degree = False
    gauges_position = np.array([])
    
    # # given water level
    # boundBox = np.array([[421601,563159,421635.8,563386.3]])
    # given_wl = np.array([[0.0,-1.0], [5.*3600.,-1.0]])
    # boundList = {
    #     'WL_GIVEN': given_wl
    # }
    # bc_type = ['WL_GIVEN']
    
    # given Q 
    Qxrange = np.linspace(1.0, 5.0, num = 1000)
    Qyrange = np.linspace(0.0, 0.0, num = 1000)
    Trange = np.linspace(3600, 6*3600, num = 1000)
    givenQ = np.array([Trange, Qxrange, Qyrange]).T
    
    given_Q1 = np.array([[0.0,1.0,0.0]]) 
    given_Q2 = np.array([[12*3600,5.0,0.0]]) 
    given_Q = np.vstack((given_Q1, givenQ, given_Q2))
     
    boundList = {
        'Q_GIVEN': given_Q
    }
    # boundBox = np.array([[421601,563159,421635.8,563386.3],[426426.88,564015.3,426432.68,564074.83]])
    # River Team: [424112.5, 561364.41, 424115.63, 561371.44]
    boundBox = np.array([[421612.74,563226.97,421620.6,563340.9]])
    bc_type = ['Q_GIVEN']
    
    # no boundary
    # boundBox = np.array([])
    # bc_type = []
    
    default_BC = 'OPEN'
    rasterPath = {
        'DEM_path': os.path.join(RASTER_PATH, 'DemRemoveSwing.tif'),
        'Landuse_path': os.path.join(RASTER_PATH, 'landuse.tif'),
        'Rainfall_path': os.path.join(RASTER_PATH, 'rainfallMask.tif'),
    }
    landLevel = 0

    paraDict = {
       'deviceID': 0,
        'dx': 2.,
        'CFL': 0.5,
        'Manning': Manning,
        'Export_timeStep': 3600,        
        't': 0.0,
        'export_n': 0,
        'secondOrder': False,
        'firstTimeStep': 0.1,
        'tensorType': torch.float64,
        'EndTime':  3600*12,        
        'Degree': Degree,
        'OUTPUT_PATH': OUTPUT_PATH,
        'rasterPath': rasterPath,
        'gauges_position': gauges_position,
        'landLevel': landLevel,
        'Rainfall_data_Path': Rainfall_data_Path,
        'hydraulic_conductivity': hydraulic_conductivity,
        'capillary_head': capillary_head,
        'water_content_diff': water_content_diff, 
        'default_BC': default_BC,
        'boundBox': boundBox,
        'bc_type': bc_type,
        'boundList':boundList
    }

    catchFlood.run(paraDict)

if __name__ == "__main__":
    main()
