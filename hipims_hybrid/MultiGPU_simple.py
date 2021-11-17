from hipims.pythonHipims import MultiGPU_CatchFlood_main as MultiGPU_CatchFlood
import hipims.pythonHipims.postProcessing as post
import os
import torch
import numpy as np

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 casename**.py

    CASE_PATH = os.path.join(os.environ['HOME'], 'Luanhe_case')
    RASTER_PATH = os.path.join(CASE_PATH, 'Luan_Data_90m')
    OUTPUT_PATH = os.path.join(CASE_PATH, 'output')
    Rainfall_data_Path = os.path.join(CASE_PATH, 'rainSource.txt')
    Manning = np.array([0.035, 0.1, 0.035, 0.04, 0.15, 0.03])
    Degree = True

    # CASE_PATH = os.path.join(os.environ['HOME'], 'Eden')
    # RASTER_PATH = os.path.join(CASE_PATH, 'Tiff_Data', 'Tiff')
    # OUTPUT_PATH = os.path.join(CASE_PATH, 'output')
    # Rainfall_data_Path = os.path.join(CASE_PATH, 'Tiff_Data',
    #                                   'rainRAD_2015120300_0800.txt')
    # Manning = np.array([0.055, 0.075])
    # Degree = False

    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    else:
        pass

    rasterPath = {
        'DEM_path': os.path.join(RASTER_PATH, 'DEM.tif'),
        'Landuse_path': os.path.join(RASTER_PATH, 'Landuse.tif'),
        'Rainfall_path': os.path.join(RASTER_PATH, 'RainMask.tif')
    }

    # 1 耕地；0.035
    # 2 林地；0.1
    # 3 草地；0.035
    # 4 水域； 0.04
    # 5 建设用地；0.15
    # 6 未利用地 0.03

    paraDict = {
        'device': torch.device('cuda'),
        'dx': 90.,
        'CFL': 0.5,
        'Export_timeStep': 12. * 3600.,
        't': 0.0,
        'export_n': 0,
        'secondOrder': False,
        'firstTimeStep': 60.0,
        'tensorType': torch.float64,
        # 'EndTime': 384. * 3600.,
        'EndTime': 48. * 3600.,
        'Degree': Degree,
        'landLevel': 1,
        'rasterPath': rasterPath,
        'Rainfall_data_Path': Rainfall_data_Path,
        'OUTPUT_PATH': OUTPUT_PATH,
        'Manning': Manning
    }

    MultiGPU_CatchFlood.multiGPU_floodSimulator(MultiGPU_CatchFlood.run, paraDict, backend='nccl')    
    post.multi_exportRaster_tiff(rasterPath['DEM_path'], paraDict['OUTPUT_PATH'], 2)