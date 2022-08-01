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

CASE_PATH = os.path.join(os.environ['HOME'], 'PYRAMID-HiPIMS/NewcastleCivilCentre')
RASTER_PATH = os.path.join(CASE_PATH, 'input')
OUTPUT_PATH = os.path.join(CASE_PATH, 'output')
paraDict = {
    'DEM_path': os.path.join(RASTER_PATH, 'DEM.tif'),
    'OUTPUT_PATH': OUTPUT_PATH 
}
post.exportRaster_tiff(paraDict['DEM_path'],
                           paraDict['OUTPUT_PATH'])