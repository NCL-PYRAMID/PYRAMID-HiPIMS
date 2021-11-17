import os
import torch
import numpy as np
from hipims.pythonHipims import SurfaceFlow 

paraDict = {
        'deviceID' : 0,
        'outputPath' : '/home/tongxue/SWE_TEST/output/',
        'filePath' : '/home/tongxue/SWE_TEST/',
        'gauge_position' : [0,0],
        'dx' : 2.,
        'CFL' : 0.5,
        'secondOrder' : False,
        'Export_timeStep' : 20.,
        'EndTime' : 240,
        'tensorsize' : [1,1000],
        'h0' : 1.0,
        'h1' : 0.5
    }
SurfaceFlow.run(paraDict)