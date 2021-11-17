import os
import torch
import numpy as np
from hipims.pythonHipims import River

z = np.loadtxt('/home/tongxue/HMS_JH/hipims_hybrid/z.txt')
h = np.loadtxt('/home/tongxue/HMS_JH/hipims_hybrid/h.txt')
h_bound = np.loadtxt('/home/tongxue/HMS_JH/hipims_hybrid/h_bound.txt')
Q_bound = np.loadtxt('/home/tongxue/HMS_JH/hipims_hybrid/q_bound.txt')
mask = np.loadtxt('/home/tongxue/HMS_JH/hipims_hybrid/mask.txt')
manning = 0

paraDict = {
        'deviceID' : 0,
        'outputPath' : '/home/tongxue/SWE_TEST/output/',
        'filePath' : '/home/tongxue/SWE_TEST/',
        'gauge_position' : [0,0],
        'dx' : 10,
        'CFL' : 0.5,
        'secondOrder' : False,
        'Export_timeStep' : 20.,
        'EndTime' : 10,
        'z' : z,
        'h' : h,
        'h_bound' : h_bound,
        'q_bound' : Q_bound,
        'mask' : mask,
        'manning' : manning,
    }

River.run(paraDict)