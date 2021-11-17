# encoding: utf-8
"""
@author: Yan Xiong, Xue Tong
@license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
@contact: xiongyan@hhu.edu.cn, xue.tong_hhu@hotmail.com
@software: hipims_torch
@file: swe.py
@time: 08.01.2020
@desc:
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import time
import math

import elementLocation
import interactGrains

class Debris:
    def __init__(self,
                 device,
                 tensorType=torch.float64,
                 tensorTypeInt=torch.int32):
        super().__init__()
        self._tensorType = tensorType
        self._tensorTypeInt = tensorTypeInt
        self._device = device

    def init_debris_tensor(self,np,layer,bar,ne,kn,nu,mu,grho,kt,gR,gx,gy,gang):
        self._np = torch.as_tensor(np, dtype=self._tensorType,device=self._device)   # 漂浮物个数
        self._layer = torch.as_tensor(layer, dtype=self._tensorType,device=self._device)    # MSM每个漂浮物有多少层
        self._bar = torch.as_tensor(bar, dtype=self._tensorType,device=self._device)  # MSM每层有多少串
        self._ne = torch.as_tensor(ne, dtype=self._tensorType,device=self._device)    # MSM每串有多少颗粒
        # Properties of grains 漂浮物物理性质 
        self._kn = torch.as_tensor(kn, dtype=self._tensorType,device=self._device)    # Normal stiffness    法向刚度系数    - 塑料/车
        self._nu = torch.as_tensor(nu, dtype=self._tensorType,device=self._device)    # Normal damping    法向粘性阻尼系数
        self._mu = torch.as_tensor(mu, dtype=self._tensorType,device=self._device)    # sliding friction 漂浮物之间的摩擦系数
        self._grho = torch.as_tensor(grho, dtype=self._tensorType,device=self._device)    # Density of the grains
        self._kt = torch.as_tensor(kt, dtype=self._tensorType,device=self._device)    # Tangential stiffness    切向刚度系数
        self._gR = torch.as_tensor(gR, dtype=self._tensorType,device=self._device)    # 单个球体颗粒半径
        self._gx = torch.as_tensor(gx, dtype=self._tensorType,device=self._device)    # 漂浮物初始位置
        self._gy = torch.as_tensor(gy, dtype=self._tensorType,device=self._device)    # 漂浮物初始位置    

        self._gz = (self._layer * self._gR).type(self._tensorType)
        self._gm =  (4.0 / 3) * self._ne * 3.14 * self._grho * self._gR * self._gR * self._gR * self._bar* self._layer # 质量
        self._gI = (1 / 12.0) * self._gm * (pow((self._bar + 1)*self._gR, 2) + pow((self._ne + 1)*self._gR, 2))   # 转动惯量
        self._gp = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)

        self._gvx = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)
        self._gvy = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)
        self._gvz = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)
        self._gax = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)
        self._gay = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)
        self._gaz = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)

        
        self._gang = torch.as_tensor(gang, dtype=self._tensorType,device=self._device)    # 初始角度
        self._gangv = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device) # 初始角速度
        self._ganga = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device) # 初始角加速度
        
        self._gt = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)

        # calculate the suitable time step for DEM 
        self._dt_gmin = torch.min(2 * 3.14 * torch.sqrt((((self._gm/(self._ne*self._bar*self._layer) )/self._kn)/10.).type(self._tensorType)))

        gNN = sum(layer * bar * ne)
        gCum = layer * bar * ne
        gCum[-1]=0
        eid = torch.arange(gNN, dtype=self._tensorTypeInt, device=self._device)
        self._ex = torch.zeros_like(eid, dtype=self._tensorType, device=self._device)
        self._ey = torch.zeros_like(eid, dtype=self._tensorType, device=self._device)
        self._eMask = torch.zeros(size=(4, eid.shape[0]), dtype=self._tensorTypeInt, device=self._device)

        self._eMask[0,:] = eid
        for i in range(np):
            for s in range(ne[i]):
                for k in range(bar[i]):
                    self._eMask[1,k+s*bar[i]+gCum[i-1]] = i    # 所属漂浮物id
                    self._eMask[2,k+s*bar[i]+gCum[i-1]] = s    # 颗数
                    self._eMask[3,k+s*bar[i]+gCum[i-1]] = k    # 串
        self._eMask = torch.flatten(self._eMask)
        
        # initiate elements' locations
        elementLocation.update(self._eMask, self._ex, self._ey, self._gx, self._gy, self._gang, self._gR, self._ne, self._bar)

        # name the value for the four walls location in DEM
        self._wleft = torch.tensor(0.0, dtype=self._tensorType, device=self._device)
        self._wdown = torch.tensor(0.0, dtype=self._tensorType, device=self._device)
        self._wright = torch.tensor(Length_x, dtype=self._tensorType, device=self._device)
        self._wup = torch.tensor(Length_y, dtype=self._tensorType, device=self._device)

    # DEM预测步
    def prediction(self,dt_g):
        self._gx += dt_g * self._gvx + 0.5 * dt_g * dt_g * self._gax
        self._gy += dt_g * self._gvy + 0.5 * dt_g * dt_g * self._gay
        self._gvx += 0.5 * dt_g * self._gax
        self._gvy += 0.5 * dt_g * self._gay
        
        self._gang += dt_g * self._gangv + 0.5 * dt_g * dt_g * self._ganga
        self._gang[torch.abs(self._gang)>360.] -= ((self._gang[torch.abs(self._gang)>360.]/360.).type(self._tensorTypeInt) * 360.).type(self._tensorType)
        self._gangv += 0.5 * dt_g * self._ganga
		
		# Zero forces 
        self._gfx = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)
        self._gfy = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)
        self._gfz = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)
        self._gt = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)
        self._gp = torch.zeros_like(self._gx, dtype=self._tensorType, device=self._device)
        
        # update elements' locations
        elementLocation.update(self._eMask, self._ex, self._ey, self._gx, self._gy, self._gang, self._gR, self._ne, self._bar)

        self._dt_g = torch.tensor(dt_g, dtype=self._tensorType, device=device)

    def interact_grains(self):
        interactGrains.update(self._mu, self._kn, self._nu, self._grho, self._kt, 
							self._dt_g, self._ne.type(self._tensorTypeInt), self._bar.type(self._tensorTypeInt), self._layer.type(self._tensorTypeInt), self._gR,
							self._gx, self._gy, self._gvx, self._gvy,
							self._gax, self._gay, self._gang, self._gangv, self._ganga,
							self._gfx, self._gfy, self._gfz, self._gp, self._gt,
							self._ex, self._ey)
        
    def interact_walls(self):
        interactWalls.update()
        
    # def loc_debris(self):
    
    # def cal_hydforce(self):
    
    # def update_acc(self):

    # def correction(self):

if __name__ == "__main__":
    deviceID = 0
    torch.cuda.set_device(deviceID)
    device = torch.device("cuda", deviceID)
    floating = Debris(device)
    floating.init_debris_tensor(np=2,layer=np.array([1,1]),bar=np.array([1,2]),ne=np.array([2,3]),mu=0.02,kn=1.,nu=[1.],grho=1.,kt=1.,gR=np.array([1.,1.]),gx=np.array([1.,2.]),gy=np.array([1.,2.]),gang=np.array([90.,90.]))
    dt_g = torch.tensor(0.1,dtype=torch.float64,device=device)
    floating.prediction(dt_g)
    floating.interact_grains()

    print('debris initiated')