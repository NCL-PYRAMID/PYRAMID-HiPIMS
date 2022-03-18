# encoding: utf-8
"""
@author: Xue Tong
@license: (C) Copyright 2021-2025. 2025~ Apache Licence 2.0
@contact: x.tong2@lboro.ac.uk
@software: hipims_torch
@file: swe.py
@time: 08.01.2021
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

from SWE_CUDA import Godunov

import parInit
import washoff
import washoff_implicit
import update_particles_after_washoff
import assign_particles
import transport
import update_after_transport

class Pollution:
    def __init__(self, device, ad0, m0, DR, b, F, omega0, ds, rho_s,
                    tensorType=torch.float64, tensorTypeInt=torch.int32):
        # tensor setting
        self._tensorType = tensorType
        self._tensorTypeInt = tensorTypeInt
        self._device = device

        # polAttributes:
        # Rainfall driven: ad0, m0, b, h0
        # Flow driven: Sf, F, omega0
        # Deposition: rho_s, vs

        rhow = 1e3
        g = 9.81
        kv = 1.006e-6 # kinematic viscosity

        rho_s = rho_s
        ds = ds
        theta = (rho_s - rhow)/ rhow
        d_star = pow((theta * g / pow(kv,2.)), 1./3.) * ds
        vs = pow((math.sqrt(25 + 1.2 * pow(d_star,2)) - 5.), 1.5) * kv / ds
        hp = 1.5 * ds

        self._polAttributes = torch.tensor([ad0, m0, 0.33*DR, b, F, omega0, rho_s, vs, hp], dtype=tensorType, device=device)

    def init_pollutionField_tensor(self, p_mass, Mg, Ms, x, y, dx, mask, manning, index):
        # as_tensor：为data生成tensor
        # 如果data已经是tensor，且dtype和device与参数相同，则生成的tensor会和data共享内存。如果data是ndarray,且dtype对应，devices为cpu，则同样共享内存。其他情况则不共享内存。                                  
        # initiate pollutants
        self._Ms = torch.as_tensor(Ms[mask>0].type(self._tensorType), device=self._device)
        self._Mg = torch.as_tensor(Mg[mask>0].type(self._tensorType), device=self._device)

        self._x = torch.as_tensor(x[mask>0].type(self._tensorType), device=self._device)
        self._y = torch.as_tensor(y[mask>0].type(self._tensorType), device=self._device)
        self._u0 = torch.zeros_like(x[mask>0].type(self._tensorType), device=self._device)
        self._v0 = torch.zeros_like(x[mask>0].type(self._tensorType), device=self._device)
        self._dx = torch.as_tensor(dx, dtype = self._tensorType, device=self._device)
        self._manning = torch.as_tensor(manning, dtype = self._tensorType, device=self._device)
        # particle attributes
        # self._PNN = torch.tensor(PNN, dtype=self._tensorType, device=device)
        # self._p_mass = torch.as_tensor(sum(sum(Ms+Mg))/PNN, dtype=self._tensorType, device=self._device)
        self._p_mass = torch.tensor(p_mass, dtype=self._tensorType, device=self._device)
        self._PNN = torch.as_tensor(torch.ceil(sum(self._Ms+self._Mg)/self._p_mass) + torch.numel(self._x), dtype=self._tensorType, device=self._device)


        self._Ms_num = (torch.ceil(self._Ms / self._p_mass)).type(self._tensorTypeInt)
        self._Mg_num = (torch.floor(self._Mg / self._p_mass)).type(self._tensorTypeInt)
        self._TolParticle_num = self._Mg_num + self._Ms_num
        self._Mrs = self._Ms - self._Ms_num.type(self._tensorType) * self._p_mass
        self._Mrg = self._Mg - self._Mg_num.type(self._tensorType) * self._p_mass

        # initiate particles
        pid = torch.arange(self._PNN).numpy()
        pid_shuffle = torch.arange(self._PNN).numpy()   # 为了使后面的选择过程随机
        np.random.shuffle(pid_shuffle)
        self._pid = torch.as_tensor(pid,dtype=self._tensorTypeInt,device=self._device)
        self._pid_shuffle = torch.as_tensor(pid_shuffle,dtype=self._tensorTypeInt,device=self._device)
        self._cellid = torch.zeros_like(self._pid, dtype = self._tensorTypeInt, device=self._device)
        self._xp = torch.zeros_like(self._pid, dtype=self._tensorType, device=self._device)
        self._yp = torch.zeros_like(self._pid, dtype=self._tensorType, device=self._device)
        self._layer = torch.zeros_like(self._pid, dtype = self._tensorTypeInt, device=self._device)

        Ms_cum = (torch.cumsum(self._Ms_num,dim=0)).type(self._tensorTypeInt)
        Mg_cum = (torch.cumsum(self._Mg_num,dim=0)).type(self._tensorTypeInt)

        self._index = index
        
        # initiate particles' positions
        parInit.initParticles(self._x, self._y, self._dx, Ms_cum, Mg_cum, self._cellid, self._xp, self._yp, self._layer)

        del Ms,Mg,x,y
        torch.cuda.empty_cache()        

    def washoff_HR(self, P, h, qx, qy, wetMask, dt):
        dM_num = torch.zeros_like(h, dtype=self._tensorTypeInt, device=self._device)     
        hp = self._polAttributes[8]

        # Before ponding, the detached sediments are not transported but accumulate into the deposition layer.
        self._Mg[h<hp] += self._Ms[h<hp]
        self._Ms[h<hp] = 0.0
        
        # After ponding, the detached sediments can be suspended in flow
        h_pond = h[h>=hp]
        if min(h_pond.shape) != 0: 
            # time control
            t_HR = 0.
            hmin = torch.min(h_pond)
            vs = self._polAttributes[7]
            dt_HR = torch.min(hmin / vs, dt)

            pondMask = wetMask[h>=hp]
            while t_HR + dt_HR <= dt:
                washoff.update(pondMask, h, qx, qy, 
                            self._Ms, self._Mg, 
                            P, self._manning, 
                            self._polAttributes, dt_HR)
                t_HR += dt_HR            
            else:
                dt_HR = dt - t_HR
                washoff.update(pondMask, h, qx, qy, 
                            self._Ms, self._Mg, 
                            P, self._manning, 
                            self._polAttributes, dt_HR)
            
            update_particles_after_washoff.update(wetMask,
                            self._Ms, self._Mg, 
                            self._Ms_num, self._Mg_num, dM_num,
                            self._Mrs, self._Mrg,
                            self._p_mass, self._pid_shuffle, self._cellid, self._layer)

            pid_unassigned = self._pid[self._layer == -1]
            dMMask = wetMask[dM_num != 0]
            if min(dMMask.shape) != 0:
                assign_particles.update(dMMask, dM_num, pid_unassigned,
                            self._x, self._y, 
                            self._xp, self._yp,
                            self._cellid, self._layer)


    def washoff_HR_implicit(self, P, h, qx, qy, wetMask, dt):
        dM_num = torch.zeros_like(h, dtype=self._tensorTypeInt, device=self._device)     
        hp = self._polAttributes[8]

        # Before ponding, the detached sediments are not transported but accumulate into the deposition layer.
        self._Mg[h<hp] += self._Ms[h<hp]
        self._Ms[h<hp] = 0.0
        
        # After ponding, the detached sediments can be suspended in flow
        h_pond = h[h>=hp]
        if min(h_pond.shape) != 0: 
            # time control
            hmin = torch.min(h_pond)
            vs = self._polAttributes[7]
            dt_HR = torch.min(hmin / vs, dt)

            pondMask = wetMask[h>=hp]
            washoff_implicit.update(pondMask, h, qx, qy, 
                        self._Ms, self._Mg, 
                        P, self._manning, 
                        self._polAttributes, dt)          
            
            update_particles_after_washoff.update(wetMask,
                            self._Ms, self._Mg, 
                            self._Ms_num, self._Mg_num, dM_num,
                            self._Mrs, self._Mrg,
                            self._p_mass, self._pid_shuffle, self._cellid, self._layer)

            pid_unassigned = self._pid[self._layer == -1]
            dMMask = wetMask[dM_num != 0]
            if min(dMMask.shape) != 0:
                assign_particles.update(dMMask, dM_num, pid_unassigned,
                            self._x, self._y, 
                            self._xp, self._yp,
                            self._cellid, self._layer)


    def transport(self, h, qx, qy, dt):
        ## particle transporting ...
        u = torch.zeros_like(self._x, dtype = self._tensorType, device=self._device)
        u[h>=10e-6] = qx[h>=10e-6]/h[h>=10e-6]
        v = torch.zeros_like(self._x, dtype = self._tensorType, device=self._device)
        v[h>=10e-6] = qy[h>=10e-6]/h[h>=10e-6]

        p_transporting = self._pid[self._layer == 1]
        cellid_ori = self._cellid.clone().detach()
        if min(p_transporting.shape) != 0:
            transport.particle_tracking(self._index, self._x, self._y, self._u0, self._v0, u, v, 
                                 self._xp, self._yp,
                                 p_transporting, self._cellid, self._layer, self._dx, dt) 

            p_transported = self._pid[(self._cellid - cellid_ori) != 0]
            if min(p_transported.shape) != 0:
                update_after_transport.update(self._Ms, self._Ms_num, p_transported, cellid_ori, self._cellid, self._p_mass)

        self._u0 = u
        self._v0 = v

    def updateFraction(self, total_mass):
        self._fw = 1. - torch.sum(self._Mg)/ total_mass  

if __name__ == "__main__":
    deviceID = 0
    torch.cuda.set_device(deviceID)
    device = torch.device("cuda", deviceID)

    # fl
    tensorsize = (22, 22)

    mask = torch.ones(tensorsize, dtype=torch.int32, device=device)
    mask *= 10
    mask[1, :] = 50
    mask[-2, :] = 50
    mask[:, 1] = 50
    mask[:, -2] = 50
    mask[0, :] = -9999
    mask[-1, :] = -9999
    mask[:, 0] = -9999
    mask[:, -1] = -9999

    h = torch.ones(tensorsize, dtype=torch.float64, device=device)
    qx = torch.ones(tensorsize, dtype=torch.float64, device=device)
    # qx = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    qy = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    z = torch.zeros(tensorsize, dtype=torch.float64, device=device)
    wl = z+h

    dx = torch.tensor([1.], dtype=torch.float64, device=device)
    dt = torch.tensor([1.],dtype=torch.float64, device=device)
    P = torch.tensor([0.],dtype=torch.float64, device=device)
    x = torch.zeros(tensorsize, device=device)
    y = torch.zeros(tensorsize, device=device)
    for i in range(22):
        x[:, i] = i - 0.5
        y[i, :] = i - 0.5

    Ms = torch.ones(tensorsize, device=device)  
    Mg = torch.zeros(tensorsize, device=device)    

    
    manning = 0.0
    numerical = Godunov(device,
                        1.0,
                        0.5,
                        100,
                        0.0,
                        0,
                        secondOrder=False,
                        tensorType=torch.float64)
    numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, device)
    numerical.set__frictionField_tensor(manning, device)
                    
    pollutant = Pollution(device=device,
                            ad0=3000, m0=0.09, DR=2*10e-3, b=1.0,
                            F=0.02, omega0=0.20,
                            ds=70e-6, rho_s=2000)           
    pollutant.init_pollutionField_tensor(p_mass=0.1, Mg=Mg, Ms=Ms, x=x, y=y, dx=dx, mask=mask, manning=numerical._manning, index=numerical._index)
    h = numerical._h_internal
    qx = numerical._qx_internal
    qy = numerical._qy_internal
    for i in range(40):
        numerical.addFlux()
        numerical.time_update_cuda(device)
        h = numerical._h_internal
        qx = numerical._qx_internal
        qy = numerical._qy_internal
        pollutant.washoff_HR(P, h, qx, qy, numerical._wetMask, numerical.dt)
        pollutant.transport(h, qx, qy, numerical.dt)
    print('done')
