# encoding: utf-8
"""
@author: Xue Tong
@license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
@contact: xue.tong_hhu@hotmail.com
@software: hipims_torch
@file: swe.py
@time: 04.01.2021
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

import multiParInit
import washoff
import transport
import multiUpdateAfterTransport
import multiUpdateAfterWashoff

class Pollution_multi:
    def __init__(self, device, ad0, DR, b, F, omega0, vs, rho_s, p_mass, tensorType=torch.float64):
        self._polAttributes = torch.tensor([ad0, DR, b, F, omega0, rho_s, p_mass], dtype=tensorType, device=device)

        self._vs = torch.tensor(vs, dtype=tensorType, device=device)

        self._p_mass = p_mass  # n class
        self._n = vs.size
        self._tensorType = tensorType
        self._device = device

    def init_pollutionField_tensor(self, Mg, Ms):
        # as_tensor：为data生成tensor
        # 如果data已经是tensor，且dtype和device与参数相同，则生成的tensor会和data共享内存。如果data是ndarray,且dtype对应，devices为cpu，则同样共享内存。其他情况则不共享内存。
        self._Ms=torch.as_tensor(Ms.type(self._tensorType),
                                           device=self._device)
        self._Mg=torch.as_tensor(Mg.type(self._tensorType),
                                           device=self._device)

        self._Ms_num = (self._Ms / self._p_mass).type(torch.int32)
        self._Mg_num = (self._Mg / self._p_mass).type(torch.int32)
        self._TolParticle_num = (self._Mg_num + self._Ms_num).type(torch.int32)
        self._Mrs = self._Ms - self._Ms_num.type(self._tensorType) * self._p_mass
        self._Mrg = self._Mg - self._Mg_num.type(self._tensorType) * self._p_mass

        self._Ms_cum = torch.cumsum(torch.flatten(self._Ms_num),dim=0).type(torch.int32)
        self._Mg_cum = torch.cumsum(torch.flatten(self._Mg_num),dim=0).type(torch.int32)

        del Ms,Mg
        torch.cuda.empty_cache()        

    def washoff_HR(self, P, h, qx, qy, manning, dx, dt):
        rho_w = 1000.
        g = 9.81

        mi = torch.sum(self._Mg,dim=0)
        mT = torch.sum(torch.sum(self._Mg))
        ratio = torch.zeros_like(self._Ms,dtype=self._tensorType,device=self._device)
        if mT != 0:
            for i in range(self._n):
                ratio[i] = mi[i] / mT

        washoff.updating(self._Ms, self._Mg, self._Ms_num, self._Mg_num, self._Mrs, self._Mrg, 
                            h, qx, qy, manning, P, self._polAttributes, self._vs, ratio, dx, dt)

class Particle:
    def __init__(self, PNN, p_mass, device):
        self._PNN = PNN
        self._p_mass = p_mass
        self._device = device
        super().__init__()
    
    def init_particle_tensor(self, tensorType=torch.float64):
        pid = torch.arange(self._PNN, device=self._device)
        self._pid = pid.type(torch.int32)

        self._cellid = torch.zeros_like(self._pid, dtype = torch.int32, device=self._device)
        self._xp = torch.zeros_like(self._pid, dtype=tensorType, device=self._device)
        self._yp = torch.zeros_like(self._pid, dtype=tensorType, device=self._device)
        self._layer = torch.zeros_like(self._pid, dtype = torch.int32, device=self._device)
        self._class = torch.zeros_like(self._pid, dtype = torch.int32, device=self._device)

        self._tensorType = tensorType

    def init_particles(self, x, y, Ms_cum, Mg_cum, dx):
        multiParInit.initParticles(x, y, dx, Ms_cum, Mg_cum, self._cellid, self._xp, self._yp, self._layer, self._class)
        # self._xp += (torch.rand_like(self._xp) - 0.5) * dx
        # self._yp += (torch.rand_like(self._yp) - 0.5) * dx

    def transport(self, index, x, y, h, qx, qy, dx, dt):
        ## particle transporting ...
        u = torch.zeros_like(x, dtype = self._tensorType, device=self._device)
        u[h>=10e-6] = qx[h>=10e-6]/h[h>=10e-6]
        v = torch.zeros_like(x, dtype = self._tensorType, device=self._device)
        v[h>=10e-6] = qy[h>=10e-6]/h[h>=10e-6]

        transport.particle_tracking(index, x, y, u, v, 
                                 self._xp, self._yp,
                                 self._cellid, self._layer, dx, dt)

    def update_particles_after_transport(self, x, Ms_num, Ms, Mrs):
        Ms_num *= 0
        cellid_transport = self._cellid[self._layer == 1].clone().detach()
        pclass = self._class[self._layer == 1].clone().detach()
        if cellid_transport.numel() != 0:
            multiUpdateAfterTransport.update(x, pclass, cellid_transport, Ms_num)
            Ms = Ms_num.type(torch.float64) * self._p_mass + Mrs
        return Ms_num, Ms

    def update_after_washoff(self, Ms_num, Mg_num, x, y):
        # layer_ori = self._layer.clone().detach()
        # pid_assigned = self._pid[layer != -1]
        dMs_num = torch.zeros_like(Ms_num, dtype=torch.int32, device=self._device)
        dMg_num = torch.zeros_like(Mg_num, dtype=torch.int32, device=self._device)

        pid_shuffle = self._pid[self._layer != -1].clone().detach()
        pid_shuffle = pid_shuffle.cpu().numpy()
        np.random.shuffle(pid_shuffle)
        pid_shuffle = torch.as_tensor(pid_shuffle, dtype=torch.int32, device = self._device)
        multiUpdateAfterWashoff.update(pid_shuffle, self._layer, self._class, self._cellid, Ms_num, Mg_num, dMs_num, dMg_num, x)

        pid_unassigned = self._pid[self._layer == -1].clone().detach()

        ## this part cannot be calculated on GPU because of thread conflict
        p_num = 0
        Ms_num_diff = torch.nonzero(Ms_num - dMs_num)
        Mg_num_diff = torch.nonzero(Mg_num - dMg_num)
        for i in Ms_num_diff:
            pid = pid_unassigned[p_num]
            cid = i[1]
            n = i[0]
            self._xp[pid] = x[cid]
            self._yp[pid] = y[cid]
            self._layer[pid] = 1
            self._cellid[pid] = cid
            self._class[pid] = n
            p_num += 1
        for i in Mg_num_diff:
            pid = pid_unassigned[p_num]
            cid2 = i[1]
            n = i[0]
            self._xp[pid] = x[cid2]
            self._yp[pid] = y[cid2]
            self._layer[pid] = 2
            self._cellid[pid] = cid2
            p_num += 1

