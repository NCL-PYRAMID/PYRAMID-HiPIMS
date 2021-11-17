# encoding: utf-8
"""
@author: Xue Tong
@license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
@contact: xue.tong_hhu@hotmail.com
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

import parInit
import transport
import update_after_transport
import update_after_washoff

class Pollution:
    def __init__(self, device, ad0, DR, P, b, Sf, F, omega0, vs, rho_s, p_mass, 
                    tensorType=torch.float64, tensorTypeInt=torch.int32):
        # Rainfall driven
        self.ad0 = torch.tensor([ad0], dtype=self._tensorType, device=device)
        self.DR = torch.tensor([DR], dtype=self._tensorType, device=device)
        self.P = torch.tensor([P], dtype=self._tensorType, device=device)
        self.b = torch.tensor([b], dtype=self._tensorType, device=device)
        self.h0 = torch.tensor([0.33*DR], dtype=self._tensorType, device=device)

        # Flow driven
        self.Sf = torch.tensor([Sf], dtype=self._tensorType, device=device)
        self.F = torch.tensor([F], dtype=self._tensorType, device=device)
        self.omega0 = torch.tensor([omega0], dtype=self._tensorType, device=device)
        # Deposition
        self.vs = torch.tensor([vs], dtype=self._tensorType, device=device)
        self.rho_s = torch.tensor([rho_s], dtype=self._tensorType, device=device)

        self._p_mass = torch.tensor([p_mass], dtype=self._tensorType, device=device)
        self._tensorType = tensorType
        self._device = device

    def init_pollutionField_tensor(self,Mg,Ms):
        # as_tensor：为data生成tensor
        # 如果data已经是tensor，且dtype和device与参数相同，则生成的tensor会和data共享内存。如果data是ndarray,且dtype对应，devices为cpu，则同样共享内存。其他情况则不共享内存。                                  
        self._Ms=torch.as_tensor(Ms.type(self._tensorType),
                                           device=self._device)
        self._Mg=torch.as_tensor(Mg.type(self._tensorType),
                                           device=self._device)

        self._Ms_num = torch.ceil(self._Ms / self._p_mass)
        self._Mg_num = torch.floor(self._Mg / self._p_mass)
        self._TolParticle_num = self._Mg_num + self._Ms_num
        self._Mrs = self._Ms - self._Ms_num.type(self._tensorType) * self._p_mass
        self._Mrg = self._Mg - self._Mg_num.type(self._tensorType) * self._p_mass

        del Ms,Mg
        torch.cuda.empty_cache()        

    def washoff_HR(self, h, qx, qy, dx, dt):
        rho_w = 1000.
        g = 9.81

        # Rainfall-driven detachment
        ad = torch.ones_like(h,dtype=self._tensorType,device=self._device) * self.ad0
        ad[h>self.h0]=self.ad0 * torch.pow((self.h0/h) , self.b)
        er=ad * self.P * self._Mg

        # Flow-driven detachment
        omega = rho_w * g * self.Sf * qx
        omega_e = self.F * (omega-self.omega0)
        omega_e[omega_e<0.0] = 0.0
        r = omega_e * self.rho_s/(self.rho_s-rho_w)/g/h

        # Deposition rate
        d=self.vs * self._Ms / h

        self._Ms = self._Ms + dt * (er + r - d)
        self._Mg = self._Mg + dt * (d - er - r)

        self._Ms_num = (self._Ms / self._p_mass).type(torch.int32)
        self._Mg_num = (self._Mg / self._p_mass).type(torch.int32)
        self._Mrs = self._Ms - (self._Ms_num).type(self._tensorType) * self._p_mass
        self._Mrg = self._Mg - (self._Mg_num).type(self._tensorType) * self._p_mass

    #     if dt + t >= (export_n + 1) * export_timeStep:
    #         dt = (export_n + 1) * export_timeStep - t
    #         get_c(h,dx)
    #         exportField(t)
    #         export_n += 1
    
    # def exportField(t):
    #     torch.save(
    #         self._Ms,
    #         self._outpath + "/Ms_" + str(t) + ".pt",
    #     )
    #     torch.save(
    #         self._Mg,
    #         self._outpath + "/Mg_" + str(t) + ".pt",
    #     )
    #     torch.save(
    #         self._c,
    #         self._outpath + "/c_" + str(t) + ".pt",
    #     )

    # def get_c(h,dx):
    #     self._c = torchs.zero_like(self._Ms, dtype = self._tensorType, device=self._device)
    #     self._c[h<10e-6] = 0.0
    #     self._c[h>=10e-6] = self._Ms[h>=10e-6] / dx / dx / h[h>=10e-6]

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

        self._tensorType = tensorType

    def init_particles(self, x, y, Ms_cum, Mg_cum, dx):
        parInit.initParticles(x, y, dx, Ms_cum, Mg_cum, self._cellid, self._xp, self._yp, self._layer)
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

    def update_particles_after_transport(self, Ms_num, Ms, Mrs):
        Ms_num *= 0
        cellid_transport = self._cellid[self._layer == 1].clone().detach()
        if cellid_transport.numel() != 0:
            update_after_transport.update(self._xp, cellid_transport, Ms_num)
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
        update_after_washoff.update(pid_shuffle, self._layer, self._cellid, Ms_num, Mg_num, dMs_num, dMg_num, self._xp)

        pid_unassigned = self._pid[self._layer == -1].clone().detach()

        ## this part cannot be calculated on GPU because of thread conflict
        p_num = 0
        Ms_num_diff = torch.nonzero(Ms_num - dMs_num)
        Mg_num_diff = torch.nonzero(Mg_num - dMg_num)

        for cid in Ms_num_diff:
            pid = pid_unassigned[p_num]
            self._xp[pid] = x[cid]
            self._yp[pid] = y[cid]
            self._layer[pid] = 1
            self._cellid[pid] = cid
            p_num += 1
        for cid2 in Mg_num_diff:
            pid = pid_unassigned[p_num]
            self._xp[pid] = x[cid2]
            self._yp[pid] = y[cid2]
            self._layer[pid] = 2
            self._cellid[pid] = cid2
            p_num += 1

if __name__ == "__main__":
    deviceID = 0
    torch.cuda.set_device(deviceID)
    device = torch.device("cuda", deviceID)
    pollutant = Pollution(device=device,
                            ad0=3000,DR=2*10e-3, b=1.0,
                            Sf=0.00,F=0.02,omega0=0.20,
                            vs=0.01,rho_s=1.5e3,p_mass=p_mass) 
    pollutant.init_pollutionField_tensor(Mg,Ms)
    particles = Particle(PNN, p_mass, device)
    particles.init_particle_tensor()