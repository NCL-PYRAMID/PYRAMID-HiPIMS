import torch
import math
import sys
import os
import numpy as np
import time

from hipims.pythonHipims.SWE_CUDA import Godunov
import hipims.pythonHipims.preProcessing as pre
import hipims.pythonHipims.postProcessing as post


class ParabolicSolution:
    def __init__(self, b, h0, tau, a, g, x0, y0):
        super().__init__()
        self.B = b
        self.h0 = h0
        self.tau = tau
        self.a = a
        self.g = g
        self.p = (8.0 * self.g * self.h0 / (self.a * self.a))**0.5
        self.s = 0.5 * (self.p**2. - self.tau**2.)**0.5
        # print(2*np.pi/self.s)
        self.x0 = x0
        self.y0 = y0

    def bottomElev(self, x, y):
        return self.h0 * ((x - self.x0)**2.0 +
                          (y - self.y0)**2.0) / (self.a * self.a)

    def surfaceElev(self, x, y, t):
        e = self.h0 - 0.5 * self.B * self.B * np.exp(
            -self.tau *
            t) / self.g - self.B * np.exp(-0.5 * self.tau * t) / self.g * (
                0.5 * self.tau * np.sin(self.s * t) +
                self.s * np.cos(self.s * t)) * (x - self.x0) - self.B * np.exp(
                    -0.5 * self.tau *
                    t) / self.g * (0.5 * self.tau * np.cos(self.s * t) -
                                   self.s * np.sin(self.s * t)) * (y - self.y0)
        z = self.bottomElev(x, y)

        return torch.max(e, z)

    def velocity(self, x, y, t):
        return self.B * np.exp(-0.5 * self.tau * t) * np.sin(
            self.s * t), self.B * np.exp(-0.5 * self.tau * t) * np.cos(
                self.s * t)

    def discharge(self, x, y, t):
        h = self.surfaceElev(x, y, t) - self.bottomElev(x, y)
        return self.B * np.exp(-0.5 * self.tau * t) * np.sin(
            self.s * t) * h, self.B * np.exp(-0.5 * self.tau * t) * np.cos(
                self.s * t) * h

    def Error(self, x, y, t, cell_size, h_simu, qx_simu, qy_simu):
        h = self.surfaceElev(x, y, t) - self.bottomElev(x, y)
        qx, qy = self.discharge(x, y, t)
        # we need to get the valid cells
        # h_error = (h_simu - h[internal_mask]).norm() / (h_simu.size()[0]**0.5)
        # qx_error = (qx_simu - qx[internal_mask]).norm() / (h_simu.size()[0]**
        #                                                    0.5)
        # qy_error = (qy_simu - qy[internal_mask]).norm() / (h_simu.size()[0]**
        #                                                    0.5)
        repeat_times = int(cell_size / 10.)
        h_temp = h_simu.reshape(int(8000. / cell_size), int(8000. / cell_size))
        qx_temp = qx_simu.reshape(int(8000. / cell_size),
                                  int(8000. / cell_size))
        qy_temp = qy_simu.reshape(int(8000. / cell_size),
                                  int(8000. / cell_size))

        h_temp = torch.repeat_interleave(h_temp, repeat_times, dim=0)
        h_temp = torch.repeat_interleave(h_temp, repeat_times, dim=1)
        qx_temp = torch.repeat_interleave(qx_temp, repeat_times, dim=0)
        qx_temp = torch.repeat_interleave(qx_temp, repeat_times, dim=1)
        qy_temp = torch.repeat_interleave(qy_temp, repeat_times, dim=0)
        qy_temp = torch.repeat_interleave(qy_temp, repeat_times, dim=1)

        h_error = torch.sum(abs(h - h_temp)) / 640000.0
        qx_error = torch.sum(abs(qx - qx_temp)) / 640000.0
        qy_error = torch.sum(abs(qy - qy_temp)) / 640000.0

        return h_error.item(), qx_error.item(), qy_error.item()


CASE_PATH = os.path.join(os.environ['HOME'], 'parabolicBowl', 'friction')

Degree = False

# snap = 1377.7

Tau = 0.002
B = 5.
h0 = 10.
a = 3000.

# cell_sizes = [100., 50., 25., 10.]
cell_sizes = [100., 50., 20., 10.]

schemes = [
    'jh_1st', 'xilin_1st', 'chen_1st', 'jh_2nd', 'chen_2nd',
    'chen_2nd_improved_bottom'
]
scheme = schemes[5]
secondOrder = True

topPath = os.path.join(CASE_PATH, scheme)


def run():

    # ===============================================
    # Make output folder
    # ===============================================
    if not os.path.isdir(topPath):
        os.mkdir(topPath)

    for cell_size in cell_sizes:
        secondPath = os.path.join(CASE_PATH, scheme, str(cell_size))
        if not os.path.isdir(secondPath):
            os.mkdir(secondPath)
        paraDict = {
            'deviceID': 1,
            'dx': float(cell_size),
            'CFL': 0.5,
            'Export_timeStep': 1377.679515113489 / 8.,
            't': 0.0,
            'export_n': 0,
            'secondOrder': secondOrder,
            'firstTimeStep': 0.001,
            'tensorType': torch.float64,
            'EndTime': 1377.679515113489 * 4.,
            'Degree': Degree
        }
        dt_list = []
        t_list = []
        rmse_list = []
        OUTPUT_PATH = os.path.join(CASE_PATH, scheme, str(cell_size))

        if not os.path.isdir(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)

        # ===============================================
        # set the device
        # ===============================================
        torch.cuda.set_device(paraDict['deviceID'])
        device = torch.device("cuda", paraDict['deviceID'])

        # ===============================================
        # set the tensors
        # ===============================================

        # prepare the dem

        tensorsize = (int(8000 / cell_size) + 2, int(8000 / cell_size) + 2)

        mask = torch.ones(tensorsize, dtype=torch.int32, device=device)

        rigid = 40

        mask[1, :] = rigid
        mask[-2, :] = rigid

        mask[:, 1] = rigid
        mask[:, -2] = rigid

        mask[0, :] = -9999
        mask[-1, :] = -9999
        mask[:, 0] = -9999
        mask[:, -1] = -9999

        internal_mask = mask > 0

        z = torch.zeros(tensorsize, device=device)
        qx = torch.zeros(tensorsize, device=device)
        qy = torch.zeros(tensorsize, device=device)
        wl = torch.zeros(tensorsize, device=device)

        row = torch.arange(tensorsize[0])
        col = torch.arange(tensorsize[1])
        y, x = torch.meshgrid(row, col)
        x, y = x.type(torch.DoubleTensor), y.type(torch.DoubleTensor)
        x = (x - 0.5) * cell_size
        x0 = 4000.
        y = (y - 0.5) * cell_size
        y0 = 4000.
        solution = ParabolicSolution(B, h0, Tau, a, 9.81, x0, y0)

        x = x.to(device)
        y = y.to(device)
        y = torch.flip(y, [0, 1])

        row = torch.arange(int(8000. / cell_sizes[-1]))
        col = torch.arange(int(8000. / cell_sizes[-1]))
        Y, X = torch.meshgrid(row, col)
        X, Y = X.type(torch.DoubleTensor), Y.type(torch.DoubleTensor)
        X = (X + 0.5) * cell_sizes[-1]
        Y = (Y + 0.5) * cell_sizes[-1]

        X = X.to(device)
        Y = Y.to(device)
        Y = torch.flip(Y, [0, 1])

        z = solution.bottomElev(x, y)
        wl = solution.surfaceElev(x, y, 0.)

        qx, qy = solution.velocity(x, y, 0.)

        h = torch.zeros(tensorsize, device=device)
        h = wl - z
        qx *= h
        qy *= h

        landuse = torch.zeros(tensorsize, device=device)

        Manning = [Tau]

        # ===============================================
        # rainfall data
        # ===============================================

        gauge_index_1D = torch.tensor([])

        gauge_index_1D = gauge_index_1D.to(device)

        rainfallMatrix = np.array([[0., 0.0], [6000., 0.0]])

        # ===============================================
        # set field data
        # ===============================================
        numerical = Godunov(device,
                            paraDict['dx'],
                            paraDict['CFL'],
                            paraDict['Export_timeStep'],
                            t=paraDict['t'],
                            export_n=paraDict['export_n'],
                            firstTimeStep=paraDict['firstTimeStep'],
                            secondOrder=paraDict['secondOrder'],
                            tensorType=paraDict['tensorType'])
        numerical.setOutPutPath(OUTPUT_PATH)
        numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, device)
        numerical.set__frictionField_tensor(Manning, device)
        numerical.set_landuse(mask, landuse, device)

        # ======================================================================
        numerical.set_uniform_rainfall_time_index()
        # ======================================================================
        # uniform rainfall test
        # ======================================================================
        # rainfallMatrix = np.array([[0.0, 0.0], [3600.0, 0.2 / 3600.0],
        #                            [3610.0, 0.0], [7200.0, 0.0]])
        # numerical.set_uniform_rainfall_time_index()
        # ======================================================================

        del landuse, h, qx, qy, wl, z
        torch.cuda.empty_cache()
        numerical.exportField()

        gauge_dataStoreList = []

        n = 0

        simulation_start = time.time()

        if paraDict['secondOrder']:
            while numerical.t.item() < paraDict['EndTime']:
                numerical.rungeKutta_update(rainfallMatrix, device)
                # numerical.add_uniform_PrecipitationSource(rainfallMatrix, device)
                # numerical.euler_update()
                numerical.time_update_cuda(device)
                # numerical.addFriction()
                # numerical.time_update_cuda(device)

                h_err, qx_err, qy_err = solution.Error(X, Y,
                                                       numerical.t.item(),
                                                       cell_size,
                                                       numerical.get_h(),
                                                       numerical.get_qx(),
                                                       numerical.get_qy())
                templist = []
                templist.append(numerical.t.item())
                templist += [h_err, qx_err, qy_err]
                rmse_list.append(templist)

                dt_list.append(numerical.dt.item())
                t_list.append(numerical.t.item())
                print(numerical.t.item())
                n += 1
        else:
            while numerical.t.item() < paraDict['EndTime']:
                numerical.addFlux()
                # numerical.add_uniform_PrecipitationSource(rainfallMatrix, device)
                numerical.time_friction_euler_update_cuda(device)
                # numerical.euler_update()
                # numerical.time_update_cuda(device)

                h_err, qx_err, qy_err = solution.Error(X, Y,
                                                       numerical.t.item(),
                                                       cell_size,
                                                       numerical.get_h(),
                                                       numerical.get_qx(),
                                                       numerical.get_qy())
                templist = []
                templist.append(numerical.t.item())
                templist += [h_err, qx_err, qy_err]
                rmse_list.append(templist)

                dt_list.append(numerical.dt.item())
                t_list.append(numerical.t.item())
                print(numerical.t.item())
                n += 1
        simulation_end = time.time()
        dt_list.append(simulation_end - simulation_start)
        t_list.append(simulation_end - simulation_start)
        dt_array = np.array(dt_list)
        t_array = np.array(t_list)

        rmse_array = np.array(rmse_list)

        gauge_dataStoreList = np.array(gauge_dataStoreList)
        T = np.column_stack((t_array, dt_array))
        np.savetxt(OUTPUT_PATH + '/t.txt', T)
        np.savetxt(OUTPUT_PATH + '/error.txt', rmse_array)
        np.savetxt(OUTPUT_PATH + '/gauges.txt', gauge_dataStoreList)
        print('Total runtime: ', simulation_end - simulation_start)
        # post.exportTiff(mask, mask, OUTPUT_PATH)


if __name__ == "__main__":
    run()