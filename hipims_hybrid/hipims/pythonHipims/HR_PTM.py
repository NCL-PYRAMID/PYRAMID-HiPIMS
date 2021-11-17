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
    
    from POL_CUDA import Pollution
    from POL_CUDA import Particle
    
except ImportError:
    from . import postProcessing as post
    from . import preProcessing as pre
    from .SWE_CUDA import Godunov

    from .POL_CUDA import Pollution
    from .POL_CUDA import Particle

def run(paraDict, z_host, h_host, x_host, y_host, Ms_host, Mg_host):
    # ===============================================
    # Make output folder
    # ===============================================
    dt_list = []
    t_list = []
    if not os.path.isdir(paraDict['OUTPUT_PATH']):
        os.mkdir(paraDict['OUTPUT_PATH'])

    # ===============================================
    # set the device
    # ===============================================
    torch.cuda.set_device(paraDict['deviceID'])
    device = torch.device("cuda", paraDict['deviceID'])

    # ======================================================================
    # ===============================================
    # set the initial hydro information
    # ===============================================
    row = np.size(z_host,0)
    col = np.size(z_host,1)
    tensorsize = (row + 2, col + 2)

    mask = torch.ones(tensorsize, dtype=torch.int32, device=device) 
    mask *= 10
    mask[:,-2] = 30 
    mask[:,1] = 30
    mask[-2,:] = 30
    mask[1,:] = 30

    mask[0, :] = -9999
    mask[-1, :] = -9999
    mask[:, 0] = -9999
    mask[:, -1] = -9999

    z = torch.zeros(tensorsize, device=device)
    h = torch.zeros(tensorsize, device=device)
    qx = torch.zeros(tensorsize, device=device)
    qy = torch.zeros(tensorsize, device=device)
    x = torch.zeros(tensorsize, device=device)
    y = torch.zeros(tensorsize, device=device)

    z[1:row+1,1:col+1] = torch.from_numpy(z_host)
    h[1:row+1,1:col+1] = torch.from_numpy(h_host)
    x[1:row+1,1:col+1] = torch.from_numpy(x_host)
    y[1:row+1,1:col+1] = torch.from_numpy(y_host)
    wl = z + h

    # ===============================================
    # set the initial hydro information
    # ===============================================
    Ms = torch.zeros(tensorsize, device=device)
    Mg = torch.zeros(tensorsize, device=device)
    Ms[1:row+1,1:col+1] = torch.from_numpy(Ms_host)
    Mg[1:row+1,1:col+1] = torch.from_numpy(Mg_host)

    # ===============================================
    # gauge data
    # ===============================================
    gauge_index_1D = torch.tensor(paraDict['gauge_position'])
    gauge_index_1D = gauge_index_1D.to(device)

    # ===============================================
    # rainfall data
    # ===============================================
    rainfallMatrix = paraDict['Rainfall_data']

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
    numerical.setOutPutPath(paraDict['OUTPUT_PATH'])
    numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, device)
    numerical.set__frictionField_tensor(paraDict['Manning'], device)
    numerical.set_uniform_rainfall_time_index()
    numerical.exportField()

    del mask, h, qx, qy, wl, z
    torch.cuda.empty_cache()
    
    x = x[mask > 0]
    y = y[mask > 0]
    x_host = x.cpu().numpy()
    y_host = y.cpu().numpy()

    # ======================================================================
    # ===============================================
    # set pollution field data
    # ===============================================
    p_mass = paraDict['p_mass']
    pol_attributes = paraDict['pol_atrributes']
    pollutant = Pollution(device=device,
                            ad0=pol_attributes.ad0, DR=pol_attributes.DR, P=pol_attributes.P, b=pol_attributes.b,
                            Sf=pol_attributes.Sf, F=pol_attributes.F, omega0=pol_attributes.omega0,
                            vs=pol_attributes.vs, rho_s=pol_attributes.rho_s, p_mass=pol_attributes.p_mass) 
    pollutant.init_pollutionField_tensor(Mg,Ms)

    Ms_cum = (torch.cumsum(pollutant._Ms_num,dim=0)).type(torch.int32)
    Mg_cum = (torch.cumsum(pollutant._Mg_num,dim=0)).type(torch.int32)
    PNN = ((torch.sum(Ms) / p_mass).ceil()).type(torch.int32)

    # ===============================================
    # set particle field data
    # ===============================================
    particles = Particle(PNN, p_mass, device)
    particles.init_particle_tensor()
    particles.init_particles(x, y, Ms_cum, Mg_cum, numerical.dx)

    # ======================================================================
    simulation_start = time.time()
    gauge_dataStoreList = []
    dt_tol = 0
    dt_washoff = paraDict['dt_washoff']
    if gauge_index_1D.size()[0] > 0:
        n = 0
        while numerical.t.item() < paraDict['EndTime']:
            numerical.observeGauges_write(gauge_index_1D, gauge_dataStoreList,
                                          n)
            numerical.addFlux()
            numerical.addStation_PrecipitationSource(rainfallMatrix, device)
            numerical.time_friction_euler_update_cuda(device)

            particles.transport(numerical._index, x, y, numerical.get_h(), numerical.get_qx(), numerical.get_qy(), numerical.dx, numerical.dt)
            
            if dt_tol > dt_washoff:
                particles.update_particles_after_transport(pollutant._Ms_num, pollutant._Ms, pollutant._Mrs)
                pollutant.washoff_HR(numerical.get_h(),numerical.get_qx(),numerical.get_qy(),numerical.dt)
                particles.update_after_washoff(pollutant._Ms_num, pollutant._Mg_num, x_host, y_host)
            else:
                dt_tol += numerical.dt

            dt_list.append(numerical.dt.item())
            t_list.append(numerical.t.item())
            print(numerical.t.item())
            n += 1
    else:
        while numerical.t.item() < paraDict['EndTime']:
            numerical.addFlux()
            numerical.addStation_PrecipitationSource(rainfallMatrix, device)
            numerical.time_friction_euler_update_cuda(device)

            particles.transport(numerical._index, x, y, numerical.get_h(), numerical.get_qx(), numerical.get_qy(), numerical.dx, numerical.dt)
            
            if dt_tol > dt_washoff:
                particles.update_particles_after_transport(pollutant._Ms_num, pollutant._Ms, pollutant._Mrs)
                pollutant.washoff_HR(numerical.get_h(),numerical.get_qx(),numerical.get_qy(),numerical.dt)
                particles.update_after_washoff(pollutant._Ms_num, pollutant._Mg_num, x_host, y_host)
            else:
                dt_tol += numerical.dt

            dt_list.append(numerical.dt.item())
            t_list.append(numerical.t.item())
            print(numerical.t.item())

    c = torchs.zero_like(self._pid, dtype = self._tensorType, device=self._device)
    c[numerical.get_h()<10e-6] = 0.0
    c[numerical.get_h()>=h_small] = Ms[numerical.get_h()>=h_small] / dx / dx / numerical.get_h()[numerical.get_h()>=h_small]

    simulation_end = time.time()
    dt_list.append(simulation_end - simulation_start)
    t_list.append(simulation_end - simulation_start)
    dt_array = np.array(dt_list)
    t_array = np.array(t_list)
    gauge_dataStoreList = np.array(gauge_dataStoreList)

    T = np.column_stack((t_array, dt_array))
    np.savetxt(paraDict['OUTPUT_PATH'] + '/t.txt', T)
    np.savetxt(paraDict['OUTPUT_PATH'] + '/gauges.txt', gauge_dataStoreList)

    np.savetxt(paraDict['OUTPUT_PATH'] + '/Ms.txt', pollutant._Ms)
    np.savetxt(paraDict['OUTPUT_PATH'] + '/Mg.txt', pollutant._Mg)
    np.savetxt(paraDict['OUTPUT_PATH'] + '/h.txt', numerical.get_h())
    np.savetxt(paraDict['OUTPUT_PATH'] + '/c.txt', pollutant._Ms)



if __name__ == "__main__":
    CASE_PATH = os.path.join(os.environ['HOME'], 'Luanhe_case')
    RASTER_PATH = os.path.join(CASE_PATH, 'Luan_Data_90m')
    OUTPUT_PATH = os.path.join(CASE_PATH, 'output_single')

    gauges_position = np.array([])
    boundBox = np.array([])
    bc_type = np.array([])

    landLevel = 1

    paraDict = {
        'deviceID': 0,
        'dx': 90.,
        'CFL': 0.5,
        'Manning': Manning,
        'Export_timeStep': 6. * 3600.,
        't': 0.0,
        'export_n': 0,
        'secondOrder': False,
        'firstTimeStep': 1.0,
        'tensorType': torch.float64,
        'EndTime': 12. * 3600.,
        'Degree': Degree,
        'OUTPUT_PATH': OUTPUT_PATH,
        'gauges_position': gauges_position,
    }

    run(paraDict,z_host, h_host, x_host, y_host, Ms_host, Mg_host)
