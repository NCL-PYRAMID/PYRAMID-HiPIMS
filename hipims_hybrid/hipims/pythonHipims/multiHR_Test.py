import numpy as np
import torch
from MULTI_POL_CUDA import Pollution_multi,Particle
from SWE_CUDA import Godunov

torch.cuda.set_device(0)
device = torch.device("cuda", 0)

dx = torch.as_tensor(1., dtype= torch.float64, device=device)
dt = torch.as_tensor(1., dtype= torch.float64, device=device)
P = torch.zeros((1,3), dtype= torch.float64, device=device)

z = torch.zeros((3,5), dtype= torch.float64, device=device)
h = torch.ones((3,5), dtype= torch.float64, device=device)
qx = torch.ones((3,5), dtype= torch.float64, device=device) * 1
qy = torch.zeros((3,5), dtype= torch.float64, device=device)
wl = h + z

# ===============================================
# set x, y tensor
# ===============================================
x = torch.arange(3) 
x = torch.as_tensor(x, dtype= torch.float64, device=device) + 0.5
y = torch.as_tensor([0.5, 0.5, 0.5], dtype= torch.float64, device=device)


numerical = Godunov(device, dx, 0.5, 1.0, 0, 0.0, 0.0, True, tensorType=torch.float64)
mask = torch.ones((3,5), dtype= torch.float64, device=device)
mask[0, :] = -9999
mask[-1, :] = -9999
mask[:, 0] = -9999
mask[:, -1] = -9999
numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, device)

Ms = torch.ones((3,3), dtype= torch.float64, device=device) 
Mg = torch.zeros((3,3), dtype= torch.float64, device=device)
manning = torch.ones((3,3), dtype= torch.float64, device=device) * 0.01

vs = np.array([0.0043, 0.0037, 0.02])
p_mass = 0.1
PNN = ((torch.sum(Ms+Mg) / p_mass).ceil()).type(torch.int32)
pollutant = Pollution_multi(device=device, ad0=3000, DR=2*10e-3, b=1.0, 
                            F=0.01,omega0=0.186,
                            vs=vs,rho_s=2.6e3,p_mass=p_mass)

pollutant.init_pollutionField_tensor(Mg, Ms)

index = numerical._index
particles = Particle(PNN, p_mass, device)
particles.init_particle_tensor()
particles.init_particles(x, y, pollutant._Ms_cum, pollutant._Mg_cum, dx)
# particles.transport(index, x, y, numerical.get_h(), numerical.get_qx(), numerical.get_qy(), dx, dt)
# pollutant._Ms_num, pollutant._Ms = particles.update_particles_after_transport(x, pollutant._Ms_num, pollutant._Ms, pollutant._Mrs)

t=0
# print(particles._cellid)
print(particles._layer)
while t<100:
    pollutant.washoff_HR(P=P, h=h, qx=qx, qy=qy, manning=manning, dx=dx, dt=dt)
    t += dt
particles.update_after_washoff(pollutant._Ms_num, pollutant._Mg_num, x, y)
# print(particles._cellid)
print(particles._layer)
