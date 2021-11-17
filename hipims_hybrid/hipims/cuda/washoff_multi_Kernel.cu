#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void washoff_kernel(const int NN, const int N, 
                        scalar_t *__restrict__ Ms, scalar_t *__restrict__ Mg,
                        int32_t *__restrict__ Ms_num,int32_t *__restrict__ Mg_num,
                        scalar_t *__restrict__ Mrs, scalar_t *__restrict__ Mrg,
                        scalar_t *__restrict__ h, scalar_t *__restrict__ qx, scalar_t *__restrict__ qy, 
                        scalar_t *__restrict__ manning, scalar_t *__restrict__ P, 
                        scalar_t *__restrict__ polAttributes, scalar_t *__restrict__ vs, 
                        scalar_t *__restrict__ ratio, 
                        scalar_t *__restrict__ dx, scalar_t *__restrict__ dt){
    scalar_t rho_w = 1000;
    scalar_t g = 9.81;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NN) {
        scalar_t _Ms = Ms[i];
        scalar_t _Mg = Mg[i];

        int j = i / N;  //jth: pollution class
        int k = i % N;  //kth: cell id

        // pollutant attributes
        // Rainfall driven
        scalar_t ad0 = polAttributes[0];
        scalar_t DR = polAttributes[1];
        scalar_t b = polAttributes[2];
        scalar_t h0 = 0.33 * DR;
        // Flow driven
        scalar_t F = polAttributes[3];
        scalar_t omega0 = polAttributes[4];
        // Deposition
        scalar_t rho_s = polAttributes[5];
        scalar_t p_mass = polAttributes[6];

        // hydrodynamic 
        scalar_t _h = h[k];
        scalar_t _qx = qx[k];
        scalar_t _qy = qy[k];
        scalar_t _P = P[k];
        scalar_t _manning = manning[k];

        // jth class pollutant attributes
        scalar_t ri = ratio[j];
        scalar_t _vs = vs[j];

        // Rainfall-driven detachment of deposite layer
        scalar_t ad = ad0;
        if (_h > h0){
            ad = ad0 * pow( (h0 / _h), b );
        }
        scalar_t er = ri * ad * _P;
        
        // Flow-driven detachment of deposite layer
        scalar_t Sf = 0.0;
        if (_h > 10e-6){
            Sf = pow( _manning, 2.) * pow( _qx, 2.) + pow( _qy, 2.) / pow( _h, (2. / 3.) );
        }
        scalar_t omega = rho_w * g * Sf * sqrt( pow( _qx, 2.) + pow( _qy, 2.) );
        scalar_t omega_e = F * (omega - omega0);
        if (omega_e < 0){
            omega_e = 0.0;
        }
        scalar_t r = ri * omega_e * rho_s / (rho_s - rho_w) / g / _h;

        // Deposition rate
        scalar_t d = _vs * _Ms / _h;

        Ms[i] = _Ms + dt[0] * (er + r - d);
        Mg[i] = _Mg + dt[0] * (d - er - r);

        Ms_num[i] = floor(Ms[i] / p_mass);
        Mg_num[i] = floor(Mg[i] / p_mass);
        Mrs[i] = Ms[i] - Ms_num[i] * p_mass;
        Mrg[i] = Mg[i] - Mg_num[i] * p_mass;
    }    
}
void washoff_cuda(at::Tensor Ms, at::Tensor Mg, 
                    at::Tensor Ms_num, at::Tensor Mg_num, 
                    at::Tensor Mrs, at::Tensor Mrg, 
                    at::Tensor h,at::Tensor qx, at::Tensor qy,
                    at::Tensor manning, at::Tensor P,
                    at::Tensor polAttributes, at::Tensor vs, 
                    at::Tensor ratio,
                    at::Tensor dx, at::Tensor dt) {
    at::cuda::CUDAGuard device_guard(h.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = h.numel();
    const int NN = Ms.numel();

    int thread_0 = 512;
    int block_0 = (NN + 512 - 1) / 512;

    AT_DISPATCH_FLOATING_TYPES(
    h.type(), "washoff_cuda", ([&] {
    washoff_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
            NN, N, Ms.data<scalar_t>(), Mg.data<scalar_t>(),
            Ms_num.data<int32_t>(), Mg_num.data<int32_t>(),
            Mrs.data<scalar_t>(), Mrg.data<scalar_t>(),
            h.data<scalar_t>(), qx.data<scalar_t>(), qy.data<scalar_t>(),
            manning.data<scalar_t>(), P.data<scalar_t>(),
            polAttributes.data<scalar_t>(), vs.data<scalar_t>(),
            ratio.data<scalar_t>(),
            dx.data<scalar_t>(), dt.data<scalar_t>());
        }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}