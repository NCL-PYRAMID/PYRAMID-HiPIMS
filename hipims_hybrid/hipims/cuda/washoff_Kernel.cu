#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void washoff_kernel(const int N, const int PNN, 
                        int32_t *__restrict__ wetMaskIndex, scalar_t *__restrict__ h, scalar_t *__restrict__ qx, scalar_t *__restrict__ qy, 
                        scalar_t *__restrict__ Ms, scalar_t *__restrict__ Mg, 
                        int32_t *__restrict__ Ms_num, int32_t *__restrict__ Mg_num, int32_t *__restrict__ dM_num,
                        scalar_t *__restrict__ Mrs, scalar_t *__restrict__ Mrg,
                        scalar_t *__restrict__ p_mass, int32_t *__restrict__ pid, int32_t *__restrict__ cellid, int32_t *__restrict__ layer,
                        scalar_t *__restrict__ P, scalar_t *__restrict__ manning,
                        scalar_t *__restrict__ polAttributes, scalar_t *__restrict__ dt){
    const scalar_t rho_w = 1000;
    const scalar_t g = 9.81;

    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (w < N) {
        int32_t i = wetMaskIndex[w]; // cell id
        scalar_t _h = h[i];

        if (_h>1e-6){
            // pollutant mass
            scalar_t _Ms = Ms[i];
            scalar_t _Mg = Mg[i];
            int32_t _Ms_num = Ms_num[i];
            int32_t _Mg_num = Mg_num[i];
            int32_t _dM_num = dM_num[i];

            // pollutant attributes
            // Rainfall driven
            scalar_t ad0 = polAttributes[0];
            scalar_t m0 = polAttributes[1];
            scalar_t h0 = polAttributes[2];
            scalar_t b = polAttributes[3];
            // Flow driven
            scalar_t F = polAttributes[4];
            scalar_t omega0 = polAttributes[5];
            // Deposition
            scalar_t rho_s = polAttributes[6];
            scalar_t _vs = polAttributes[7];

            // hydrodynamic 
            scalar_t _qx = qx[i];
            scalar_t _qy = qy[i];
            scalar_t _manning = manning[i];

            scalar_t _P = P[0];
            scalar_t _p_mass = p_mass[0];

            scalar_t dMs = 0;
            scalar_t dMg = 0;

            // Rainfall-driven detachment of deposite layer
            scalar_t ad = ad0;
            scalar_t m_star = m0;
            if (_h > h0){
                ad = ad0 * pow( (h0 / _h), b);
                m_star = m0 * pow((h0 / _h), b);
            }
            scalar_t er = (ad * _P * _Mg) / m_star;
            
            // Flow-driven detachment of deposite layer
            scalar_t Sf = pow( _manning, 2.) * pow( _qx, 2.) + pow( _qy, 2.) / pow(_h, (2. / 3.) );
            scalar_t omega = rho_w * g * Sf * sqrt( pow( _qx, 2.) + pow( _qy, 2.) );
            scalar_t omega_e = F * (omega - omega0);
            if (omega_e < 0){
                omega_e = 0.0;
            }
            scalar_t r = omega_e * rho_s / (rho_s - rho_w) / g / _h;

            // Deposition rate
            scalar_t d;
            d = _vs * _Ms / _h;    
                
            dMs = (er + r - d) * dt[0];
            dMg = (d - er - r) * dt[0];

            if (dMs > _Mg){
                dMs = _Mg;
                dMg = 0. - dMs;
            }
            if (dMg > _Ms){
                dMg = _Ms;
                dMs = 0. - dMg;
            }
                            
            _Ms += dMs;
            _Mg += dMg; 

            _Ms_num = ceil(_Ms / _p_mass);
            _Mg_num = floor(_Mg / _p_mass);
            Mrs[i] = _Ms - _Ms_num * _p_mass;
            Mrg[i] = _Mg - _Mg_num * _p_mass;

            Ms[i] = _Ms;
            Mg[i] = _Mg;
            Ms_num[i] = _Ms_num;
            Mg_num[i] = _Mg_num;

            // particle selection
            int32_t _dMs_num = 0;
            int32_t _dMg_num = 0;

            int32_t _pid;
            int32_t _cid;

            for (int j=0; j<PNN; j++){
                _pid = pid[j];
                _cid = cellid[_pid];
                if (_cid == i){
                    if (layer[_pid]==1) {
                        if (_dMs_num < _Ms_num){
                            _dMs_num += 1;
                        }
                        else if(_dMg_num < _Mg_num){
                            layer[_pid] = 2;
                            _dMg_num += 1;
                        }
                        else{
                            layer[_pid] = -1;
                            cellid[_pid] = -9999;
                        }
                    }

                    else if (layer[_pid]==2){
                        if (_dMg_num < _Mg_num){ 
                            _dMg_num += 1;
                        }
                        else if(_dMs_num < _Ms_num){
                            layer[_pid] = 1;
                            _dMs_num += 1;
                        }
                        else{
                            layer[_pid] = -1;
                            cellid[_pid] = -9999;
                        }
                    }
                }
            }

            // determing if new particles need to be assigned in this cell
            _dM_num = (_Ms_num + _Mg_num) -  (_dMs_num + _dMg_num);
            if (_dM_num > 0){
                if(_Mg_num > _dMg_num){
                    _dM_num = 0 - _dM_num;
                }
            }
            dM_num[i] = _dM_num;

        }          

    }
}


void washoff_cuda(at::Tensor wetMaskIndex, 
                    at::Tensor h, at::Tensor qx, at::Tensor qy, 
                    at::Tensor Ms, at::Tensor Mg, 
                    at::Tensor Ms_num, at::Tensor Mg_num, at::Tensor dM_num,
                    at::Tensor Mrs, at::Tensor Mrg,
                    at::Tensor p_mass, at::Tensor pid, at::Tensor cellid, at::Tensor layer, 
                    at::Tensor P, at::Tensor manning, 
                    at::Tensor polAttributes, at::Tensor dt) {

    at::cuda::CUDAGuard device_guard(h.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = wetMaskIndex.numel();
    const int PNN = pid.numel();

    int thread_0 = 512;
    int block_0 = (N + 512 - 1) / 512;

    AT_DISPATCH_FLOATING_TYPES(
        h.type(), "washoff_cuda", ([&] {
            washoff_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
                N, PNN, wetMaskIndex.data<int32_t>(), 
                    h.data<scalar_t>(), qx.data<scalar_t>(), qy.data<scalar_t>(), 
                    Ms.data<scalar_t>(), Mg.data<scalar_t>(), 
                    Ms_num.data<int32_t>(), Mg_num.data<int32_t>(), dM_num.data<int32_t>(),
                    Mrs.data<scalar_t>(), Mrg.data<scalar_t>(),                        
                    p_mass.data<scalar_t>(), pid.data<int32_t>(), cellid.data<int32_t>(), layer.data<int32_t>(), 
                    P.data<scalar_t>(), manning.data<scalar_t>(), 
                    polAttributes.data<scalar_t>(), dt.data<scalar_t>());
            }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}