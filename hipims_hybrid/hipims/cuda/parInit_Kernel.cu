// #include <torch/extension.h>
#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

// layer: 1 - surface layer; 2 - deposite layer; -1 - unassigned
template <typename scalar_t>
__global__ void parInit_kernel(const int N, const int PNN, 
                                scalar_t *x, scalar_t *y, scalar_t *dx,
                                int32_t *Ms_cum, int32_t *Mg_cum, 
                                int32_t *cellid, scalar_t *xp, scalar_t *yp, int32_t *layer){
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < N){
        int Tol_cumj1 = Ms_cum[j-1]+Mg_cum[j-1];
        int Tol_cumj2 = Ms_cum[j]+Mg_cum[j-1];
        int Tol_cum = Ms_cum[j]+Mg_cum[j];

        int Ms0 = Ms_cum[0];
        int Tol_cum0=Ms_cum[0]+Mg_cum[0];
        int Tol_cumN = Ms_cum[N-1]+Mg_cum[N-1];

        if (j==0){
            for (int i=0; i<Ms0; i++){
                cellid[i] = j;
                xp[i] = x[j] - 0.5 * dx[0] + (0.5 + i) * dx[0] / Ms0;
                yp[i] = y[j] - 0.5 * dx[0] + (0.5 + i) * dx[0] / Ms0;
                // xp[i] = x[j];
                // yp[i] = y[j];
                layer[i] = 1;   //surface layer
            }
            for (int i=Ms0; i<Tol_cum0; i++){
                cellid[i] = j;
                xp[i] = x[j] - 0.5 * dx[0] + (0.5 + i - Ms0) * dx[0] / (Tol_cum0 - Ms0);
                yp[i] = y[j] - 0.5 * dx[0] + (0.5 + i - Ms0) * dx[0] / (Tol_cum0 - Ms0);
                // xp[i] = x[j];
                // yp[i] = y[j];
                layer[i] = 2;   //deposite layer
            }
        }
        else {
            for (int i=Tol_cumj1; i<Tol_cumj2; i++){
                cellid[i] = j;
                xp[i] = x[j] - 0.5 * dx[0] + (0.5 + i - Tol_cumj1) * dx[0] / (Tol_cumj2 - Tol_cumj1);
                yp[i] = y[j] - 0.5 * dx[0] + (0.5 + i - Tol_cumj1) * dx[0] / (Tol_cumj2 - Tol_cumj1);
                // xp[i] = x[j];
                // yp[i] = y[j];
                layer[i] = 1;
            }
            for (int i=Tol_cumj2; i<Tol_cum; i++){
                cellid[i] = j;
                xp[i] = x[j] - 0.5 * dx[0] + (0.5 + i - Tol_cumj2) * dx[0] / (Tol_cum - Tol_cumj2);
                yp[i] = y[j] - 0.5 * dx[0] + (0.5 + i - Tol_cumj2) * dx[0] / (Tol_cum - Tol_cumj2);
                // xp[i] = x[j];
                // yp[i] = y[j];
                layer[i] = 2;
            }
        }

        for (int i=Tol_cumN; i<PNN; i++){
            //unassigned particles
            cellid[i] = -9999;
            xp[i] = -9999;
            yp[i] = -9999;
            layer[i] = -1;
        }
    }
}

void parInit_cuda(at::Tensor x, at::Tensor y, at::Tensor dx,
                    at::Tensor Ms_cum, at::Tensor Mg_cum,
                    at::Tensor cellid, at::Tensor xp, at::Tensor yp,
                    at::Tensor layer) {
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = x.numel();
    const int PNN = cellid.numel();

    int thread_0 = 512;
    int block_0 = (N + 512 - 1) / 512;
    
    AT_DISPATCH_FLOATING_TYPES(
        x.type(), "polInit_cuda", ([&] {
          parInit_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
              N, PNN, x.data<scalar_t>(), y.data<scalar_t>(), dx.data<scalar_t>(),
              Ms_cum.data<int32_t>(),Mg_cum.data<int32_t>(), 
              cellid.data<int32_t>(), xp.data<scalar_t>(), yp.data<scalar_t>(),
              layer.data<int32_t>());
        }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
