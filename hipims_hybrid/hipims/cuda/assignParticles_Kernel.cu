#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void assignPar_kernel(const int N,
                                int32_t *__restrict__ dMMask, int32_t *__restrict__ dM_num,
                                int32_t *__restrict__ pid_unassigned,
                                scalar_t *__restrict__ x, scalar_t *__restrict__ y,
                                scalar_t *__restrict__ xp, scalar_t *__restrict__ yp,
                                int32_t *__restrict__ cellid, int32_t *__restrict__ layer){
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (w < N) {
        int32_t cid = dMMask[w]; // cell id
        int32_t pid = pid_unassigned[w];

        xp[pid] = x[cid];
        yp[pid] = y[cid];
        cellid[pid] = cid;
        if(dM_num[w]>0){
            layer[pid] = 1;
        }
        else{
            layer[pid] = -1;
        }
    }
}  

void assignPar_cuda(at::Tensor dMMask, at::Tensor dM_num, at::Tensor pid_unassigned,
                    at::Tensor x, at::Tensor y, 
                    at::Tensor xp, at::Tensor yp,
                    at::Tensor cellid, at::Tensor layer) {

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = dMMask.numel();

    int thread_0 = 512;
    int block_0 = (N + 512 - 1) / 512;
    
    AT_DISPATCH_FLOATING_TYPES(
        x.type(), "washoff_cuda", ([&] {
            assignPar_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
                N, dMMask.data<int32_t>(), dM_num.data<int32_t>(),
                pid_unassigned.data<int32_t>(),
                x.data<scalar_t>(), y.data<scalar_t>(),
                xp.data<scalar_t>(), yp.data<scalar_t>(),
                cellid.data<int32_t>(), layer.data<int32_t>());
            }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}