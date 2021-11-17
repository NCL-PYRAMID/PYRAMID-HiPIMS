#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void update_after_transport_kernel(const int PNN, const int N, int32_t *__restrict__ pclass, int32_t *__restrict__ cellid, int32_t *__restrict__ Ms_num){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i==0){
        for (int j=0; j<PNN; j++){
            int32_t cid = cellid[j];
            int32_t n = pclass[j];
            if(cid>=0){
                Ms_num[cid + n * N] += 1 ;
            }
        }
    }
}

void update_after_transport_cuda(at::Tensor x, at::Tensor pclass, at::Tensor cellid, at::Tensor Ms_num) {

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int PNN = cellid.numel();
    const int N = x.numel();

    int thread_0 = 512;
    int block_0 = (PNN + 512 - 1) / 512;

    AT_DISPATCH_FLOATING_TYPES(
    x.type(), "update_after_transport_cuda", ([&] {
    update_after_transport_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
    PNN, N, pclass.data<int32_t>(), cellid.data<int32_t>(), Ms_num.data<int32_t>());
    }));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}