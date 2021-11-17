#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

// template <typename scalar_t>
// __global__ void update_after_transport_kernel(const int PNN, int32_t *__restrict__ cellid, int32_t *__restrict__ Ms_num){
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i==0){
//         for (int j=0; j<PNN; j++){
//             // int32_t cid_ori = cellid_ori[j];
//             int32_t cid = cellid[j];
//             // Ms_num[cid_ori] -= 1 ;
//             if(cid>=0){
//                 Ms_num[cid] += 1 ;
//             }
//         }
//     }
// }
// void update_after_transport_cuda(at::Tensor x, at::Tensor cellid, at::Tensor Ms_num) {

//     at::cuda::CUDAGuard device_guard(x.device());
//     cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//     const int PNN = cellid.numel();

//     int thread_0 = 512;
//     int block_0 = (PNN + 512 - 1) / 512;

//     AT_DISPATCH_FLOATING_TYPES(
//     x.type(), "update_after_transport_cuda", ([&] {
//     update_after_transport_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
//     PNN, cellid.data<int32_t>(), Ms_num.data<int32_t>());
//     }));
    
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//         printf("Error in load_textures: %s\n", cudaGetErrorString(err));
// }

template <typename scalar_t>
__global__ void update_after_transport_kernel(const int N, const int PNN, 
                                                scalar_t *__restrict__ Ms, scalar_t *__restrict__ Mrs, int32_t *__restrict__ Ms_num, 
                                                int32_t *__restrict__ cellid, int32_t *__restrict__ layer,scalar_t *__restrict__ p_mass){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N){
        int _Ms_num = 0;
        for (int j=0; j<PNN; j++){
            int cid = cellid[j];
            int lay = layer[j];
            if(cid==i && lay==1){
                _Ms_num += 1 ;
            }
        }
        Ms_num[i] = _Ms_num;
        Ms[i] = _Ms_num * p_mass[0] + Mrs[i];
    }
}

void update_after_transport_cuda(at::Tensor Ms, at::Tensor Mrs, at::Tensor Ms_num, at::Tensor cellid, at::Tensor layer, at::Tensor p_mass) {

    at::cuda::CUDAGuard device_guard(Ms.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = Ms.numel();
    const int PNN = cellid.numel();

    int thread_0 = 512;
    int block_0 = (N + 512 - 1) / 512;

    AT_DISPATCH_FLOATING_TYPES(
    Ms.type(), "update_after_transport_cuda", ([&] {
    update_after_transport_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
    N, PNN, Ms.data<scalar_t>(), Mrs.data<scalar_t>(), Ms_num.data<int32_t>(), cellid.data<int32_t>(), layer.data<int32_t>(), p_mass.data<scalar_t>());
    }));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}