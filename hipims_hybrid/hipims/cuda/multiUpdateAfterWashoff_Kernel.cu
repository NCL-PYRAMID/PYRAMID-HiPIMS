#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void update_after_washoff_kernel(const int NN, const int N, const int PNN_assigned, 
                                                int32_t *__restrict__ pid_assigned, int32_t *__restrict__ layer,
                                                int32_t *__restrict__ pclass, int32_t *__restrict__ cellid, 
                                                int32_t *__restrict__ Ms_num, int32_t *__restrict__ Mg_num,
                                                int32_t *__restrict__ dMs_num, int32_t *__restrict__ dMg_num){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<NN){
        int32_t pid, cid, n;
        int32_t _dMs_num = dMs_num[i];
        int32_t _dMg_num = dMg_num[i];
        int32_t _Ms_num = Ms_num[i];
        int32_t _Mg_num = Mg_num[i];

        for (int j=0; j<PNN_assigned; j++){
            pid = pid_assigned[j];
            n = pclass[pid];
            cid = cellid[pid];
            cid = n * N +cid;

            if (cid == i){
                if (layer[pid]==1) {
                    if (_dMs_num < _Ms_num){
                        _dMs_num += 1;
                    }
                    else if(_dMg_num < _Mg_num){
                        layer[pid] = 2;
                        _dMg_num += 1;
                    }
                    else{
                        layer[pid] = -1;
                        cellid[pid] = -9999;
                    }
                }

                else if (layer[pid]==2){
                    if (_dMg_num < _Mg_num){
                        _dMg_num += 1;
                    }
                    else if(_dMs_num < _Ms_num){
                        layer[pid] = 1;
                        _dMs_num += 1;
                    }
                    else{
                        layer[pid] = -1;
                        cellid[pid] = -9999;
                    }
                }
            }
        }
        dMs_num[i] = _dMs_num;
        dMg_num[i] = _dMg_num;
    }
}

void update_after_washoff_cuda(at::Tensor pid_assigned, at::Tensor layer, 
                                at::Tensor pclass, at::Tensor cellid, 
                                at::Tensor Ms_num, at::Tensor Mg_num, 
                                at::Tensor dMs_num, at::Tensor dMg_num, at::Tensor x) {

    at::cuda::CUDAGuard device_guard(pid_assigned.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int PNN_assigned = pid_assigned.numel();
    const int NN = Ms_num.numel();
    const int N = x.numel();

    int thread_0 = 512;
    int block_0 = (NN + 512 - 1) / 512;

    AT_DISPATCH_FLOATING_TYPES(
    x.type(), "update_after_washoff_cuda", ([&] {
    update_after_washoff_kernel<int32_t><<<block_0, thread_0, 0, stream>>>(
        NN, N, PNN_assigned, pid_assigned.data<int32_t>(), 
        layer.data<int32_t>(), pclass.data<int32_t>(),
        cellid.data<int32_t>(), Ms_num.data<int32_t>(), Mg_num.data<int32_t>(),
        dMs_num.data<int32_t>(), dMg_num.data<int32_t>());
    }));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}