// #include <torch/extension.h>
#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void
station_PrecipitationCalculation_kernel(int N, scalar_t *__restrict__ h_update,
                                        int16_t *__restrict__ rainStationMask,
                                        scalar_t *__restrict__ rainStationData,
                                        scalar_t *__restrict__ dt) {
  // get the index of cell
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    h_update[i] += rainStationData[rainStationMask[i]] * dt[0];
  }
}

void station_PrecipitationCalculation_cuda(at::Tensor h_update,
                                           at::Tensor rainStationMask,
                                           at::Tensor rainStationData,
                                           at::Tensor dt) {
  const int N = rainStationMask.numel();
  at::cuda::CUDAGuard device_guard(rainStationMask.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      h_update.type(), "station_PrecipitationCalculation", ([&] {
        station_PrecipitationCalculation_kernel<
            scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
            N, h_update.data<scalar_t>(), rainStationMask.data<int16_t>(),
            rainStationData.data<scalar_t>(), dt.data<scalar_t>());
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
