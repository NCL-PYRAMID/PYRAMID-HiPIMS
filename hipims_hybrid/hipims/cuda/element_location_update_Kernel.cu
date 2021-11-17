// #include <torch/extension.h>
#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

const double pi = 3.14159;
/*
eMask:
 i      # element id
 N+i -> np   # 所属漂浮物id
 2*N+i -> ne  # 颗数id
 3*N+i -> bar  # 串id
*/

template <typename scalar_t>
__global__ void element_location_update_kernel(
    int N, int32_t *__restrict__ eMask, 
    scalar_t *__restrict__ ex, scalar_t *__restrict__ ey,
    scalar_t *__restrict__ gx, scalar_t *__restrict__ gy,
    scalar_t *__restrict__ gang, scalar_t *__restrict__ gR,
    scalar_t *__restrict__ ne, scalar_t *__restrict__ bar) {
  // get the index of cell

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
      int32_t i = eMask[N+j];
      int32_t s = eMask[2*N+j];
      int32_t k = eMask[3*N+j];

      ex[j] = gx[i] + (s-(ne[i]-1.)/2.)*cos(gang[i]/180.*pi)*gR[i] - (k - (bar[i] - 1.) / 2.)*sin(gang[i] / 180. * pi)*gR[i];
      ey[j] = gy[i] + (s-(ne[i]-1.)/2.)*sin(gang[i]/180.*pi)*gR[i] + (k - (bar[i] - 1.) / 2.)*cos(gang[i] / 180. * pi)*gR[i];
    }

}

void element_location_update_cuda(at::Tensor eMask, at::Tensor ex, at::Tensor ey,
                       at::Tensor gx, at::Tensor gy,
                       at::Tensor gang, at::Tensor gR, 
                       at::Tensor ne, at::Tensor bar) {
  const int N = ex.numel();
  at::cuda::CUDAGuard device_guard(ex.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      ex.type(), "element_location_update", ([&] {
        element_location_update_kernel<
            scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
            N, eMask.data<int32_t>(), ex.data<scalar_t>(), ey.data<scalar_t>(),
            gx.data<scalar_t>(), gy.data<scalar_t>(),
            gang.data<scalar_t>(), gR.data<scalar_t>(), 
            ne.data<scalar_t>(), bar.data<scalar_t>());
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
