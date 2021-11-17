// #include <torch/extension.h>
#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

namespace {}

template <typename scalar_t>
__global__ void stormCalculation_kernel(
    int N, int32_t *__restrict__ wetMask, scalar_t *__restrict__ x,
    scalar_t *__restrict__ y, scalar_t *__restrict__ h,
    scalar_t *__restrict__ wl, scalar_t *__restrict__ qx,
    scalar_t *__restrict__ qy, scalar_t *__restrict__ z,
    scalar_t *__restrict__ stormData, scalar_t *__restrict__ dt) {

  scalar_t h_small = 1.0e-6;
  scalar_t g = 9.81;
  //   scalar_t P_default = 101300;
  //   scalar_t p_cen = stormData[5] * 100.0;
  scalar_t p_delta = max(101300 - stormData[5] * 100.0, 100.0);
  scalar_t rho_a = 1.297;
  scalar_t r_max = stormData[6];
  scalar_t v_max = stormData[7];
  //   scalar_t lat = stormData[2] / 180.0 * 3.141592654;
  scalar_t f = 7.292e-5 * 2.0 * sin(stormData[2] / 180.0 * 3.141592654);
  scalar_t e = 2.718281828459045;
  scalar_t B =
      (v_max * v_max * e * rho_a + f * v_max * r_max * e * rho_a) / p_delta;
  scalar_t x_cen = stormData[3];
  scalar_t y_cen = stormData[4];

  //    the storm will along conter clock at north earth, and clockwise at south
  //    earth

  // get the index of cell
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    int32_t i = wetMask[j];
    // first let's calculate the gradient widn velocity

    scalar_t r = max(1.0, sqrt((x[i] - x_cen) * (x[i] - x_cen) +
                               (y[i] - y_cen) * (y[i] - y_cen)));

    scalar_t v_tan =
        sqrt(pow(r_max / r, B) * (B * p_delta * pow(e, -pow(r_max / r, B))) /
                 rho_a +
             (f * r * f * r) / 4.0) -
        (f * r) / 2.0;

    scalar_t unit_vector[2];
    unit_vector[0] = -(y[i] - y_cen) / r;
    unit_vector[1] = -(x[i] - x_cen) / r;

    scalar_t v_x = v_tan * unit_vector[0];
    scalar_t v_y = v_tan * unit_vector[1];

    scalar_t c_d = min(0.002, 0.00075 + 0.000067 * v_tan);

    scalar_t tau_x = rho_a / 1000.0 * c_d * v_x * v_tan;
    scalar_t tau_y = rho_a / 1000.0 * c_d * v_y * v_tan;

    qx[i] += tau_x;
    qy[i] += tau_y;
  }
}

// Storm_cuda(at::Tensor wetMask, at::Tensor h, at::Tensor wl,
// at::Tensor qx, at::Tensor qy, at::Tensor z,
// at::Tensor stormData, at::Tensor dt)
void Storm_cuda(at::Tensor wetMask, at::Tensor x, at::Tensor y, at::Tensor h,
                at::Tensor wl, at::Tensor qx, at::Tensor qy, at::Tensor z,
                at::Tensor stormData, at::Tensor dt) {
  const int N = wetMask.numel();
  at::cuda::CUDAGuard device_guard(h.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "stormcuda_Calculation", ([&] {
        stormCalculation_kernel<
            scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
            N, wetMask.data<int32_t>(), x.data<scalar_t>(), y.data<scalar_t>(),
            h.data<scalar_t>(), wl.data<scalar_t>(), qx.data<scalar_t>(),
            qy.data<scalar_t>(), z.data<scalar_t>(), stormData.data<scalar_t>(),
            dt.data<scalar_t>());
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
