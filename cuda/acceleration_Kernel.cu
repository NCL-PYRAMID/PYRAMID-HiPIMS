// #include <torch/extension.h>
#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

// index of coordination 
const int i_x = 0;
const int i_y = 1;
const int i_z = 2;

template <typename scalar_t>
__global__ void interact_grains_kernel(int np, 
    scalar_t *__restrict__ g_gm_device, 
    scalar_t *__restrict__ g_gI_device,
	scalar_t *__restrict__ g_ga_device,  
    scalar_t *__restrict__ g_ganga_device,	
    scalar_t *__restrict__ g_gf_device, 
    scalar_t *__restrict__ g_gfy_device,  
    scalar_t *__restrict__ g_gt_device)
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < np){
        g_gax_device[i] = g_gfx_device[i] / g_gm_device[i];
		g_gay_device[i] = g_gfy_device[i] / g_gm_device[i];
		g_ganga_device[i] = g_gt_device[i] / g_gI_device[i];
    }
}
    