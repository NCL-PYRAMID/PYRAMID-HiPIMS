#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

const double pi = 3.14159;

template <typename scalar_t>
__global__ void interact_walls_kernel(const int np, scalar_t *__restrict__ wleft, scalar_t *__restrict__ wright, scalar_t *__restrict__ wup, scalar_t *__restrict__ wdown,
									scalar_t *__restrict__ wp_up, scalar_t *__restrict__ wp_down, scalar_t *__restrict__ wp_left, scalar_t *__restrict__ wp_right, 
									scalar_t *__restrict__ mu, scalar_t *__restrict__ g_gkn_device, scalar_t *__restrict__ g_gnu_device, scalar_t *__restrict__ g_grho_device, scalar_t *__restrict__ g_gkt_device,
									scalar_t *__restrict__ dt_g, int32_t *__restrict__ g_gne_device, int32_t *__restrict__ g_gbar_device, int32_t *__restrict__ g_glay_device, scalar_t *__restrict__ g_gR_device,
									scalar_t *__restrict__ g_gx_device, scalar_t *__restrict__ g_gy_device, scalar_t *__restrict__ g_gvx_device, scalar_t *__restrict__ g_gvy_device,
									scalar_t *__restrict__ g_gax_device, scalar_t *__restrict__ g_gay_device, scalar_t *__restrict__ g_gang_device, scalar_t *__restrict__ g_gangv_device, scalar_t *__restrict__ g_ganga_device,
									scalar_t *__restrict__ g_gfx_device, scalar_t *__restrict__ g_gfy_device, scalar_t *__restrict__ g_gfz_device, scalar_t *__restrict__ g_gp_device, scalar_t *__restrict__ g_gt_device,
									scalar_t *__restrict__ neloc_x_device, scalar_t *__restrict__ neloc_y_device, scalar_t *__restrict__ t){
	scalar_t wall_u = 0.0;
	scalar_t wall_d = 0.0;
	scalar_t wall_r = 0.0;
	scalar_t wall_l = 0.0;

	int  i = blockIdx.x * blockDim.x + threadIdx.x;
	
	while (i<np) {
		wall_u = compute_force_upper_wall(i, 0, 0, 0, wup[0], mu[0], g_gkn_device, g_gnu_device, g_grho_device, g_gkt_device, np, dt_g[0], g_gne_device, g_gbar_device, g_glay_device, g_gR_device,
			g_gx_device, g_gy_device, g_gvx_device, g_gvy_device, g_gax_device, g_gay_device, g_gang_device, g_gangv_device, g_ganga_device,
			g_gfx_device, g_gfy_device, g_gfz_device, g_gp_device, g_gt_device, neloc_x_device, neloc_y_device);
		wall_d = compute_force_lower_wall(i, 0, 0, 0, wdown[0], mu[0], g_gkn_device, g_gnu_device, g_grho_device, g_gkt_device, np, dt_g[0], g_gne_device, g_gbar_device, g_glay_device, g_gR_device,
			g_gx_device, g_gy_device, g_gvx_device, g_gvy_device, g_gax_device, g_gay_device, g_gang_device, g_gangv_device, g_ganga_device,
			g_gfx_device, g_gfy_device, g_gfz_device, g_gp_device, g_gt_device, neloc_x_device, neloc_y_device);
		wall_r = compute_force_right_wall(i, 0, 0, 0, wright[0], mu[0], g_gkn_device, g_gnu_device, g_grho_device, g_gkt_device, np, dt_g[0], g_gne_device, g_gbar_device, g_glay_device, g_gR_device,
			g_gx_device, g_gy_device, g_gvx_device, g_gvy_device, g_gax_device, g_gay_device, g_gang_device, g_gangv_device, g_ganga_device,
			g_gfx_device, g_gfy_device, g_gfz_device, g_gp_device, g_gt_device, neloc_x_device, neloc_y_device);
		wall_l = compute_force_left_wall(i, 0, 0, 0, wleft[0], mu[0], g_gkn_device, g_gnu_device, g_grho_device, g_gkt_device, np, dt_g[0], g_gne_device, g_gbar_device, g_glay_device, g_gR_device,
			g_gx_device, g_gy_device, g_gvx_device, g_gvy_device, g_gax_device, g_gay_device, g_gang_device, g_gangv_device, g_ganga_device,
			g_gfx_device, g_gfy_device, g_gfz_device, g_gp_device, g_gt_device, neloc_x_device, neloc_y_device);		
	}
}

template <typename scalar_t>
__device__ scalar_t compute_force_left_wall(int i, int sk_01, int ss, int kk, scalar_t wleft,
	scalar_t mu, scalar_t *g_gkn_device, scalar_t *g_gnu_device, scalar_t *g_grho_device, scalar_t *g_gkt_device,
	int np, scalar_t dt_g, int32_t *g_gne_device, int32_t *g_gbar_device, int32_t *g_glay_device, scalar_t *g_gR_device,
	scalar_t *g_gx_device, scalar_t *g_gy_device, scalar_t *g_gvx_device, scalar_t *g_gvy_device,
	scalar_t *g_gax_device, scalar_t *g_gay_device, scalar_t *g_gang_device, scalar_t *g_gangv_device, scalar_t *g_ganga_device,
	scalar_t *g_gfx_device, scalar_t *g_gfy_device, scalar_t *g_gfz_device, scalar_t *g_gp_device, scalar_t *g_gt_device,
	scalar_t *neloc_x_device, scalar_t *neloc_y_device){

	scalar_t dn, dnmin;
	scalar_t vn, fn;
	int32_t mid_ne, mid_bar;
	dnmin = 0;
	int32_t ele_ne; 
	int32_t ele_bar;
	ele_ne = 0;
	ele_bar = 0;
	if (sk_01 == 0){
		for (int s = 0; s < g_gne_device[i]; s++){
			for (int k = 0; k < g_gbar_device[i]; k++){
				dn = neloc_x_device[k + (i + s*np)*g_gbar_device[i]] - g_gR_device[i] - wleft;
				if (dn < dnmin){
					dnmin = dn;
					ele_ne = s;
					ele_bar = k;
				}
			}

		}
	}
	else{
		//printf("  sk==0  \n");
		dn = neloc_x_device[kk + (i + ss*np)*g_gbar_device[i]] - g_gR_device[i] - wleft;
		dnmin = dn;
		ele_ne = ss;
		ele_bar = kk;
	}
	
	
	if (dnmin<0.0 ){

		/* velocity (wall velocity = 0) */
		vn = g_gvx_device[i];
		/* force */
		fn = (-g_gkn_device[i] * dnmin - g_gnu_device[i] * vn)*g_glay_device[i];
		/* Update sum of forces on grains */
		g_gfx_device[i] = g_gfx_device[i] + fn;
		/* Add fn to pressure on grains i */
		g_gp_device[i] = g_gp_device[i] + fn;
	
		/*Update sum of torques on grains */
		mid_ne = int((g_gne_device[i] - 1) / 2);
		mid_bar = int((g_gbar_device[i] - 1) / 2);
		g_gt_device[i] = g_gt_device[i] - fn*(neloc_y_device[ele_bar + (i + ele_ne*np)*g_gbar_device[i]] - neloc_y_device[mid_bar + (i + mid_ne*np)*g_gbar_device[i]]);
		/* Update stress to the wall */
		return fn;
	}
	else{
		return 0;
	}

}

template <typename scalar_t>
__device__ scalar_t compute_force_right_wall(int i, int sk_01, int ss, int kk, scalar_t wright, 
	scalar_t mu, scalar_t *g_gkn_device, scalar_t *g_gnu_device, scalar_t *g_grho_device, scalar_t *g_gkt_device,
	int np, scalar_t dt_g, int *g_gne_device, int *g_gbar_device, int *g_glay_device, scalar_t *g_gR_device,
	scalar_t *g_gx_device, scalar_t *g_gy_device, scalar_t *g_gvx_device, scalar_t *g_gvy_device,
	scalar_t *g_gax_device, scalar_t *g_gay_device, scalar_t *g_gang_device, scalar_t *g_gangv_device, scalar_t *g_ganga_device,
	scalar_t *g_gfx_device, scalar_t *g_gfy_device, scalar_t *g_gfz_device, scalar_t *g_gp_device, scalar_t *g_gt_device,
	scalar_t *neloc_x_device, scalar_t *neloc_y_device)
{
	scalar_t dn, dnmin;
	scalar_t vn, fn;
	int32_t mid_ne, mid_bar;
	dnmin = 0;
	int32_t ele_ne = 0;
	int32_t ele_bar = 0;

	if (sk_01 == 0){
		for (int s = 0; s < g_gne_device[i]; s++){
			for (int k = 0; k < g_gbar_device[i]; k++){
				dn = wright - (neloc_x_device[k + (i + s*np)*g_gbar_device[i]] + g_gR_device[i]);
				if (dn < dnmin){
					dnmin = dn;
					ele_ne = s;
					ele_bar = k;
				}
			}
		}
	}
	else{
		dn = wright - (neloc_x_device[kk + (i + ss*np)*g_gbar_device[i]] + g_gR_device[i]);
		dnmin = dn;
		ele_ne = ss;
		ele_bar = kk;
	}
	

	if (dnmin < 0.0 ){ 
		/* velocity (wall velocity = 0) */
		vn = -g_gvx_device[i];
		/* force */
		fn = (-g_gkn_device[i] * dnmin - g_gnu_device[i] * vn)*g_glay_device[i];
		/* Update sum of forces on grains */
		g_gfx_device[i] = g_gfx_device[i] - fn;
		/* Add fn to pressure on grains i */
		g_gp_device[i] = g_gp_device[i] + fn;
		/*Update sum of torques on grains */
		mid_ne = int((g_gne_device[i] - 1) / 2);
		mid_bar = int((g_gbar_device[i] - 1) / 2);
		g_gt_device[i] = g_gt_device[i] + fn*(neloc_y_device[ele_bar + (i + ele_ne*np)*g_gbar_device[i]] - neloc_y_device[mid_bar + (i + mid_ne*np)*g_gbar_device[i]]);
		/* Update stress to the wall */
		return fn;
		
	}
	else {
		return 0;
	}
}

template <typename scalar_t>
__device__ scalar_t compute_force_upper_wall(int i, int sk_01, int ss, int kk, scalar_t wup,
	scalar_t mu, scalar_t *g_gkn_device, scalar_t *g_gnu_device, scalar_t *g_grho_device, scalar_t *g_gkt_device,
	int np, scalar_t dt_g, int* g_gne_device, int* g_gbar_device, int* g_glay_device, scalar_t *g_gR_device,
	scalar_t *g_gx_device, scalar_t *g_gy_device, scalar_t *g_gvx_device, scalar_t *g_gvy_device,
	scalar_t *g_gax_device, scalar_t *g_gay_device, scalar_t *g_gang_device, scalar_t *g_gangv_device, scalar_t *g_ganga_device,
	scalar_t *g_gfx_device, scalar_t *g_gfy_device, scalar_t *g_gfz_device, scalar_t *g_gp_device, scalar_t *g_gt_device,
	scalar_t *neloc_x_device, scalar_t *neloc_y_device)
{
	scalar_t dn, dnmin;
	scalar_t vn, fn;
	int32_t mid_ne, mid_bar;
	dnmin = 0;
	int32_t ele_ne = 0;
	int32_t ele_bar = 0;

	if (sk_01 == 0) {
		for (int s = 0; s < g_gne_device[i]; s++) {
			for (int k = 0; k < g_gbar_device[i]; k++) {
				dn = wup - (neloc_y_device[k + (i + s * np) * g_gbar_device[i]] + g_gR_device[i]);
				if (dn < dnmin) {
					dnmin = dn;
					ele_ne = s;
					ele_bar = k;
				}
			}

		}
	}
	else {
		dn = wup - (neloc_y_device[kk + (i + ss * np) * g_gbar_device[i]] + g_gR_device[i]);
		dnmin = dn;
		ele_ne = ss;
		ele_bar = kk;
	}

	if (dnmin < 0.0){

		/* velocity (wall velocity = 0) */
		vn = -g_gvy_device[i];
		/* force */
		fn = (-g_gkn_device[i] * dnmin - g_gnu_device[i] * vn)*g_glay_device[i];
		/* Update sum of forces on grains */
		g_gfy_device[i] = g_gfy_device[i] - fn;
		/* Add fn to pressure on grains i */
		g_gp_device[i] = g_gp_device[i] + fn;
		
		/*Update sum of torques on grains */
		mid_ne = int((g_gne_device[i] - 1) / 2);
		mid_bar = int((g_gbar_device[i] - 1) / 2);
		g_gt_device[i] = g_gt_device[i] - fn*(neloc_x_device[ele_bar + (i + ele_ne*np)*g_gbar_device[i]] - neloc_x_device[mid_bar + (i + mid_ne*np)*g_gbar_device[i]]);
		/* Update stress to the wall */
		return fn;
		
	}
	else {
		return 0;
	}
}

template <typename scalar_t>
__device__ scalar_t compute_force_lower_wall(int i, int sk_01, int ss, int kk, scalar_t wdown,
	scalar_t mu, scalar_t *g_gkn_device, scalar_t *g_gnu_device, scalar_t *g_grho_device, scalar_t *g_gkt_device,
	int np, scalar_t dt_g, int *g_gne_device, int *g_gbar_device, int *g_glay_device, scalar_t *g_gR_device,
	scalar_t *g_gx_device, scalar_t *g_gy_device, scalar_t *g_gvx_device, scalar_t *g_gvy_device,
	scalar_t *g_gax_device, scalar_t *g_gay_device, scalar_t *g_gang_device, scalar_t *g_gangv_device, scalar_t *g_ganga_device,
	scalar_t *g_gfx_device, scalar_t *g_gfy_device, scalar_t *g_gfz_device, scalar_t *g_gp_device, scalar_t *g_gt_device,
	scalar_t *neloc_x_device, scalar_t *neloc_y_device)
{

	scalar_t dn, dnmin;
	scalar_t vn, fn;
	int32_t mid_ne, mid_bar;
	dnmin = 0;
	int32_t ele_ne = 0;
	int32_t ele_bar = 0;
	if (sk_01 == 0){
		for (int s = 0; s < g_gne_device[i]; s++){
			for (int k = 0; k < g_gbar_device[i]; k++){
				dn = neloc_y_device[k + (i + s*np)*g_gbar_device[i]] - g_gR_device[i] - wdown;
				if (dn < dnmin){
					dnmin = dn;
					ele_ne = s;
					ele_bar = k;
				}
			}

		}
	}
	else{
		dn = neloc_y_device[kk + (i + ss*np)*g_gbar_device[i]] - g_gR_device[i] - wdown;
		dnmin = dn;
		ele_ne = ss;
		ele_bar = kk;
	}
	
	if (dnmin < 0.0 ){
		/* velocity (wall velocity = 0) */
		vn = g_gvy_device[i];
		/* force */
		fn = (-g_gkn_device[i] * dnmin - g_gnu_device[i] * vn)*g_glay_device[i];
		/* Update sum of forces on grains */
		g_gfy_device[i] = g_gfy_device[i] + fn;
		/* Add fn to pressure on grains i */
		g_gp_device[i] = g_gp_device[i] + fn;
		/*Update sum of torques on grains */
		mid_ne = int((g_gne_device[i] - 1) / 2);
		mid_bar = int((g_gbar_device[i] - 1) / 2);
		g_gt_device[i] = g_gt_device[i] + fn*(neloc_x_device[ele_bar + (i + ele_ne*np)*g_gbar_device[i]] - neloc_x_device[mid_bar + (i + mid_ne*np)*g_gbar_device[i]]);
		/* Update stress to the wall */
		return fn;
		
	}
	else{
		return 0;
	}

}

void interact_walls_cuda(at::Tensor wleft, at::Tensor wright, at::Tensor wup, at::Tensor wdown, 
							at::Tensor wp_up, at::Tensor wp_down, at::Tensor wp_left, at::Tensor wp_right,
							at::Tensor mu, at::Tensor g_gkn_device, at::Tensor g_gnu_device, at::Tensor g_grho_device, at::Tensor g_gkt_device,
							at::Tensor dt_g, at::Tensor g_gne_device, at::Tensor g_gbar_device, at::Tensor g_glay_device, at::Tensor g_gR_device,
							at::Tensor g_gx_device, at::Tensor g_gy_device, at::Tensor g_gvx_device, at::Tensor g_gvy_device,
							at::Tensor g_gax_device, at::Tensor g_gay_device, at::Tensor g_gang_device, at::Tensor g_gangv_device, at::Tensor g_ganga_device,
							at::Tensor g_gfx_device, at::Tensor g_gfy_device, at::Tensor g_gfz_device, at::Tensor g_gp_device, at::Tensor g_gt_device,
							at::Tensor neloc_x_device, at::Tensor neloc_y_device, at::Tensor t) {

  at::cuda::CUDAGuard device_guard(g_gx_device.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int np = g_gx_device.numel();
  if (np == 0) {
    return;
  }

  int thread_0 = 512;
  int block_0 = (np + 512 - 1) / 512;

  AT_DISPATCH_FLOATING_TYPES(
	g_gx_device.type(), "interact_walls_cuda", ([&] {
        interact_walls_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
            np, wleft.data<scalar_t>(), wright.data<scalar_t>(), wup.data<scalar_t>(), wdown.data<scalar_t>(),
			wp_up.data<scalar_t>(), wp_down.data<scalar_t>(), wp_left.data<scalar_t>(), wp_right.data<scalar_t>(), 
			mu.data<scalar_t>(), g_gkn_device.data<scalar_t>(), g_gnu_device.data<scalar_t>(), g_grho_device.data<scalar_t>(), g_gkt_device.data<scalar_t>(),
			dt_g.data<scalar_t>(), g_gne_device.data<int32_t>(), g_gbar_device.data<int32_t>(), g_glay_device.data<int32_t>(), g_gR_device.data<scalar_t>(),
			g_gx_device.data<scalar_t>(), g_gy_device.data<scalar_t>(), g_gvx_device.data<scalar_t>(), g_gvy_device.data<scalar_t>(),
			g_gax_device.data<scalar_t>(), g_gay_device.data<scalar_t>(), g_gang_device.data<scalar_t>(), g_gangv_device.data<scalar_t>(), g_ganga_device.data<scalar_t>(),
			g_gfx_device.data<scalar_t>(), g_gfy_device.data<scalar_t>(), g_gfz_device.data<scalar_t>(), g_gp_device.data<scalar_t>(), g_gt_device.data<scalar_t>(),
			neloc_x_device.data<scalar_t>(), neloc_y_device.data<scalar_t>(), t.data<scalar_t>());
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
