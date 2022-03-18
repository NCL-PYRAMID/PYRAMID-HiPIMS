#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

// index of machanical properties
const int i_kn = 0;    // Normal stiffness    法向刚度系数    - 塑料/车
const int i_nu = 1;    // Normal damping    法向粘性阻尼系数
const int i_kt = 2;    // Tangential stiffness    切向刚度系数

template <typename scalar_t>
__device__ scalar_t compute_force_left_wall(int i, int sk_01, int ss, int kk, scalar_t wleft,
	scalar_t mu, scalar_t kn, scalar_t nu, scalar_t kt,
	int np, int ne, int bar, int lay, scalar_t gR, 
	scalar_t gv, scalar_t &gf, scalar_t &gp, scalar_t &gt,
	scalar_t *neloc_x_device, scalar_t *neloc_y_device)
{
	scalar_t dn, dnmin;
	scalar_t vn, fn;
	int32_t mid_ne, mid_bar;
	dnmin = 0;
	int32_t ele_ne; 
	int32_t ele_bar;
	ele_ne = 0;
	ele_bar = 0;
	if (sk_01 == 0){
		for (int s = 0; s < ne; s++){
			for (int k = 0; k < bar; k++){
				dn = neloc_x_device[k + (i + s*np)*bar] - gR - wleft;
				if (dn < dnmin){
					dnmin = dn;
					ele_ne = s;
					ele_bar = k;
				}
			}

		}
	}
	else{
		dn = neloc_x_device[kk + (i + ss*np)*bar] - gR - wleft;
		dnmin = dn;
		ele_ne = ss;
		ele_bar = kk;
	}
	
	
	if (dnmin<0.0 ){

		/* velocity (wall velocity = 0) */
		vn = gv;
		/* force */
		fn = (-kn * dnmin - nu * vn)*lay;
		/* Update sum of forces on grains */
		gf = gf + fn;
		/* Add fn to pressure on grains i */
		gp = gp + fn;
	
		/*Update sum of torques on grains */
		mid_ne = int((ne - 1) / 2);
		mid_bar = int((bar - 1) / 2);
		gt = gt - fn*(neloc_y_device[ele_bar + (i + ele_ne*np)*bar] - neloc_y_device[mid_bar + (i + mid_ne*np)*bar]);
		/* Update stress to the wall */
		return fn;
	}
	else{
		return 0;
	}

}

template <typename scalar_t>
__device__ scalar_t compute_force_right_wall(int i, int sk_01, int ss, int kk, scalar_t wright,
	scalar_t mu, scalar_t kn, scalar_t nu, scalar_t kt,
	int np, int ne, int bar, int lay, scalar_t gR, 
	scalar_t gv, scalar_t &gf, scalar_t &gp, scalar_t &gt,
	scalar_t *neloc_x_device, scalar_t *neloc_y_device)
{
	scalar_t dn, dnmin;
	scalar_t vn, fn;
	int32_t mid_ne, mid_bar;
	dnmin = 0;
	int32_t ele_ne = 0;
	int32_t ele_bar = 0;

	if (sk_01 == 0){
		for (int s = 0; s < ne; s++){
			for (int k = 0; k < bar; k++){
				dn = wright - (neloc_x_device[k + (i + s*np)*bar] + gR);
				if (dn < dnmin){
					dnmin = dn;
					ele_ne = s;
					ele_bar = k;
				}
			}
		}
	}
	else{
		dn = wright - (neloc_x_device[kk + (i + ss*np)*bar] + gR);
		dnmin = dn;
		ele_ne = ss;
		ele_bar = kk;
	}
	

	if (dnmin < 0.0 ){ 
		/* velocity (wall velocity = 0) */
		vn = -gv;
		/* force */
		fn = (-kn * dnmin - nu * vn)*lay;
		/* Update sum of forces on grains */
		gf = gf - fn;
		/* Add fn to pressure on grains i */
		gp = gp + fn;
		/*Update sum of torques on grains */
		mid_ne = int((ne - 1) / 2);
		mid_bar = int((bar - 1) / 2);
		gt = gt + fn*(neloc_y_device[ele_bar + (i + ele_ne*np)*bar] - neloc_y_device[mid_bar + (i + mid_ne*np)*bar]);
		/* Update stress to the wall */
		return fn;
		
	}
	else {
		return 0;
	}
}
template <typename scalar_t>
__device__ scalar_t compute_force_upper_wall(int i, int sk_01, int ss, int kk, scalar_t wup,
	scalar_t mu, scalar_t kn, scalar_t nu, scalar_t kt,
	int np, int ne, int bar, int lay, scalar_t gR, 
	scalar_t gv, scalar_t &gf, scalar_t &gp, scalar_t &gt,
	scalar_t *neloc_x_device, scalar_t *neloc_y_device)
{
	scalar_t dn, dnmin;
	scalar_t vn, fn;
	int32_t mid_ne, mid_bar;
	dnmin = 0;
	int32_t ele_ne = 0;
	int32_t ele_bar = 0;

	if (sk_01 == 0) {
		for (int s = 0; s < ne; s++) {
			for (int k = 0; k < bar; k++) {
				dn = wup - (neloc_y_device[k + (i + s * np) * bar] + gR);
				if (dn < dnmin) {
					dnmin = dn;
					ele_ne = s;
					ele_bar = k;
				}
			}

		}
	}
	else {
		dn = wup - (neloc_y_device[kk + (i + ss * np) * bar] + gR);
		dnmin = dn;
		ele_ne = ss;
		ele_bar = kk;
	}

	if (dnmin < 0.0){

		/* velocity (wall velocity = 0) */
		vn = -gv;
		/* force */
		fn = (-kn * dnmin - nu * vn)*lay;
		/* Update sum of forces on grains */
		gf = gf - fn;
		/* Add fn to pressure on grains i */
		gp = gp + fn;
		
		/*Update sum of torques on grains */
		mid_ne = int((ne - 1) / 2);
		mid_bar = int((bar - 1) / 2);
		gt = gt - fn*(neloc_x_device[ele_bar + (i + ele_ne*np)*bar] - neloc_x_device[mid_bar + (i + mid_ne*np)*bar]);
		/* Update stress to the wall */
		return fn;
		
	}
	else {
		return 0;
	}
}

template <typename scalar_t>
__device__ scalar_t compute_force_lower_wall(int i, int sk_01, int ss, int kk, scalar_t wdown,
						scalar_t mu, scalar_t kn, scalar_t nu, scalar_t kt,
						int np, int ne, int bar, int lay, scalar_t gR, 
						scalar_t gv, scalar_t &gf, scalar_t &gp, scalar_t &gt,
						scalar_t *neloc_x_device, scalar_t *neloc_y_device)
{

	scalar_t dn, dnmin;
	scalar_t vn, fn;
	int32_t mid_ne, mid_bar;
	dnmin = 0;
	int32_t ele_ne = 0;
	int32_t ele_bar = 0;
	if (sk_01 == 0){
		for (int s = 0; s < ne; s++){
			for (int k = 0; k < bar; k++){
				// printf("%d \t" "%d \n", i, k + (i + s*np)*bar);
				dn = neloc_y_device[k + (i + s*np)*bar] - gR - wdown;
				if (dn < dnmin){
					dnmin = dn;
					ele_ne = s;
					ele_bar = k;
				}
			}

		}
	}
	else{
		dn = neloc_y_device[kk + (i + ss*np)*bar] - gR - wdown;
		dnmin = dn;
		ele_ne = ss;
		ele_bar = kk;
	}
	
	if (dnmin < 0.0 ){
		/* velocity (wall velocity = 0) */
		vn = gv;
		/* force */
		fn = (-kn * dnmin - nu * vn)*lay;
		/* Update sum of forces on grains */
		gf = gf + fn;
		/* Add fn to pressure on grains i */
		gp = gp + fn;
		/*Update sum of torques on grains */
		mid_ne = int((ne - 1) / 2);
		mid_bar = int((bar - 1) / 2);
		gt = gt + fn*(neloc_x_device[ele_bar + (i + ele_ne*np)*bar] - neloc_x_device[mid_bar + (i + mid_ne*np)*bar]);
		/* Update stress to the wall */
		return fn;
		
	}
	else{
		return 0;
	}

}

template <typename scalar_t>
__global__ void interact_walls_kernel(const int np, scalar_t *__restrict__ walls, 
									scalar_t *__restrict__ mu, 
									scalar_t *__restrict__ g_properties, 
									int32_t *__restrict__ g_gne_device, 
									int32_t *__restrict__ g_gbar_device, 
									int32_t *__restrict__ g_glay_device, 
									scalar_t *__restrict__ g_gR_device,	
									scalar_t *__restrict__ g_gvx_device, 
									scalar_t *__restrict__ g_gvy_device,
									scalar_t *__restrict__ g_gfx_device,
									scalar_t *__restrict__ g_gfy_device, 
									scalar_t *__restrict__ g_gp_device, 
									scalar_t *__restrict__ g_gt_device,
									scalar_t *__restrict__ neloc_x_device, 
									scalar_t *__restrict__ neloc_y_device){

	int  i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i<np) {
		scalar_t wall_u = 0.0;
		scalar_t wall_d = 0.0;
		scalar_t wall_r = 0.0;
		scalar_t wall_l = 0.0;

		scalar_t kn = g_properties[i + i_kn*np];
		scalar_t nu = g_properties[i + i_nu*np];
		scalar_t kt = g_properties[i + i_kt*np]; 
		scalar_t gfx = g_gfx_device[i];
		scalar_t gfy = g_gfy_device[i];
		scalar_t gvx = g_gvx_device[i];
		scalar_t gvy = g_gvy_device[i];
		int32_t ne = g_gne_device[i];
		int32_t bar = g_gbar_device[i];
		int32_t lay = g_glay_device[i]; 
		scalar_t gR = g_gR_device[i];
		
		scalar_t gp = g_gp_device[i];
		scalar_t gt = g_gt_device[i];

		// 0- left, 1-down, 2-right, 3-up
		wall_u = compute_force_upper_wall(i, 0, 0, 0, walls[3], mu[0], kn, nu, kt, 
			np, ne, bar, lay, gR, gvy, gfy, gp, gt, neloc_x_device, neloc_y_device);
		wall_d = compute_force_lower_wall(i, 0, 0, 0, walls[1], mu[0], kn, nu, kt, 
			np, ne, bar, lay, gR, gvy, gfy, gp, gt, neloc_x_device, neloc_y_device);
		wall_r = compute_force_right_wall(i, 0, 0, 0, walls[2], mu[0], kn, nu, kt, 
			np, ne, bar, lay, gR, gvx, gfx, gp, gt, neloc_x_device, neloc_y_device);
		// printf("compute force right wall");
		wall_l = compute_force_left_wall(i, 0, 0, 0, walls[0], mu[0], kn, nu, kt, 
			np, ne, bar, lay, gR, gvx, gfx, gp, gt, neloc_x_device, neloc_y_device);	
		
		// update variables
		g_gfx_device[i] = gfx;
		g_gfy_device[i] = gfy;
		g_gp_device[i] = gp;
		g_gt_device[i] = gt;
	}
}

void interact_walls_cuda(at::Tensor walls,
							at::Tensor mu, 
							at::Tensor g_properties, 
							at::Tensor g_gne_device, 
							at::Tensor g_gbar_device, 
							at::Tensor g_glay_device, 
							at::Tensor g_gR_device,
							at::Tensor g_gvx_device,
							at::Tensor g_gvy_device,
							at::Tensor g_gfx_device,
							at::Tensor g_gfy_device, 
							at::Tensor g_gp_device, 
							at::Tensor g_gt_device,
							at::Tensor neloc_x_device, 
							at::Tensor neloc_y_device) {

  at::cuda::CUDAGuard device_guard(g_gvx_device.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int np = g_gvx_device.numel();

  if (np == 0) {
    return;
  }

  int thread_0 = 512;
  int block_0 = (np + 512 - 1) / 512;

  AT_DISPATCH_FLOATING_TYPES(
	g_gvx_device.type(), "interact_walls_cuda", ([&] {
        interact_walls_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
            np, walls.data<scalar_t>(), 
			mu.data<scalar_t>(), 
			g_properties.data<scalar_t>(), 
			g_gne_device.data<int32_t>(), 
			g_gbar_device.data<int32_t>(), 
			g_glay_device.data<int32_t>(), 
			g_gR_device.data<scalar_t>(), 
			g_gvx_device.data<scalar_t>(),
			g_gvy_device.data<scalar_t>(),
			g_gfx_device.data<scalar_t>(), 
			g_gfy_device.data<scalar_t>(), 
			g_gp_device.data<scalar_t>(), 
			g_gt_device.data<scalar_t>(),
			neloc_x_device.data<scalar_t>(), 
			neloc_y_device.data<scalar_t>());
      }));
	
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
