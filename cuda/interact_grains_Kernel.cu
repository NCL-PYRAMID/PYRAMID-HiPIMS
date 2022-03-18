// #include <torch/extension.h>
#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

const double pi = 3.14159;
// index of machanical properties
const int i_kn = 0;    // Normal stiffness    法向刚度系数    - 塑料/车
const int i_nu = 1;    // Normal damping    法向粘性阻尼系数
const int i_kt = 2;    // Tangential stiffness    切向刚度系数

/*computer force between two discs */
template <typename scalar_t>
__device__ __forceinline__ void interparticle_force(int a, int b, scalar_t mu, scalar_t *g_properties, 
	int np, scalar_t dt_g, int32_t *g_gne_device, int32_t *g_gbar_device, int32_t *g_glay_device, scalar_t *g_gR_device,
	scalar_t *g_gvx_device, scalar_t *g_gvy_device, scalar_t *g_gang_device, scalar_t *g_gangv_device, 
	scalar_t *g_gfx_device, scalar_t *g_gfy_device,
	scalar_t *g_gp_device, scalar_t *g_gt_device,
	scalar_t *neloc_x_device, scalar_t *neloc_y_device){
	/* Particle center coordinate component differences */
	double x_ab, y_ab;//z_ab;

	/* Particle center distance */
	double dist;
	/* Size of overlap */
	double dn;// dn_min;

	double xn, yn, vn, fn; /* Normal components */
	
	double xt, yt, vt, ft; /* Tangential components */
	
	double vx_ab;
	double vy_ab;
	
	//printf(" interact force part1 \n");
	for (int s = 0; s < g_gne_device[a]; s++){
		for (int k = 0; k < g_gbar_device[a]; k++){
			for (int r = 0; r < g_gne_device[b]; r++){
				for (int j = 0; j < g_gbar_device[b]; j++){
					x_ab = neloc_x_device[k + (a + s*np)*g_gbar_device[a]] - neloc_x_device[j + (b + r*np)*g_gbar_device[b]];
					y_ab = neloc_y_device[k + (a + s*np)*g_gbar_device[a]] - neloc_y_device[j + (b + r*np)*g_gbar_device[b]];
					dist = sqrt(x_ab*x_ab + y_ab*y_ab);
					dn = dist - (g_gR_device[a] + g_gR_device[b]);

					if (dn < 0.0) {   //Contact
						xn = x_ab / dist; //cos beta
						yn = y_ab / dist; //sin beta
						xt = -yn;
						yt = xn;

						/* Compute the velocity of the contact */
						vx_ab = g_gvx_device[a] - g_gvx_device[b];
						vy_ab = g_gvy_device[a] - g_gvy_device[b];
						vn = vx_ab*xn + vy_ab*yn;
						vt = vx_ab*xt + vy_ab*yt - (((s - (g_gne_device[a] - 1.0)*0.5) + 1)*g_gR_device[a] * g_gangv_device[a]
							+ ((r - (g_gne_device[b] - 1.0)*0.5) + 1)*g_gR_device[b] * g_gangv_device[b]);
						

						//角加速度与线速度转化乘以的距离 约等于所在球到质心的最远距离

						/* Compute force in local axes */
						fn = -g_properties[a + i_kn*np] * dn - g_properties[a + i_nu*np] * vn;
						ft = fabs(g_properties[a + i_kt*np] * vt*dt_g);
						if (ft > mu*fn) {   /* Coefficient of friction */
							ft = mu*fn;
						}

						if (vt > 0) {
							ft = -ft;
						}

						/* Calculate sum of forces on a and b in global coordinates */
						g_gfx_device[a] = g_gfx_device[a] + fn * xn*g_glay_device[a];
						g_gfy_device[a] = g_gfy_device[a] + fn * yn*g_glay_device[a];
						g_gp_device[a] = g_gp_device[a] + fn*g_glay_device[a];
						g_gt_device[a] = g_gt_device[a] - ft*g_glay_device[a] * (s - (g_gne_device[a] - 1) / 2)*g_gR_device[a] * sin(g_gang_device[a] / 180 * pi - acos(xn));// 颗粒间相互作用力力臂

						g_gfx_device[b] = g_gfx_device[b] - fn * xn*g_glay_device[b];
						g_gfy_device[b] = g_gfy_device[b] - fn * yn*g_glay_device[b];
						g_gp_device[b] = g_gp_device[b] + fn*g_glay_device[b];
						g_gt_device[b] = g_gt_device[b] - ft*g_glay_device[b] * (r - (g_gne_device[b] - 1) / 2)*g_gR_device[b] * sin(g_gang_device[b] / 180 * pi - acos(xn));// 颗粒间相互作用力力臂
					}
				}
			}
		}
	}
}

template <typename scalar_t>
__global__ void interact_grains_kernel(int np, 
	scalar_t *__restrict__ mu, 
	scalar_t *__restrict__ g_properties, 
	scalar_t *__restrict__ dt_g, 
	int32_t *__restrict__ g_gne_device, 
	int32_t *__restrict__ g_gbar_device, 
	int32_t *__restrict__ g_glay_device, 
	scalar_t *__restrict__ g_gR_device,
	scalar_t *__restrict__ g_gvx_device,
	scalar_t *__restrict__ g_gvy_device,
	scalar_t *__restrict__ g_gang_device, 
	scalar_t *__restrict__ g_gangv_device, 
	scalar_t *__restrict__ g_gfx_device, 
	scalar_t *__restrict__ g_gfy_device, 
	scalar_t *__restrict__ g_gp_device, 
	scalar_t *__restrict__ g_gt_device,
	scalar_t *__restrict__ neloc_x_device, 
	scalar_t *__restrict__ neloc_y_device){
	
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	int b;
	int c, d;
	int e, f;
	double dx, dy, dL;

	if (a<np) {
		for (b = a + 1; b < np; b++){
			c = int((g_gne_device[a] - 1) / 2);
			d = int((g_gne_device[b] - 1) / 2);
			e = int((g_gbar_device[a] - 1) / 2);
			f = int((g_gbar_device[b] - 1) / 2);

			// printf("%d \t" "%d \n", a, e + (a + c*np)*g_gbar_device[a]);

			dx = neloc_x_device[e + (a + c*np)*g_gbar_device[a]] - neloc_x_device[f + (b + d*np)*g_gbar_device[b]];
			dy = neloc_y_device[e + (a + c*np)*g_gbar_device[a]] - neloc_y_device[f + (b + d*np)*g_gbar_device[b]];
			dL = g_gR_device[a] * sqrt(double((c + 1)*(c + 1) / 4.0 + (e + 1)*(e + 1) / 4.0)) + g_gR_device[b] * sqrt(double((d + 1)*(d + 1) / 4.0 + (f + 1) *(f + 1) / 4.0));
			if (sqrt(dx*dx + dy*dy) <= dL){
				interparticle_force(a, 
					b, 
					mu[0], 
					g_properties, 
					np, 
					dt_g[0], 
					g_gne_device, 
					g_gbar_device, 
					g_glay_device, 
					g_gR_device,
					g_gvx_device,
					g_gvy_device,
					g_gang_device, 
					g_gangv_device,
					g_gfx_device, 
					g_gfy_device, 
					g_gp_device, 
					g_gt_device, 
					neloc_x_device, 
					neloc_y_device);
			}
			// print(g_gne_device);
		}
		// printf("%d \n", a);
	}
}


 
void interact_grains_cuda(at::Tensor mu, 
	at::Tensor g_properties, 
	at::Tensor dt_g, 
	at::Tensor ne, 
	at::Tensor bar, 
	at::Tensor lay, 
	at::Tensor gR,
	at::Tensor gvx,
	at::Tensor gvy,
	at::Tensor gang, 
	at::Tensor gangv, 
	at::Tensor gfx, 
	at::Tensor gfy, 
	at::Tensor gp, 
	at::Tensor gt,
	at::Tensor ex, 
	at::Tensor ey) {
								
    const int np = gvx.numel();
    at::cuda::CUDAGuard device_guard(gvx.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int thread_0 = 512;
    int block_0 = (np + 512 - 1) / 512;

    AT_DISPATCH_FLOATING_TYPES(
        gvx.type(), "interact_grains_cuda", ([&] {
			interact_grains_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
              np, mu.data<scalar_t>(), 
			  g_properties.data<scalar_t>(),
			  dt_g.data<scalar_t>(), 
			  ne.data<int32_t>(), 
			  bar.data<int32_t>(), 
			  lay.data<int32_t>(), 
			  gR.data<scalar_t>(),
			  gvx.data<scalar_t>(),
			  gvy.data<scalar_t>(),
			  gang.data<scalar_t>(), 
			  gangv.data<scalar_t>(),
			  gfx.data<scalar_t>(),
			  gfy.data<scalar_t>(), 
			  gp.data<scalar_t>(), 
			  gt.data<scalar_t>(),
			  ex.data<scalar_t>(), 
			  ey.data<scalar_t>());
        }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}

