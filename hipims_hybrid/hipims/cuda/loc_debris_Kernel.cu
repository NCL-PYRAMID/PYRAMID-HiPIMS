#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

const double pi = 3.14159;

template <typename scalar_t>
__global__ void loc_debris(int M, int N, double dx, double dy, int *num_x_device, int *num_y_device,
	double *zb_device, double *z_device, double *qx_device, double *qy_device, double *z_rela, double *r_rela, int *lay_sub, 
	int np, double dt_g, int *g_gne_device, int *g_gbar_device, int *g_glay_device, double *g_gR_device, double *g_gm_device, double *g_gI_device,
	double *g_gz_device, double *g_gx_device, double *g_gy_device, double *neloc_x_device, double *neloc_y_device,
	double mu, double *g_gkn_device, double *g_gnu_device, double *g_grho_device, double *g_gkt_device,
	double *g_gvx_device, double *g_gvy_device,	double *g_gax_device, double *g_gay_device, double *g_gang_device, double *g_gangv_device, double *g_ganga_device,
	double *g_gfx_device, double *g_gfy_device, double *g_gfz_device, double *g_gp_device, double *g_gt_device,	
	double *zb_g, double *z_g, double *qx_g, double *qy_g, double *h_g, double *zb_xf, double *z_xf, double *qx_xf, double *h_xf, 
	double *zb_yf, double *z_yf, double *qy_yf, double *h_yf, double *zb_xb, double *z_xb, double *qx_xb, double *h_xb, 
	double *zb_yb, double *z_yb, double *qy_yb, double *h_yb, double *zb_xff, double *z_xff, double *qx_xff, double *h_xff, 
	double *zb_yff, double *z_yff, double *qy_yff, double *h_yff,double *zb_xbb, double *z_xbb, double *qx_xbb, double *h_xbb,
	double *zb_ybb, double *z_ybb, double *qy_ybb, double *h_ybb, double t_dem, 
	double *cell_f_xP_device, double *cell_f_xN_device, double *cell_f_yP_device, double *cell_f_yN_device)
{
	const double tol_h = 1.0e-10;

	const double grav = 9.81;

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	
	double h_xbb_max, h_xff_max, h_ybb_max, h_yff_max, h_g_max, h_xb_max, h_xf_max, h_yb_max, h_yf_max;
	
	int h_xbb_max_ele, h_xff_max_ele, h_ybb_max_ele, h_yff_max_ele, h_g_max_ele, h_xb_max_ele, h_xf_max_ele, h_yb_max_ele, h_yf_max_ele;
	
	int num_x_front, num_x_back, num_xf, num_xb;
	int num_y_front, num_y_back, num_yf, num_yb;
	int num_xff, num_xbb, num_yff, num_ybb;
	//int *num_xx, *num_yy, *num_xc, *num_yc;
	int num_xc, num_yc;
	
	//double g_sub, h_tri;
	double obs_r, obs_l, obs_u, obs_d;
	double obsp_r, obsp_l, obsp_u, obsp_d;

	double h_maxs;
	double h_max;
	double xff, xbb, yff, ybb;
	int cell_r, cell_l, cell_u, cell_d;
	int deb_xc, deb_yc;

	//printf(" enter update acc \n");

	while (i<np){
		z_rela[i] = g_gm_device[i] / (1000 * g_gR_device[i] * (g_gbar_device[i] + 1) * g_gR_device[i] * (g_gne_device[i] + 1));//rho * g * V
		lay_sub[i] = int(z_rela[i] / (2 * g_gR_device[i]));
		r_rela[i] = (lay_sub[i] * 2 + 1) * g_gR_device[i] - z_rela[i];

		//correct gz
		//the location of the central of debris
		deb_xc = int(floor(g_gx_device[i] / dx));
		deb_yc = int(floor(g_gy_device[i] / dy));
		g_gz_device[i] = z_device[deb_xc + deb_yc * N] + g_glay_device[i] * g_gR_device[i];


		for (int s = 0; s < g_gne_device[i]; s++){
			for (int k = 0; k < g_gbar_device[i]; k++){
				//为了计算水流作用力，和判断漂浮物前后左右四个方向上的zb，所以用了比较多的定位关系。
				num_x_device[k + (i + s*np)*g_gbar_device[i]] = int(floor(neloc_x_device[k + (i + s*np)*g_gbar_device[i]] / dx));  //i--N
				num_y_device[k + (i + s*np)*g_gbar_device[i]] = int(floor(neloc_y_device[k + (i + s*np)*g_gbar_device[i]] / dy));  //j--M

				num_x_front = int(ceil((neloc_x_device[k + (i + s*np)*g_gbar_device[i]] + g_gR_device[i]) / dx)) + 2;
				num_x_back = int(ceil((neloc_x_device[k + (i + s*np)*g_gbar_device[i]] - g_gR_device[i]) / dx)) - 2;
				num_y_front = int(ceil((neloc_y_device[k + (i + s*np)*g_gbar_device[i]] + g_gR_device[i]) / dy)) + 2;
				num_y_back = int(ceil((neloc_y_device[k + (i + s*np)*g_gbar_device[i]] - g_gR_device[i]) / dy)) - 2;

				num_xc = dem_bound(num_x_device[k + (i + s*np)*g_gbar_device[i]], N);
				num_yc = dem_bound(num_y_device[k + (i + s*np)*g_gbar_device[i]], M);

				num_xf = dem_bound(num_x_front, N);
				num_xb = dem_bound(num_x_back, N);
				num_yf = dem_bound(num_y_front, M);
				num_yb = dem_bound(num_y_back, M);

				num_x_front = int(floor((neloc_x_device[k + (i + s*np)*g_gbar_device[i]] + g_gR_device[i]) / dx));
				num_x_back = int(floor((neloc_x_device[k + (i + s*np)*g_gbar_device[i]] - g_gR_device[i]) / dx));
				num_y_front = int(floor((neloc_y_device[k + (i + s*np)*g_gbar_device[i]] + g_gR_device[i]) / dy));
				num_y_back = int(floor((neloc_y_device[k + (i + s*np)*g_gbar_device[i]] - g_gR_device[i]) / dy));

				num_xff = dem_bound(num_x_front, N);
				num_xbb = dem_bound(num_x_back, N);

				num_yff = dem_bound(num_y_front, M);
				num_ybb = dem_bound(num_y_back, M);

				zb_g[k + (i + s*np)*g_gbar_device[i]] = zb_device[num_xc + num_yc * N];
				z_g[k + (i + s*np)*g_gbar_device[i]] = z_device[num_xc + num_yc * N];
				

				qx_g[k + (i + s*np)*g_gbar_device[i]] = qx_device[num_xc + num_yc * N];
				qy_g[k + (i + s*np)*g_gbar_device[i]] = qy_device[num_xc + num_yc * N];

				h_g[k + (i + s*np)*g_gbar_device[i]] = z_g[k + (i + s*np)*g_gbar_device[i]] - zb_g[k + (i + s*np)*g_gbar_device[i]];

				zb_xf[k + (i + s*np)*g_gbar_device[i]] = zb_device[num_xf + num_yc * N];
				z_xf[k + (i + s*np)*g_gbar_device[i]] = z_device[num_xf + num_yc * N];
				qx_xf[k + (i + s*np)*g_gbar_device[i]] = qx_device[num_xf + num_yc * N];

				zb_xb[k + (i + s*np)*g_gbar_device[i]] = zb_device[num_xb + num_yc * N];
				z_xb[k + (i + s*np)*g_gbar_device[i]] = z_device[num_xb + num_yc * N];
				qx_xb[k + (i + s*np)*g_gbar_device[i]] = qx_device[num_xb + num_yc * N];

				zb_yf[k + (i + s*np)*g_gbar_device[i]] = zb_device[num_xc + num_yf * N];
				z_yf[k + (i + s*np)*g_gbar_device[i]] = z_device[num_xc + num_yf * N];
				qy_yf[k + (i + s*np)*g_gbar_device[i]] = qy_device[num_xc + num_yf * N];

				zb_yb[k + (i + s*np)*g_gbar_device[i]] = zb_device[num_xc + num_yb * N];
				z_yb[k + (i + s*np)*g_gbar_device[i]] = z_device[num_xc + num_yb * N];
				qy_yb[k + (i + s*np)*g_gbar_device[i]] = qy_device[num_xc + num_yb * N];


				zb_xff[k + (i + s*np)*g_gbar_device[i]] = zb_device[num_xff + num_yc * N];
				z_xff[k + (i + s*np)*g_gbar_device[i]] = z_device[num_xff + num_yc * N];
				qx_xff[k + (i + s*np)*g_gbar_device[i]] = qx_device[num_xff + num_yc * N];



				zb_xbb[k + (i + s*np)*g_gbar_device[i]] = zb_device[num_xbb + num_yc * N];
				z_xbb[k + (i + s*np)*g_gbar_device[i]] = z_device[num_xbb + num_yc * N];
				qx_xbb[k + (i + s*np)*g_gbar_device[i]] = qx_device[num_xbb + num_yc * N];

				zb_yff[k + (i + s*np)*g_gbar_device[i]] = zb_device[num_xc + num_yff * N];
				z_yff[k + (i + s*np)*g_gbar_device[i]] = z_device[num_xc + num_yff * N];
				qy_yff[k + (i + s*np)*g_gbar_device[i]] = qy_device[num_xc + num_yff * N];

				zb_ybb[k + (i + s*np)*g_gbar_device[i]] = zb_device[num_xc + num_ybb * N];
				z_ybb[k + (i + s*np)*g_gbar_device[i]] = z_device[num_xc + num_ybb * N];
				qy_ybb[k + (i + s*np)*g_gbar_device[i]] = qy_device[num_xc + num_ybb * N];
				xff = zb_device[num_xff + num_yc*N];
				for (int ghost = num_ybb; ghost <= num_yff; ghost++){
					if (zb_device[num_xff + ghost*N] > xff){
						xff = zb_device[num_xff + ghost*N];
					}
				}
				xbb = zb_device[num_xbb + num_yc*N];
				for (int ghost = num_ybb; ghost <= num_yff; ghost++){
					if (zb_device[num_xbb + ghost*N] > xbb){
						xbb = zb_device[num_xbb + ghost*N];
					}
				}
				yff = zb_device[num_xc + num_yff*N];
				for (int ghost = num_xbb; ghost <= num_xff; ghost++){
					if (zb_device[ghost + num_yff*N] > yff){
						yff = zb_device[ghost + num_yff*N];
					}
				}
				ybb = zb_device[num_xc + num_ybb*N];
				for (int ghost = num_xbb; ghost <= num_xff; ghost++){
					if (zb_device[ghost + num_ybb*N] > ybb){
						ybb = zb_device[ghost + num_ybb*N];
					}
				}

				h_xf[k + (i + s*np)*g_gbar_device[i]] = z_xf[k + (i + s*np)*g_gbar_device[i]] - zb_xf[k + (i + s*np)*g_gbar_device[i]];
				h_xb[k + (i + s*np)*g_gbar_device[i]] = z_xb[k + (i + s*np)*g_gbar_device[i]] - zb_xb[k + (i + s*np)*g_gbar_device[i]];
				h_yf[k + (i + s*np)*g_gbar_device[i]] = z_yf[k + (i + s*np)*g_gbar_device[i]] - zb_yf[k + (i + s*np)*g_gbar_device[i]];
				h_yb[k + (i + s*np)*g_gbar_device[i]] = z_yb[k + (i + s*np)*g_gbar_device[i]] - zb_yb[k + (i + s*np)*g_gbar_device[i]];
				h_xff[k + (i + s*np)*g_gbar_device[i]] = z_xff[k + (i + s*np)*g_gbar_device[i]] - zb_xff[k + (i + s*np)*g_gbar_device[i]];
				h_xbb[k + (i + s*np)*g_gbar_device[i]] = z_xbb[k + (i + s*np)*g_gbar_device[i]] - zb_xbb[k + (i + s*np)*g_gbar_device[i]];
				h_yff[k + (i + s*np)*g_gbar_device[i]] = z_yff[k + (i + s*np)*g_gbar_device[i]] - zb_yff[k + (i + s*np)*g_gbar_device[i]];
				h_ybb[k + (i + s*np)*g_gbar_device[i]] = z_ybb[k + (i + s*np)*g_gbar_device[i]] - zb_ybb[k + (i + s*np)*g_gbar_device[i]];
			
				cell_r = 0;	cell_l = 0;	cell_u = 0;	cell_d = 0;
				if (zb_xff[k + (i + s*np)*g_gbar_device[i]] >= g_gz_device[i]){
					cell_r = 1;
				}
				if (zb_xbb[k + (i + s*np)*g_gbar_device[i]] >= g_gz_device[i]){
					cell_l = 1;
				}

				if (zb_yff[k + (i + s*np)*g_gbar_device[i]] >= g_gz_device[i]){
					cell_u = 1;
				}
				if (zb_ybb[k + (i + s*np)*g_gbar_device[i]] >= g_gz_device[i]){
					cell_d = 1;
				}
				//----------------------------------------------------------------------------
				if (zb_xff[k + (i + s*np)*g_gbar_device[i]] >= g_gz_device[i]){
					obs_r = num_xff * dx;// -dx / 2;
					obsp_r = compute_force_right_wall(i, 1, s, k, obs_r, mu, g_gkn_device, g_gnu_device, g_grho_device, g_gkt_device, np, dt_g, g_gne_device, g_gbar_device, g_glay_device, g_gR_device,
						g_gx_device, g_gy_device, g_gvx_device, g_gvy_device, g_gax_device, g_gay_device, g_gang_device, g_gangv_device, g_ganga_device,
						g_gfx_device, g_gfy_device, g_gfz_device, g_gp_device, g_gt_device, neloc_x_device, neloc_y_device);

					cell_f_xP_device[num_xff + num_yc * N] = obsp_r;
				}
				else if (xff >= g_gz_device[i] && cell_r == 0 && (cell_u + cell_d) == 0){
					obs_r = num_xff * dx;// -dx / 2;
					obsp_r = compute_force_right_wall(i, 1, s, k, obs_r, mu, g_gkn_device, g_gnu_device, g_grho_device, g_gkt_device, np, dt_g, g_gne_device, g_gbar_device, g_glay_device, g_gR_device,
						g_gx_device, g_gy_device, g_gvx_device, g_gvy_device, g_gax_device, g_gay_device, g_gang_device, g_gangv_device, g_ganga_device,
						g_gfx_device, g_gfy_device, g_gfz_device, g_gp_device, g_gt_device, neloc_x_device, neloc_y_device);
					cell_f_xP_device[num_xff + num_yc * N] = obsp_r;
				}

				if (zb_xbb[k + (i + s*np)*g_gbar_device[i]] >= g_gz_device[i]){
					obs_l = num_xbb * dx + dx;// / 2;
					obsp_l = compute_force_left_wall(i, 1, s, k, obs_l, mu, g_gkn_device, g_gnu_device, g_grho_device, g_gkt_device, np, dt_g, g_gne_device, g_gbar_device, g_glay_device, g_gR_device,
						g_gx_device, g_gy_device, g_gvx_device, g_gvy_device, g_gax_device, g_gay_device, g_gang_device, g_gangv_device, g_ganga_device,
						g_gfx_device, g_gfy_device, g_gfz_device, g_gp_device, g_gt_device, neloc_x_device, neloc_y_device);
					cell_f_xN_device[num_xbb + num_yc * N] = obsp_l;
					}
				else if (xbb >= g_gz_device[i] && cell_l == 0 && (cell_u + cell_d) == 0){
					obs_l = num_xbb * dx + dx;// / 2;
					obsp_l = compute_force_left_wall(i, 1, s, k, obs_l, mu, g_gkn_device, g_gnu_device, g_grho_device, g_gkt_device, np, dt_g, g_gne_device, g_gbar_device, g_glay_device, g_gR_device,
						g_gx_device, g_gy_device, g_gvx_device, g_gvy_device, g_gax_device, g_gay_device, g_gang_device, g_gangv_device, g_ganga_device,
						g_gfx_device, g_gfy_device, g_gfz_device, g_gp_device, g_gt_device, neloc_x_device, neloc_y_device);
					cell_f_xN_device[num_xbb + num_yc * N] = obsp_l;
				}

				if (zb_yff[k + (i + s*np)*g_gbar_device[i]] >= g_gz_device[i]){
					obs_u = num_yff * dy; // - dy / 2;
				}
				else if (yff >= g_gz_device[i] && cell_u == 0 && (cell_l + cell_r) == 0){
					obs_u = num_yff * dy; // - dy / 2;
					obsp_u = compute_force_upper_wall(i, 1, s, k, obs_u, mu, g_gkn_device, g_gnu_device, g_grho_device, g_gkt_device, np, dt_g, g_gne_device, g_gbar_device, g_glay_device, g_gR_device,
						g_gx_device, g_gy_device, g_gvx_device, g_gvy_device, g_gax_device, g_gay_device, g_gang_device, g_gangv_device, g_ganga_device,
						g_gfx_device, g_gfy_device, g_gfz_device, g_gp_device, g_gt_device, neloc_x_device, neloc_y_device);
					cell_f_yP_device[num_xc + num_yff * N] = obsp_u;					
				}
				if (zb_ybb[k + (i + s*np)*g_gbar_device[i]] >= g_gz_device[i]){
					obs_d = num_ybb * dy + dy;// / 2;
					obsp_d = compute_force_lower_wall(i, 1, s, k, obs_d, mu, g_gkn_device, g_gnu_device, g_grho_device, g_gkt_device, np, dt_g, g_gne_device, g_gbar_device, g_glay_device, g_gR_device,
						g_gx_device, g_gy_device, g_gvx_device, g_gvy_device, g_gax_device, g_gay_device, g_gang_device, g_gangv_device, g_ganga_device,
						g_gfx_device, g_gfy_device, g_gfz_device, g_gp_device, g_gt_device, neloc_x_device, neloc_y_device);
					cell_f_yN_device[num_xc + num_ybb * N] = obsp_d;
				}
				else if (ybb >= g_gz_device[i] && cell_d == 0 && (cell_l + cell_r) == 0){
					obs_d = num_ybb * dy + dy;// / 2;
					obsp_d = compute_force_lower_wall(i, 1, s, k, obs_d, mu, g_gkn_device, g_gnu_device, g_grho_device, g_gkt_device, np, dt_g, g_gne_device, g_gbar_device, g_glay_device, g_gR_device,
						g_gx_device, g_gy_device, g_gvx_device, g_gvy_device, g_gax_device, g_gay_device, g_gang_device, g_gangv_device, g_ganga_device,
						g_gfx_device, g_gfy_device, g_gfz_device, g_gp_device, g_gt_device, neloc_x_device, neloc_y_device);
					cell_f_yN_device[num_xc + num_ybb * N] = obsp_d;
				}
			} //end of 1st loop of k++
		}// end of 1st loop of s++

		h_xbb_max = max_element(h_xbb, np, i, g_gne_device, g_gbar_device, h_xbb_max_ele);  //h_...系列的矩阵里，ele还是一一对应每个单元球体的编号，先暂时不改
		h_g_max = max_element(h_g, np, i, g_gne_device, g_gbar_device, h_g_max_ele);
		h_xff_max = max_element(h_xff, np, i, g_gne_device, g_gbar_device, h_xff_max_ele);
		h_ybb_max = max_element(h_ybb, np, i, g_gne_device, g_gbar_device, h_ybb_max_ele);
		h_yff_max = max_element(h_yff, np, i, g_gne_device, g_gbar_device, h_yff_max_ele);

		h_xb_max = max_element(h_xb, np, i, g_gne_device, g_gbar_device, h_xb_max_ele);
		h_xf_max = max_element(h_xf, np, i, g_gne_device, g_gbar_device, h_xf_max_ele);
		h_yb_max = max_element(h_yb, np, i, g_gne_device, g_gbar_device, h_yb_max_ele);
		h_yf_max = max_element(h_yf, np, i, g_gne_device, g_gbar_device, h_yf_max_ele);

		h_maxs = maximum(h_xbb_max, h_g_max, h_xff_max);
		h_maxs = maximum(h_maxs, h_ybb_max, h_yff_max);

		h_max = maximum(h_xb_max, h_g_max, h_xf_max);
		h_max = maximum(h_max, h_yb_max, h_yf_max);

		h_maxs = MAX(h_max, h_maxs);

		if (h_maxs > z_rela[i]){
			if (h_g_max == h_maxs){
				g_gz_device[i] = z_g[h_g_max_ele] + r_rela[i];// +lay_sub[i] * 2 * g_gR_device[i];
			}
			else if (h_xff_max == h_maxs){
				g_gz_device[i] = z_xff[h_xff_max_ele] + r_rela[i];// +lay_sub[i] * 2 * g_gR_device[i];

			}
			else if (h_ybb_max == h_maxs){
				g_gz_device[i] = z_ybb[h_ybb_max_ele] + r_rela[i];// +lay_sub[i] * 2 * g_gR_device[i];
			}
			else if (h_yff_max == h_maxs){
				g_gz_device[i] = z_yff[h_yff_max_ele] + r_rela[i];// +lay_sub[i] * 2 * g_gR_device[i];
			}
			else if (h_xbb_max == h_maxs) {  //h_xbb==h_maxs
				g_gz_device[i] = z_xbb[h_xbb_max_ele] + r_rela[i];// +lay_sub[i] * 2 * g_gR_device[i];
			}
			else if (h_xf_max == h_maxs){
				g_gz_device[i] = z_xf[h_xf_max_ele] + r_rela[i];// +lay_sub[i] * 2 * g_gR_device[i];

			}
			else if (h_yb_max == h_maxs){
				g_gz_device[i] = z_yb[h_yb_max_ele] + r_rela[i];// +lay_sub[i] * 2 * g_gR_device[i];
			}
			else if (h_yf_max == h_maxs){
				g_gz_device[i] = z_yf[h_yf_max_ele] + r_rela[i];// +lay_sub[i] * 2 * g_gR_device[i];
			}
			else {  //h_xb==h_maxs
				g_gz_device[i] = z_xb[h_xb_max_ele] + r_rela[i];// +lay_sub[i] * 2 * g_gR_device[i];
			}
		}

		else {  //h_mins=0
			g_gz_device[i] = z_g[(g_gbar_device[i] - 1) / 2 + (i + (g_gne_device[i] - 1) * np / 2) * g_gbar_device[i]] + g_glay_device[i] * g_gR_device[i];
		}
	}
}

__device__ int dem_bound(int xy, int bound)
{
	int xyb;
	if (xy <= 0){
		xyb = 0;
	}
	else if (xy >= (bound - 1)){
		xyb = bound - 1;
	}
	else{
		xyb = xy;
	}
	return xyb;

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
