#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define maximum(x,y,z) (((MAX (x,y)) > (z)) ? (MAX (x,y)) : (z))
#define minimum(x,y,z) (((MIN (x,y)) < (z)) ? (MIN (x,y)) : (z))

// index of machanical properties
const int i_kn = 0;    // Normal stiffness    法向刚度系数    - 塑料/车
const int i_nu = 1;    // Normal damping    法向粘性阻尼系数
const int i_kt = 2;    // Tangential stiffness    切向刚度系数

const double PI = 3.1415926;

__device__ double func(double x){
	return x*x;
}

__device__ double integral_b(double(*f)(double), double min, double max, double hb, double hf, double R){
	double result = 0;
	const int N = 1000;
	double delta = (max - min) / N;
	for (double i = min + delta; i < max; i = i + delta)
	{
		result = result + f(sqrt(pow(R, 2) - pow(i, 2)) + hb - (hb + hf) / 2)*delta;
	}
	return result;
}

__device__ double integral_f(double(*f)(double), double min, double max, double hb, double hf, double R){
	double result = 0;
	const int N = 1000;
	double delta = (max - min) / N;
	for (double i = min + delta; i < max; i = i + delta)
	{
		result = result + f(sqrt(pow(R, 2) - pow(i, 2)) - (hb - hf) / 2)*delta;
	}
	return result;
}

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
__device__ scalar_t max_element(scalar_t *a, int np, int i, int ne, int g_gbar_device, int &ele){
	scalar_t max;
	max = a[i*g_gbar_device];
	ele = i*g_gbar_device;

	for (int s = 0; s < ne; s++){
		for (int k = 0; k < g_gbar_device; k++){
			if (a[k + (i + np*s)*g_gbar_device]>max){
				max = a[k + (i + np*s)*g_gbar_device];
				ele = k + (i + np*s)*g_gbar_device;
			}

		}
	}
	return max;
}

template <typename scalar_t>
__device__ scalar_t min_element(scalar_t *a, int np, int i, int ne, int bar, int &ele){
	scalar_t min;
	min = a[i*bar];
	ele = i*bar;
	for (int s = 0; s < ne; s++){
		for (int k = 0; k < bar; k++){
			if (a[k + (i + np*s)*bar]<min){
				min = a[k + (i + np*s)*bar];
				ele = k + (i + np*s)*bar;

			}
		}
	}
	return min;

}

template <typename scalar_t>
__global__ void loc_debris_hydforce_kernel(const int M, const int N, const int np, const int ng,
											scalar_t *__restrict__ dx,
											scalar_t *__restrict__ dt_g, 
											int32_t *__restrict__ maskIndex,
											int32_t *__restrict__ num_x_device,
											int32_t *__restrict__ num_y_device,
											scalar_t *__restrict__ zb_device,
											scalar_t *__restrict__ h_device,
											scalar_t *__restrict__ z_device,
											scalar_t *__restrict__ qx_device,
											scalar_t *__restrict__ qy_device, 
											int32_t *__restrict__ g_gne_device,
											int32_t *__restrict__ g_gbar_device,
											int32_t *__restrict__ g_glay_device,
											scalar_t *__restrict__ g_gR_device, 
											scalar_t *__restrict__ g_gp_device,
											scalar_t *__restrict__ g_gt_device,
											scalar_t *__restrict__ g_gm_device,
											scalar_t *__restrict__ g_gz_device,
											scalar_t *__restrict__ g_gx_device,
											scalar_t *__restrict__ g_gy_device,
											scalar_t *__restrict__ neloc_x_device,
											scalar_t *__restrict__ neloc_y_device,
											scalar_t *__restrict__ mu,
											scalar_t *__restrict__ g_properties,
											scalar_t *__restrict__ g_grho_device,
											scalar_t *__restrict__ g_gvx_device,
											scalar_t *__restrict__ g_gvy_device,	
											scalar_t *__restrict__ g_gfx_device,
											scalar_t *__restrict__ g_gfy_device,
											scalar_t *__restrict__ cell_f_xP_device,
											scalar_t *__restrict__ cell_f_xN_device,
											scalar_t *__restrict__ cell_f_yP_device,
											scalar_t *__restrict__ cell_f_yN_device,
											scalar_t *__restrict__ tau_wgx_device, 
											scalar_t *__restrict__ tau_wgy_device, 
											scalar_t *__restrict__ g_gI_device,
											scalar_t *__restrict__ g_gax_device,
											scalar_t *__restrict__ g_gay_device,
											scalar_t *__restrict__ g_gang_device,
											scalar_t *__restrict__ g_gangv_device,
											scalar_t *__restrict__ g_ganga_device){

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i<np){
		// local variables
		// loc_debris
		scalar_t h_xbb_max, h_xff_max, h_ybb_max, h_yff_max, h_g_max, h_xb_max, h_xf_max, h_yb_max, h_yf_max;
	
		int32_t h_xbb_max_ele, h_xff_max_ele, h_ybb_max_ele, h_yff_max_ele, h_g_max_ele, h_xb_max_ele, h_xf_max_ele, h_yb_max_ele, h_yf_max_ele;
		
		int32_t num_xf, num_xb;
		int32_t num_yf, num_yb;
		int32_t num_xff, num_xbb, num_yff, num_ybb;
		int32_t num_xc, num_yc;
		
		scalar_t obs_r, obs_l, obs_u, obs_d;
		scalar_t obsp_r, obsp_l, obsp_u, obsp_d;

		scalar_t h_maxs;
		scalar_t h_max;
		scalar_t xff, xbb, yff, ybb;
		int32_t cell_r, cell_l, cell_u, cell_d;
		int32_t deb_xc, deb_yc;
		scalar_t _r_rela;
		scalar_t _z_rela;
		int32_t _lay_sub;

		// cal_hydforce
		const scalar_t tol_h = 1.0e-10;
		const scalar_t grav = 9.81;

		scalar_t u_g, v_g;
		//scalar_t g_sub, h_tri;
		scalar_t h_xbb_min, h_xff_min, h_ybb_min, h_yff_min, h_g_min, h_xb_min, h_xf_min, h_yb_min, h_yf_min;
		int h_xbb_min_ele, h_xff_min_ele, h_ybb_min_ele, h_yff_min_ele, h_g_min_ele, h_xb_min_ele, h_xf_min_ele, h_yb_min_ele, h_yf_min_ele;

		scalar_t h_min, h_mins;
		
		scalar_t t_sign;
		
		scalar_t uc, h_tem;

		scalar_t delta_zx_one, delta_zx_two, delta_zx_three, delta_zy_one, delta_zy_two, delta_zy_three;

		// global variables
		scalar_t _mu = mu[0];
		scalar_t kn = g_properties[i + i_kn*np];
		scalar_t nu = g_properties[i + i_nu*np];
		scalar_t kt = g_properties[i + i_kt*np]; 
		scalar_t gfx = g_gfx_device[i];
		scalar_t gfy = g_gfy_device[i];
		scalar_t gvx = g_gvx_device[i];
		scalar_t gvy = g_gvy_device[i];
		scalar_t _dx = dx[0];
		int32_t ne = g_gne_device[i];
		int32_t bar = g_gbar_device[i];
		int32_t lay = g_glay_device[i]; 

		scalar_t gR = g_gR_device[i];
		scalar_t gp = g_gp_device[i];
		scalar_t gt = g_gt_device[i];
		scalar_t gm = g_gm_device[i];

		// hydForce global variables
		scalar_t gang = g_gang_device[i];
		scalar_t grho = g_grho_device[i];

		// need to be updated after calculation
		scalar_t g_gz = g_gz_device[i];
        scalar_t _f_wgx;
        scalar_t _f_wgy;
        scalar_t _tau_wgx = tau_wgx_device[i];
        scalar_t _tau_wgy = tau_wgy_device[i];
        scalar_t _t_wg;

		// declaration
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		scalar_t zb_g;
		scalar_t *h_g = new scalar_t[ng];
		scalar_t *z_g = new scalar_t[ng];
		scalar_t *qx_g = new scalar_t[ng];
		scalar_t *qy_g = new scalar_t[ng];
		
		scalar_t zb_xf;
		scalar_t *z_xf = new scalar_t[ng];
		scalar_t *h_xf = new scalar_t[ng];

		scalar_t zb_yf; 
		scalar_t *z_yf = new scalar_t[ng];
		scalar_t *h_yf = new scalar_t[ng];


		scalar_t zb_xb; 
		scalar_t *z_xb = new scalar_t[ng];
		scalar_t *h_xb = new scalar_t[ng];

		scalar_t zb_yb; 
		scalar_t *z_yb = new scalar_t[ng];
		scalar_t *h_yb = new scalar_t[ng];

		scalar_t zb_xff; 
		scalar_t *z_xff = new scalar_t[ng];
		scalar_t *h_xff = new scalar_t[ng];

		scalar_t zb_yff; 
		scalar_t *z_yff = new scalar_t[ng];
		scalar_t *h_yff = new scalar_t[ng];
				
		scalar_t zb_xbb;
		scalar_t *z_xbb = new scalar_t[ng];
		scalar_t *h_xbb = new scalar_t[ng];

		scalar_t zb_ybb;
		scalar_t *z_ybb = new scalar_t[ng];
		scalar_t *h_ybb = new scalar_t[ng];

		int id_c;
		int id_xf;
		int id_yf;
		int id_xff;
		int id_yff;
		int id_xb;
		int id_yb;
		int id_xbb;
		int id_ybb;
		int id_ghost;
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		_z_rela = g_gm_device[i] / (1000 * gR * (bar + 1)*gR * (ne + 1));//rho * g * V
		_lay_sub = int(_z_rela / (2 * gR));
		_r_rela = (_lay_sub * 2 + 1)*gR - _z_rela;

		//correct gz
		//the location of the central of debris
		deb_xc = int(floor(g_gx_device[i] / _dx));
		deb_yc = int(floor(g_gy_device[i] / _dx));
		g_gz = z_device[deb_xc + deb_yc * N] + lay * gR;

		for (int s = 0; s < ne; s++){
			for (int k = 0; k < bar; k++){
// Par 1: 为了计算水流作用力，和判断漂浮物前后左右四个方向上的zb，所以用了比较多的定位关系。
				// calculate x,y cell number by x/yllcorner
				int pid = k + (i + s*np)*bar;

				num_x_device[pid] = int(floor((neloc_x_device[pid]) / _dx));  //i--N
				num_y_device[pid] = int(floor((neloc_y_device[pid]) / _dx));  //j--M
				
				// g
				num_xc = num_x_device[pid];
				num_yc = num_y_device[pid];
				id_c = maskIndex[num_xc + num_yc * N];

				zb_g = zb_device[id_c];
				z_g[pid] = z_device[id_c];
				qx_g[pid] = qx_device[id_c];
				qy_g[pid] = qy_device[id_c];
				h_g[pid] = h_device[id_c];

				// f
				num_xf = int32_t(ceil((neloc_x_device[pid] + gR) / _dx)) + 2;
				num_yf = int32_t(ceil((neloc_y_device[pid] + gR) / _dx)) + 2;

				id_xf = maskIndex[num_xf + num_yc * N];
				zb_xf = zb_device[id_xf];
				z_xf[pid] = z_device[id_xf];
				h_xf[pid] = h_device[id_xf];

				id_yf = maskIndex[num_xc + num_yf * N];
				zb_yf = zb_device[id_yf];
				z_yf[pid] = z_device[id_yf];
				h_yf[pid] = h_device[id_yf];
				

				// b
				num_xb = int32_t(ceil((neloc_x_device[pid] - gR) / _dx)) - 2;
				num_yb = int32_t(ceil((neloc_y_device[pid] - gR) / _dx)) - 2;

				id_xb = maskIndex[num_xb + num_yc * N];
				zb_xb = zb_device[id_xb];
				z_xb[pid] = z_device[id_xb];
				h_xb[pid] = h_device[id_xb];
				
				id_yb = maskIndex[num_xc + num_yb * N];
				zb_yb = zb_device[id_yb];
				z_yb[pid] = z_device[id_yb];		
				h_yb[pid] = h_device[id_yb];

				// ff
				num_xff = int32_t(floor((neloc_x_device[pid] + gR) / _dx));
				num_yff = int32_t(floor((neloc_y_device[pid] + gR) / _dx));

				id_xff = maskIndex[num_xff + num_yc * N];
				zb_xff = zb_device[id_xff];
				z_xff[pid] = z_device[id_xff];
				h_xff[pid] = h_device[id_xff];
				
				id_yff = maskIndex[num_xc + num_yff * N];
				zb_yff = zb_device[id_yff];
				z_yff[pid] = z_device[id_yff];
				h_yff[pid] = h_device[id_yff];

				// bb
				num_xbb = int32_t(floor((neloc_x_device[pid] - gR) / _dx));			
				num_ybb = int32_t(floor((neloc_y_device[pid] - gR) / _dx));

				id_xbb = maskIndex[num_xbb + num_yc * N];
				zb_xbb = zb_device[id_xbb];
				z_xbb[pid] = z_device[id_xbb];
				h_xbb[pid] = h_device[id_xbb];

				id_ybb = maskIndex[num_xc + num_ybb * N];
				zb_ybb = zb_device[id_ybb];
				z_ybb[pid] = z_device[id_ybb];
				h_ybb[pid] = h_device[id_ybb];

// Part 2: find the largest elevation for xff, yff, xbb, ybb
				// xff
				xff = zb_xff;
				for (int ghost = num_ybb; ghost <= num_yff; ghost++){
					id_ghost = maskIndex[num_xff + ghost*N];
					if (zb_device[id_ghost] > xff){
						xff = zb_device[id_ghost];
					}
				}
				
				// yff
				yff = zb_yff;
				for (int ghost = num_xbb; ghost <= num_xff; ghost++){
					id_ghost = maskIndex[ghost + num_yff*N];
					if (zb_device[id_ghost] > yff){
						yff = zb_device[id_ghost];
					}
				}
				
				// xbb
				xbb = zb_xbb;
				for (int ghost = num_ybb; ghost <= num_yff; ghost++){
					id_ghost = maskIndex[num_xbb + ghost*N];
					if (zb_device[id_ghost] > xbb){
						xbb = zb_device[id_ghost];
					}
				}
				
				// ybb
				ybb = zb_ybb;
				for (int ghost = num_xbb; ghost <= num_xff; ghost++){
					id_ghost = maskIndex[ghost + num_ybb*N];
					if (zb_device[id_ghost] > ybb){
						ybb = zb_device[id_ghost];
					}
				}

				// used for identify if the debris crushed into a wall
				cell_r = 0;	cell_l = 0;	cell_u = 0;	cell_d = 0;
				if (zb_xff >= g_gz){
					cell_r = 1;
				}
				if (zb_xbb >= g_gz){
					cell_l = 1;
				}

				if (zb_yff >= g_gz){
					cell_u = 1;
				}
				if (zb_ybb >= g_gz){
					cell_d = 1;
				}

// Part 3: Calculate the interact force between walls(calculated from part 2) and grains
				if (zb_xff >= g_gz){
					obs_r = num_xff * _dx;// -dx / 2;
					obsp_r = compute_force_right_wall(i, 1, s, k, obs_r, _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvx, gfx, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_xP_device[id_xff] = obsp_r;
				}
				else if (xff >= g_gz && cell_r == 0 && (cell_u + cell_d) == 0){
					obs_r = num_xff * _dx;// -dx / 2;
					obsp_r = compute_force_right_wall(i, 1, s, k, obs_r, _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvx, gfx, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_xP_device[id_xff] = obsp_r;
				}

				if (zb_xbb >= g_gz){
					obs_l = num_xbb * _dx + _dx;// / 2;
					obsp_l = compute_force_left_wall(i, 1, s, k, obs_l,  _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvx, gfx, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_xN_device[id_xbb] = obsp_l;
					}
				else if (xbb >= g_gz && cell_l == 0 && (cell_u + cell_d) == 0){
					obs_l = num_xbb * _dx + _dx;// / 2;
					obsp_l = compute_force_left_wall(i, 1, s, k, obs_l,  _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvx, gfx, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_xN_device[id_xbb] = obsp_l;
				}
				
				if (zb_yff >= g_gz){
					obs_u = num_yff * _dx; // - dy / 2;
					obsp_u = compute_force_upper_wall(i, 1, s, k, obs_u, _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvy, gfy, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_yP_device[id_yff] = obsp_u;
				}
				else if (yff >= g_gz && cell_u == 0 && (cell_l + cell_r) == 0){
					obs_u = num_yff * _dx; // - dy / 2;
					obsp_u = compute_force_upper_wall(i, 1, s, k, obs_u, _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvy, gfy, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_yP_device[id_yff] = obsp_u;					
				}
				
				if (zb_ybb >= g_gz){
					obs_d = num_ybb * _dx + _dx;// / 2;
					obsp_d = compute_force_lower_wall(i, 1, s, k, obs_d, _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvy, gfy, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_yN_device[id_ybb] = obsp_d;
				}
				else if (ybb >= g_gz && cell_d == 0 && (cell_l + cell_r) == 0){
					obs_d = num_ybb * _dx + _dx;// / 2;
					obsp_d = compute_force_lower_wall(i, 1, s, k, obs_d, _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvy, gfy, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_yN_device[id_ybb] = obsp_d;
				}	
			} //end of 1st loop of k++
		}// end of 1st loop of s++

// Part 4: find max h of all 9 cells, and calculate gz
		h_xbb_max = max_element(h_xbb, np, i, ne, bar, h_xbb_max_ele);  //h_...系列的矩阵里，ele还是一一对应每个单元球体的编号，先暂时不改
		h_g_max = max_element(h_g, np, i, ne, bar, h_g_max_ele);
		h_xff_max = max_element(h_xff, np, i, ne, bar, h_xff_max_ele);
		h_ybb_max = max_element(h_ybb, np, i, ne, bar, h_ybb_max_ele);
		h_yff_max = max_element(h_yff, np, i, ne, bar, h_yff_max_ele);

		h_xb_max = max_element(h_xb, np, i, ne, bar, h_xb_max_ele);
		h_xf_max = max_element(h_xf, np, i, ne, bar, h_xf_max_ele);
		h_yb_max = max_element(h_yb, np, i, ne, bar, h_yb_max_ele);
		h_yf_max = max_element(h_yf, np, i, ne, bar, h_yf_max_ele);

		h_maxs = maximum(h_xbb_max, h_g_max, h_xff_max);
		h_maxs = maximum(h_maxs, h_ybb_max, h_yff_max);

		h_max = maximum(h_xb_max, h_g_max, h_xf_max);
		h_max = maximum(h_max, h_yb_max, h_yf_max);

		h_maxs = MAX(h_max, h_maxs);

		if (h_maxs > _z_rela){
			if (h_g_max == h_maxs){
				g_gz = z_g[h_g_max_ele] + _r_rela;// +lay_sub[i] * 2 * g_gR_device[i];
			}
			else if (h_xff_max == h_maxs){
				g_gz = z_xff[h_xff_max_ele] + _r_rela;// +lay_sub[i] * 2 * g_gR_device[i];

			}
			else if (h_ybb_max == h_maxs){
				g_gz = z_ybb[h_ybb_max_ele] + _r_rela;// +lay_sub[i] * 2 * g_gR_device[i];
			}
			else if (h_yff_max == h_maxs){
				g_gz = z_yff[h_yff_max_ele] + _r_rela;// +lay_sub[i] * 2 * g_gR_device[i];
			}
			else if (h_xbb_max == h_maxs) {  //h_xbb==h_maxs
				g_gz = z_xbb[h_xbb_max_ele] + _r_rela;// +lay_sub[i] * 2 * g_gR_device[i];
			}
			else if (h_xf_max == h_maxs){
				g_gz = z_xf[h_xf_max_ele] + _r_rela;// +lay_sub[i] * 2 * g_gR_device[i];

			}
			else if (h_yb_max == h_maxs){
				g_gz = z_yb[h_yb_max_ele] + _r_rela;// +lay_sub[i] * 2 * g_gR_device[i];
			}
			else if (h_yf_max == h_maxs){
				g_gz = z_yf[h_yf_max_ele] + _r_rela;// +lay_sub[i] * 2 * g_gR_device[i];
			}
			else {  //h_xb==h_maxs
				g_gz = z_xb[h_xb_max_ele] + _r_rela;// +lay_sub[i] * 2 * g_gR_device[i];
			}
		}
		else {  //h_mins=0
			g_gz = z_g[(bar - 1) / 2 + (i + (ne - 1) * np / 2) * bar] + lay * gR;
		}
		
// --------------------hydro forces-----------------------------------------------
// Part 5: find min h of all 9 cells, and calculate u, v
		h_xbb_min = min_element(h_xbb, np, i, ne, bar, h_xbb_min_ele);
		h_g_min = min_element(h_g, np, i, ne, bar, h_g_min_ele);
		h_xff_min = min_element(h_xff, np, i, ne, bar, h_xff_min_ele);
		h_ybb_min = min_element(h_ybb, np, i, ne, bar, h_ybb_min_ele);
		h_yff_min = min_element(h_yff, np, i, ne, bar, h_yff_min_ele);

		h_xb_min = min_element(h_xb, np, i, ne, bar, h_xb_min_ele);
		h_xf_min = min_element(h_xf, np, i, ne, bar, h_xf_min_ele);
		h_yb_min = min_element(h_yb, np, i, ne, bar, h_yb_min_ele);
		h_yf_min = min_element(h_yf, np, i, ne, bar, h_yf_min_ele);

		h_mins = minimum(h_xbb_min, h_g_min, h_xff_min);
		h_mins = minimum(h_mins, h_ybb_min, h_yff_min);

		h_min = minimum(h_xb_min, h_g_min, h_xf_min);
		h_min = minimum(h_min, h_yb_min, h_yf_min);
		
		if (h_g_min > tol_h && h_min > tol_h){
			//wet cell only 		
			u_g = 1 * qx_g[int((bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar)]
				/ h_g[int((bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar)]; //1.1*
			v_g = 1 * qy_g[int((bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar)]
				/ h_g[int((bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar)]; //1.1*
			//cout << "u_g = " << u_g << endl;
		}
		else{
			u_g = 0;
			v_g = 0;
		}

// Part 6: Calculating the counter force between debris and fluid
// f_wgx, f_wgy: fluid -> debris
// tau_wgx, tau_wgy: debris -> fluid
// 3 conditions
// condition 1: h_min >= _z_rela
// condition 2: h_min > tol_h && h_min < _z_rela
// condition 3: else

		if (h_min >= _z_rela){
			
			//---------------------hydrodynamic ----------------------------------
			_f_wgx = 1000 * 1.21*(u_g - gvx)*fabs(u_g - gvx)
				//*((bar - 1)*fabs(sin(gang / 180 * PI)) / 2 + (ne - 1)*fabs(cos(gang / 180 * PI)) / 2)
				*(3.14*pow(gR, 2) * (asin((sqrt(pow(gR, 2) - pow(_r_rela, 2))) / gR)) / 3.1415926 - _r_rela*sqrt(pow(gR, 2) - pow(_r_rela, 2))
				+ 3.14*pow(gR, 2)*_lay_sub);
			//  扇形的面积 -------------------------------------------------------------------------减去-------三角形的面积----------------------------------

			_tau_wgx = 1000 * 1.21*(u_g - gvx)*fabs(u_g - gvx);//*bar  *ne;

			_f_wgy = 1000 * 1.21*(v_g - gvy)*fabs(v_g - gvy)
				//*((bar - 1)*fabs(cos(gang / 180 * PI)) / 2 + (ne - 1)*fabs(sin(gang / 180 * PI)) / 2)
				*(3.14*pow(gR, 2) * (asin((sqrt(pow(gR, 2) - pow(_r_rela, 2))) / gR)) / 3.1415926 - _r_rela*sqrt(pow(gR, 2) - pow(_r_rela, 2))
				+ 3.14*pow(gR, 2)*_lay_sub);


			_tau_wgy = 1000 * 1.21* (v_g - gvy)*fabs(v_g - gvy);// *bar  *ne;
			
			//---------------------hydrostatic --main force---whole side
			//  s  ------>  k+s*bar  so what's the value of 'k'?  the middle one  (bar  -1)/2
			// i+s*np ------>  k + (i + s*np)*bar  

			if (sin(gang / 180 * PI)*cos(gang / 180 * PI) == 0){
				if (sin(gang / 180 * PI) == 0 && cos(gang / 180 * PI) == 1){  //  ---@  0度 水平 0号在左
					
					delta_zx_one = z_xb[(bar - 1) / 2 + i * bar] - z_xf[(bar - 1) / 2 + (i + np * (ne - 1)) * bar];
					delta_zy_one = z_yb[0 + (i + np * (ne - 1) / 2) * bar] - z_yf[(bar - 1) + (i + np * (ne - 1) / 2) * bar];

					_f_wgx = _f_wgx + 0.5 * 1000 * grav*delta_zx_one
						*3.14*pow(gR, 2)*bar * lay;
					_f_wgy = _f_wgy + 0.5 * 1000 * grav*delta_zy_one
						*3.14*pow(gR, 2)*ne*lay;

					delta_zy_one = z_yb[i * bar] - z_yf[(bar - 1) + i * bar];
					delta_zy_two = z_yb[0 + (i + np * (ne - 1)) * bar] - z_yf[(bar - 1) + (i + np * (ne - 1)) * bar];
					delta_zy_three = z_yb[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar] - z_yf[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar];
					
					delta_zx_one = z_xb[i * bar] - z_xf[0 + (i + np * (ne - 1)) * bar];
					delta_zx_two = z_xb[(bar - 1) + i * bar] - z_xf[(bar - 1) + (i + np * (ne - 1)) * bar];

					//逆时针为+
					_t_wg = (0.5 * 1000 *lay* grav*(-1 * delta_zy_one
						+ delta_zy_two) *3.14*pow(gR, 2)
						*((ne + 1)*gR) / 2
						+ 0.5 * 1000 * grav* delta_zy_three
						*3.14*pow(gR, 2) *0.5*gR * (0.5*0.5*(ne + 1) + (ne * ne - 1.0) / 8)
						+ 0.5 * 1000 * grav*(delta_zx_one
						- delta_zx_two) *3.14*pow(gR, 2)
						*((bar + 1)*gR) / 2)*lay;
				}
				else if (sin(gang / 180 * PI) == 1 && cos(gang / 180 * PI) == 0){  //90度 竖直 0号在下

					delta_zx_one = z_xb[(bar - 1) + (i + np * (ne - 1) / 2) * bar] - z_xf[0 + (i + np * (ne - 1) / 2) * bar];
					delta_zy_one = z_yb[(bar - 1) / 2 + i * bar] - z_yf[(bar - 1) / 2 + (i + np * (ne - 1)) * bar];

					_f_wgx = _f_wgx + 0.5 * 1000 * grav*delta_zx_one
						*3.14*pow(gR, 2)*ne*lay;
					_f_wgy = _f_wgy + 0.5 * 1000 * grav*delta_zy_one
						*3.14*pow(gR, 2)*bar  *lay;

					delta_zx_one = z_xb[(bar - 1) + i * bar] - z_xf[i * bar];
					delta_zx_two = z_xb[(bar - 1) + (i + np * (ne - 1)) * bar] - z_xf[0 + (i + np * (ne - 1)) * bar];
					delta_zx_three = z_xb[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar] - z_xf[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar];
					delta_zy_one = z_yb[i * bar] - z_yf[0 + (i + np * (ne - 1)) * bar];
					delta_zy_two = z_yb[(bar - 1) + i * bar] - z_yf[(bar - 1) + (i + np * (ne - 1)) * bar];

					_t_wg = (0.5 * 1000 * grav*(delta_zx_one
						- delta_zx_two) *3.14*pow(gR, 2)
						*((ne + 1)*gR) / 2
						+ 0.5 * 1000 * grav* delta_zx_three
						*3.14*pow(gR, 2) *0.5*gR * (0.5*0.5*(ne + 1) + (ne * ne - 1.0) / 8)
						+ 0.5 * 1000 * grav*(delta_zy_one
						- delta_zy_two) *3.14*pow(gR, 2)
						*((bar + 1)*gR) / 2)*lay;

				}
				else if (sin(gang / 180 * PI) == 0 && cos(gang / 180 * PI) == -1) {  //  @---  180度 水平 0号在右
					delta_zx_one = z_xb[(bar - 1) / 2 + (i + np * (ne - 1)) * bar] - z_xf[(bar - 1) / 2 + i * bar];
					delta_zy_one = z_yb[(bar - 1) + (i + np * (ne - 1) / 2) * bar] - z_yf[0 + (i + np * (ne - 1) / 2) * bar];
					
					_f_wgx = _f_wgx + 0.5 * 1000 * grav*delta_zx_one
						*3.14*pow(gR, 2)*bar  *lay;
					_f_wgy = _f_wgy + 0.5 * 1000 * grav*delta_zy_one
						*3.14*pow(gR, 2)*ne*lay;

					delta_zy_one = z_yb[(bar - 1) + i * bar] - z_yf[i * bar];
					delta_zy_two = z_yb[(bar - 1) + (i + np * (ne - 1)) * bar] - z_yf[0 + (i + np * (ne - 1)) * bar];
					delta_zy_three = z_yb[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar] - z_yf[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar];

					delta_zx_one = z_xb[0 + (i + np * (ne - 1)) * bar] - z_xf[i * bar];
					delta_zx_two = z_xb[(bar - 1) + (i + np * (ne - 1)) * bar] - z_xf[(bar - 1) + i * bar];

					_t_wg = (0.5 * 1000 * grav*(delta_zy_one
						- delta_zy_two) *3.14*pow(gR, 2)
						*((ne + 1)*gR) / 2
						+ 0.5 * 1000 * grav* delta_zy_three
						*3.14*pow(gR, 2) *0.5*gR * (0.5*0.5*(ne + 1) + (ne * ne - 1.0) / 8)
						+ 0.5 * 1000 * grav*(-1 * delta_zx_one
						+ delta_zx_two) *3.14*pow(gR, 2)
						*((bar + 1)*gR) / 2)*lay;

				}
				else {  //if (sin(gang / 180 * PI) == 1 && cos(gang / 180 * PI) == 0){ //270度 竖直 0号在上
				    
				    delta_zx_one = z_xb[0 + (i + np * (ne - 1) / 2) * bar] - z_xf[(bar - 1) + (i + np * (ne - 1) / 2) * bar];
					delta_zy_one = z_yb[(bar - 1) / 2 + (i + np * (ne - 1)) * bar] - z_yf[(bar - 1) / 2 + i * bar];

					_f_wgx = _f_wgx + 0.5 * 1000 * grav*delta_zx_one
						*3.14*pow(gR, 2)*ne*lay;
					_f_wgy = _f_wgy + 0.5 * 1000 * grav*delta_zy_one
						*3.14*pow(gR, 2)*bar  *lay;

					delta_zx_one = z_xb[i * bar] - z_xf[(bar - 1) + i * bar];
					delta_zx_two = z_xb[0 + (i + np * (ne - 1)) * bar] - z_xf[(bar - 1) + (i + np * (ne - 1)) * bar];
					delta_zx_three = z_xb[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar] - z_xf[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar];
					delta_zy_one = z_yb[0 + (i + np * (ne - 1)) * bar] - z_yf[i * bar];
					delta_zy_two = z_yb[(bar - 1) + (i + np * (ne - 1)) * bar] - z_yf[(bar - 1) + i * bar];

					_t_wg =( 0.5 * 1000 * grav*(-1 * delta_zx_one
						+ delta_zx_two) *3.14*pow(gR, 2)
						*((ne + 1)*gR) / 2
						+ 0.5 * 1000 * grav* delta_zx_three
						*3.14*pow(gR, 2) *0.5*gR * (0.5*0.5*(ne + 1) + (ne * ne - 1.0) / 8)
						+ 0.5 * 1000 * grav*(-1 * delta_zy_one
						+ delta_zy_two) *3.14*pow(gR, 2)
						*((bar + 1)*gR) / 2)*lay;
					//+ 0.5 * 1000 * grav*(z_yb[(bar - 1) / 2 + (ne - 1)* bar / 2] - z_yf[(bar - 1) / 2 + (ne - 1)* bar / 2])
					//*3.14*pow(gR, 2) *0.5*gR*(0.5*0.5*(bar + 1) + (pow(bar  , 2) - 1) / 8);
				}

			}  //horizantal or vertical debris end

			//斜着运动的debris
			else {
				//  s  ------>  k+s*bar  so what's the value of 'k'?  the middle one  (bar  -1)/2
				// i+s*np ------>  k + (i + s*np)*bar .  the middle one (bar  -1)/2+(i+(ne-1)*np/2)*bar  

				if (sin(gang / 180 * PI) > 0 && cos(gang / 180 * PI) > 0){ //0-90
					_f_wgx = _f_wgx + 0.5 * 1000 * grav*(z_xb[(bar - 1) + i * bar] - z_xf[0 + (i + np*(ne - 1))*bar])*3.14*pow(gR, 2)*lay;
					_f_wgy = _f_wgy + 0.5 * 1000 * grav*(z_yb[i*bar] - z_yf[(bar - 1) + (i + np*(ne - 1))*bar])*3.14*pow(gR, 2)*lay;

					_t_wg = (1000 * grav*(-1) * 2 * integral_b(func, 0, gR, z_xb[(bar - 1) + i * bar], z_xf[0 + (i + np*(ne - 1))*bar], gR)
						*(neloc_y_device[bar - 1 + i*bar] - neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar])
						- 1000 * grav * 2 * integral_f(func, 0, gR, z_xb[(bar - 1) + i * bar], z_xf[0 + (i + np*(ne - 1))*bar], gR)
						*(neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_y_device[0 + (i + (ne - 1)*np)*bar]))*lay;  //ok
					_t_wg = _t_wg + (1000 * grav*(-1) * 2 * integral_b(func, 0, gR, z_yb[i*bar], z_yf[(bar - 1) + (i + np*(ne - 1)) * bar], gR)
						*(neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_x_device[i*bar])
						- 1000 * grav * 2 * integral_f(func, 0, gR, z_yb[i*bar], z_yf[(bar - 1) + (i + np*(ne - 1)) * bar], gR)
						*(neloc_x_device[(bar - 1) + (i + (ne - 1)*np)*bar] - neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar]))*lay;  //ok
				}
				else if (sin(gang / 180 * PI) > 0 && cos(gang / 180 * PI) < 0){  //90-180
					_f_wgx = _f_wgx + 0.5 * 1000 * grav*(z_xb[i*bar] - z_xf[(bar - 1) + (i + np*(ne - 1))*bar])*3.14*pow(gR, 2)*lay;
					_f_wgy = _f_wgy + 0.5 * 1000 * grav*(z_yb[(bar - 1) + i * bar] - z_yf[0 + (i + np*(ne - 1))*bar])*3.14*pow(gR, 2)*lay;

					_t_wg = (1000 * grav*(-1) * 2 * integral_b(func, 0, gR, z_xb[(bar - 1) + (i + np*(ne - 1)) * bar], z_xf[i*bar], gR)
						*(neloc_y_device[(bar - 1) + (i + (ne - 1)*np)*bar] - neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar])
						- 1000 * grav * 2 * integral_f(func, 0, gR, z_xb[(bar - 1) + (i + np*(ne - 1)) * bar], z_xf[i*bar], gR)
						*(neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_y_device[i*bar]))*lay;  //ok
					_t_wg = _t_wg + (1000 * grav * 2 * integral_b(func, 0, gR, z_yb[(bar - 1) + i * bar], z_yf[0 + (i + np*(ne - 1))*bar], gR)
						*(neloc_x_device[bar - 1 + i*bar] - neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar])
						+ 1000 * grav * 2 * integral_f(func, 0, gR, z_yb[(bar - 1) + i * bar], z_yf[0 + (i + np*(ne - 1))*bar], gR)
						*(neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_x_device[0 + (i + (ne - 1)*np)*bar]))*lay;  //ok


				}
				else if (sin(gang / 180 * PI) < 0 && cos(gang / 180 * PI) < 0) {  //180-270
					_f_wgx = _f_wgx + 0.5 * 1000 * grav*(z_xb[0 + (i + np*(ne - 1))*bar] - z_xf[(bar - 1) + i * bar])*3.14*pow(gR, 2)*lay;
					_f_wgy = _f_wgy + 0.5 * 1000 * grav*(z_yb[(bar - 1) + (i + np*(ne - 1))*bar] - z_yf[i*bar])*3.14*pow(gR, 2)*lay;

					_t_wg = (1000 * grav*(-1) * 2 * integral_b(func, 0, gR, z_xb[0 + (i + np*(ne - 1))*bar], z_xf[(bar - 1) + i * bar], gR)
						*(neloc_y_device[0 + (i + (ne - 1)*np)*bar] - neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar])
						- 1000 * grav * 2 * integral_f(func, 0, gR, z_xb[0 + (i + np*(ne - 1))*bar], z_xf[(bar - 1) + i * bar], gR)
						*(neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_y_device[bar - 1 + i*bar]))*lay;  //ok

					_t_wg = _t_wg + (1000 * grav*(-1) * 2 * integral_b(func, 0, gR, z_yb[(bar - 1) + (i + np*(ne - 1)) * bar], z_yf[i*bar], gR)
						*(neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_x_device[(bar - 1) + (i + (ne - 1)*np)*bar])
						- 1000 * grav * 2 * integral_f(func, 0, gR, z_yb[i*bar], z_yf[(bar - 1) + (i + np*(ne - 1)) * bar], gR)
						*(neloc_x_device[i*bar] - neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar]))*lay;  //ok
				}

				else { //if (sin(gang / 180 * PI) < 0 && cos(gang / 180 * PI) > 0) { //270-360
					_f_wgx = _f_wgx + 0.5 * 1000 * grav*(z_xb[(bar - 1) + (i + np*(ne - 1))*bar] - z_xf[i*bar])*3.14*pow(gR, 2)*lay;
					_f_wgy = _f_wgy + 0.5 * 1000 * grav*(z_yb[0 + (i + np*(ne - 1))*bar] - z_yf[(bar - 1) + i * bar])*3.14*pow(gR, 2)*lay;


					_t_wg = (1000 * grav*(-1) * 2 * integral_b(func, 0, gR, z_xb[i*bar], z_xf[(bar - 1) + (i + np*(ne - 1)) * bar], gR)
						*(neloc_y_device[i*bar] - neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar])
						- 1000 * grav * 2 * integral_f(func, 0, gR, z_xb[i*bar], z_xf[(bar - 1) + (i + np*(ne - 1)) * bar], gR)
						*(neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_y_device[(bar - 1) + (i + (ne - 1)*np)*bar]))*lay; //ok

					_t_wg = _t_wg + (1000 * grav * 2 * integral_b(func, 0, gR, z_yb[0 + (i + np*(ne - 1))*bar], z_yf[(bar - 1) + i * bar], gR)
						*(neloc_x_device[0 + (i + (ne - 1)*np)*bar] - neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar])
						+ 1000 * grav * 2 * integral_f(func, 0, gR, z_yb[(bar - 1) + i * bar], z_yf[0 + (i + np*(ne - 1))*bar], gR)
						*(neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_x_device[bar - 1 + i*bar]))*lay;  //ok

				}

				//--------------part hydrostatic forces when ang 
				
				//---------------------hydrodynamic---other force--part sidel
				
				_f_wgx = _f_wgx + 0.5 * 1000 * grav*(z_xb[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar] - z_xf[(bar - 1) / 2 + (i + np*(ne - 1)/2) *bar])
					*3.14*pow(gR, 2) *(fabs(sin(gang / 180 * PI) / 2)*(ne - 1) + fabs(cos(gang / 180 * PI) / 2)
					*(bar - 1)
					)*lay;
				_f_wgy = _f_wgy + 0.5 * 1000 * grav*(z_yb[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar] - z_yf[(bar - 1) / 2 + (i + np*(ne - 1)/2) *bar])
					*3.14*pow(gR, 2) * (fabs(cos(gang / 180 * PI) / 2)*(ne - 1) + fabs(sin(gang / 180 * PI) / 2)
					*(bar - 1)
					)*lay;
				
				//--------------torque---------------------------
				t_sign = fabs(cos(gang / 180 * PI)*sin(gang / 180 * PI)) / (cos(gang / 180 * PI)*sin(gang / 180 * PI));
					
				_t_wg = _t_wg - t_sign*lay * (0.5 * 1000 * grav*(z_xb[(bar - 1) / 2 + (i + np*(ne - 1) / 2)*bar] - z_xf[(bar - 1) / 2 + (i + np*(ne - 1)/2)*bar])
					*3.14*pow(gR, 2)*fabs(sin(gang / 180 * PI) / 2)*gR*(
					(1 - 0.5*fabs(sin(gang / 180 * PI)))*0.5*(ne + 1) + fabs(sin(gang / 180 * PI))*((ne * ne) - 1) / 8)
					+ 0.5 * 1000 * grav*(z_yb[(bar - 1) / 2 + (i + np*(ne - 1) / 2)*bar] - z_yf[(bar - 1) / 2 + (i + np*(ne - 1) / 2)*bar])
					*3.14*pow(gR, 2) * fabs(cos(gang / 180 * PI) / 2)*gR*(
					(1 - 0.5*fabs(cos(gang / 180 * PI)))*0.5*(ne + 1) + fabs(cos(gang / 180 * PI))*((ne * ne) - 1) / 8)
					+ 0.5 * 1000 * grav*(z_xb[(bar - 1) / 2 + (i + np*(ne - 1) / 2)*bar] - z_xf[(bar - 1) / 2 + (i + np*(ne - 1) / 2)*bar])
					*3.14*pow(gR, 2) * fabs(cos(gang / 180 * PI) / 2)*gR*(
					(1 - 0.5*fabs(cos(gang / 180 * PI)))*0.5*(bar + 1) + fabs(cos(gang / 180 * PI))*((bar * bar) - 1) / 8)
					+ 0.5 * 1000 * grav*(z_yb[(bar - 1) / 2 + (i + np*(ne - 1) / 2)*bar] - z_yf[(bar - 1) / 2 + (i + np*(ne - 1) / 2)*bar])
					*3.14*pow(gR, 2) * fabs(sin(gang / 180 * PI) / 2)*gR*(
					(1 - 0.5*fabs(sin(gang / 180 * PI)))*0.5*(bar + 1) + fabs(sin(gang / 180 * PI))*((bar * bar) - 1) / 8)
					);
			}		
		} //end of  h_min>z_rela

		//-----------------------end of  h_min>z_rela------------------------------------------------------------------------------------------------------------------------------

		//else if (h_min > tol_h && h_min <= _z_rela && (gvx != 0 || gvy != 0)){
		
		else if (h_min > tol_h && h_min < _z_rela){

			uc = 0.5*pow(h_min / (2 * gR), -0.1)*sqrt(2 * grav *gR*(ne + 1)*
				(grho * 2 * gR / (1000 * h_min) - (grho * 2 * gR / (1000 * _z_rela))));
			//cout << "uc = " << uc << endl;


			if (gvx != 0 || gvy != 0){
				_f_wgx = -(1 - h_min / _z_rela)*_mu*gm*grav*gvx / (sqrt(pow(gvx, 2) + pow(gvy, 2)));
				_f_wgy = -(1 - h_min / _z_rela)*_mu*gm*grav*gvy / (sqrt(pow(gvx, 2) + pow(gvy, 2)));
				_tau_wgx = 0;
				_tau_wgy = 0;
				_t_wg = 0.0;
			}

			
			else if (u_g > uc){   //  && ((gvx < emsmall || gvy <emsmall))){ //pow(u_g, 2) + pow(v_g, 2) >= pow(uc, 2)
				h_tem = gR - h_min;
				
				_f_wgx = 1000 * 1.21*(u_g - gvx)*fabs(u_g - gvx)
					//*((bar - 1)*fabs(sin(gang / 180 * PI)) / 2 + (ne - 1)*fabs(cos(gang / 180 * PI)) / 2)
					*(3.14*pow(gR, 2) * (asin((sqrt(pow(gR, 2) - pow(h_tem, 2))) / gR)) / 3.1415926 - h_tem*sqrt(pow(gR, 2) - pow(h_tem, 2)))*lay;


				_tau_wgx = 1000 * 1.21*(u_g - gvx)*fabs(u_g - gvx);// *bar  *ne;

				_f_wgy = 1000 * 1.21*(v_g - gvy)*fabs(v_g - gvy)
					//*((bar - 1)*fabs(cos(gang / 180 * PI)) / 2 + (ne - 1)*fabs(sin(gang / 180 * PI)) / 2)
					*(3.14*pow(gR, 2) * (asin((sqrt(pow(gR, 2) - pow(h_tem, 2))) / gR)) / 3.1415926 - h_tem*sqrt(pow(gR, 2) - pow(h_tem, 2)))*lay;

				_tau_wgy = 1000 * 1.21* (v_g - gvy)*fabs(v_g - gvy);// *bar  *ne;

				//---------------------hydrostatic --main force---whole side
				//  s  ------>  k+s*bar  so what's the value of 'k'?  the middle one  (bar  -1)/2
				// i+s*np ------>  k + (i + s*np)*bar  

				if (sin(gang / 180 * PI)*cos(gang / 180 * PI) == 0){
					if (sin(gang / 180 * PI) == 0 && cos(gang / 180 * PI) == 1){  //  ---@  0度 水平 0号在左
						
						
						delta_zx_one = z_xb[(bar - 1) / 2 + i * bar] - z_xf[(bar - 1) / 2 + (i + np * (ne - 1)) * bar];
						delta_zy_one = z_yb[0 + (i + np * (ne - 1) / 2) * bar] - z_yf[(bar - 1) + (i + np * (ne - 1) / 2) * bar];

						_f_wgx = _f_wgx + 0.5 * 1000 * grav * delta_zx_one
							* 3.14 * pow(gR, 2) * bar * lay;
						_f_wgy = _f_wgy + 0.5 * 1000 * grav * delta_zy_one
							* 3.14 * pow(gR, 2) * ne * lay;

						delta_zy_one = z_yb[i * bar] - z_yf[(bar - 1) + i * bar];
						delta_zy_two = z_yb[0 + (i + np * (ne - 1)) * bar] - z_yf[(bar - 1) + (i + np * (ne - 1)) * bar];
						delta_zy_three = z_yb[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar] - z_yf[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar];

						delta_zx_one = z_xb[i * bar] - z_xf[0 + (i + np * (ne - 1)) * bar];
						delta_zx_two = z_xb[(bar - 1) + i * bar] - z_xf[(bar - 1) + (i + np * (ne - 1)) * bar];

						//逆时针为+
						_t_wg = (0.5 * 1000 * lay * grav * (-1 * delta_zy_one
							+ delta_zy_two) * 3.14 * pow(gR, 2)
							* ((ne + 1) * gR) / 2
							+ 0.5 * 1000 * grav * delta_zy_three
							* 3.14 * pow(gR, 2) * 0.5 * gR * (0.5 * 0.5 * (ne + 1) + (ne * ne - 1.0) / 8)
							+ 0.5 * 1000 * grav * (delta_zx_one
								- delta_zx_two) * 3.14 * pow(gR, 2)
							* ((bar + 1) * gR) / 2) * lay;

					}
					else if (sin(gang / 180 * PI) == 1 && cos(gang / 180 * PI) == 0){  //90度 竖直 0号在下
						delta_zx_one = z_xb[(bar - 1) + (i + np * (ne - 1) / 2) * bar] - z_xf[0 + (i + np * (ne - 1) / 2) * bar];
						delta_zy_one = z_yb[(bar - 1) / 2 + i * bar] - z_yf[(bar - 1) / 2 + (i + np * (ne - 1)) * bar];

						_f_wgx = _f_wgx + 0.5 * 1000 * grav * delta_zx_one
							* 3.14 * pow(gR, 2) * ne * lay;
						_f_wgy = _f_wgy + 0.5 * 1000 * grav * delta_zy_one
							* 3.14 * pow(gR, 2) * bar * lay;

						delta_zx_one = z_xb[(bar - 1) + i * bar] - z_xf[i * bar];
						delta_zx_two = z_xb[(bar - 1) + (i + np * (ne - 1)) * bar] - z_xf[0 + (i + np * (ne - 1)) * bar];
						delta_zx_three = z_xb[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar] - z_xf[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar];
						delta_zy_one = z_yb[i * bar] - z_yf[0 + (i + np * (ne - 1)) * bar];
						delta_zy_two = z_yb[(bar - 1) + i * bar] - z_yf[(bar - 1) + (i + np * (ne - 1)) * bar];

						_t_wg = (0.5 * 1000 * grav * (delta_zx_one
							- delta_zx_two) * 3.14 * pow(gR, 2)
							* ((ne + 1) * gR) / 2
							+ 0.5 * 1000 * grav * delta_zx_three
							* 3.14 * pow(gR, 2) * 0.5 * gR * (0.5 * 0.5 * (ne + 1) + (ne * ne - 1.0) / 8)
							+ 0.5 * 1000 * grav * (delta_zy_one
								- delta_zy_two) * 3.14 * pow(gR, 2)
							* ((bar + 1) * gR) / 2) * lay;

					}
					else if (sin(gang / 180 * PI) == 0 && cos(gang / 180 * PI) == -1) {  //  @---  180度 水平 0号在右
						delta_zx_one = z_xb[(bar - 1) / 2 + (i + np * (ne - 1)) * bar] - z_xf[(bar - 1) / 2 + i * bar];
						delta_zy_one = z_yb[(bar - 1) + (i + np * (ne - 1) / 2) * bar] - z_yf[0 + (i + np * (ne - 1) / 2) * bar];


						_f_wgx = _f_wgx + 0.5 * 1000 * grav * delta_zx_one
							* 3.14 * pow(gR, 2) * bar * lay;
						_f_wgy = _f_wgy + 0.5 * 1000 * grav * delta_zy_one
							* 3.14 * pow(gR, 2) * ne * lay;

						delta_zy_one = z_yb[(bar - 1) + i * bar] - z_yf[i * bar];
						delta_zy_two = z_yb[(bar - 1) + (i + np * (ne - 1)) * bar] - z_yf[0 + (i + np * (ne - 1)) * bar];
						delta_zy_three = z_yb[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar] - z_yf[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar];

						delta_zx_one = z_xb[0 + (i + np * (ne - 1)) * bar] - z_xf[i * bar];
						delta_zx_two = z_xb[(bar - 1) + (i + np * (ne - 1)) * bar] - z_xf[(bar - 1) + i * bar];

						_t_wg = (0.5 * 1000 * grav * (delta_zy_one
							- delta_zy_two) * 3.14 * pow(gR, 2)
							* ((ne + 1) * gR) / 2
							+ 0.5 * 1000 * grav * delta_zy_three
							* 3.14 * pow(gR, 2) * 0.5 * gR * (0.5 * 0.5 * (ne + 1) + (ne * ne - 1.0) / 8)
							+ 0.5 * 1000 * grav * (-1 * delta_zx_one
								+ delta_zx_two) * 3.14 * pow(gR, 2)
							* ((bar + 1) * gR) / 2) * lay;

					}
					else {  //if (sin(gang / 180 * PI) == 1 && cos(gang / 180 * PI) == 0){ //270度 竖直 0号在上
					delta_zx_one = z_xb[0 + (i + np * (ne - 1) / 2) * bar] - z_xf[(bar - 1) + (i + np * (ne - 1) / 2) * bar];
					delta_zy_one = z_yb[(bar - 1) / 2 + (i + np * (ne - 1)) * bar] - z_yf[(bar - 1) / 2 + i * bar];


					_f_wgx = _f_wgx + 0.5 * 1000 * grav * delta_zx_one
						* 3.14 * pow(gR, 2) * ne * lay;
					_f_wgy = _f_wgy + 0.5 * 1000 * grav * delta_zy_one
						* 3.14 * pow(gR, 2) * bar * lay;

					delta_zx_one = z_xb[i * bar] - z_xf[(bar - 1) + i * bar];
					delta_zx_two = z_xb[0 + (i + np * (ne - 1)) * bar] - z_xf[(bar - 1) + (i + np * (ne - 1)) * bar];
					delta_zx_three = z_xb[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar] - z_xf[(bar - 1) / 2 + (i + np * (ne - 1) / 2) * bar];
					delta_zy_one = z_yb[0 + (i + np * (ne - 1)) * bar] - z_yf[i * bar];
					delta_zy_two = z_yb[(bar - 1) + (i + np * (ne - 1)) * bar] - z_yf[(bar - 1) + i * bar];

					_t_wg = (0.5 * 1000 * grav * (-1 * delta_zx_one
						+ delta_zx_two) * 3.14 * pow(gR, 2)
						* ((ne + 1) * gR) / 2
						+ 0.5 * 1000 * grav * delta_zx_three
						* 3.14 * pow(gR, 2) * 0.5 * gR * (0.5 * 0.5 * (ne + 1) + (ne * ne - 1.0) / 8)
						+ 0.5 * 1000 * grav * (-1 * delta_zy_one
							+ delta_zy_two) * 3.14 * pow(gR, 2)
						* ((bar + 1) * gR) / 2) * lay;
					//+ 0.5 * 1000 * grav*(z_yb[(bar - 1) / 2 + (ne - 1)* bar / 2] - z_yf[(bar - 1) / 2 + (ne - 1)* bar / 2])
					//*3.14*pow(gR, 2) *0.5*gR*(0.5*0.5*(bar + 1) + (pow(bar  , 2) - 1) / 8);
					}
				}
				else {
					//  s  ------>  k+s*bar  so what's the value of 'k'?  the middle one  (bar  -1)/2
					// i+s*np ------>  k + (i + s*np)*bar .  the middle one (bar  -1)/2+(i+(ne-1)*np/2)*bar  

					if (sin(gang / 180 * PI) > 0 && cos(gang / 180 * PI) > 0){ //0-90
						_f_wgx = _f_wgx + 0.5 * 1000 * grav*(z_xb[(bar - 1) + i * bar] - z_xf[0 + (i + np*(ne - 1))*bar])*3.14*pow(gR, 2)*lay;
						_f_wgy = _f_wgy + 0.5 * 1000 * grav*(z_yb[i*bar] - z_yf[(bar - 1) + (i + np*(ne - 1))*bar])*3.14*pow(gR, 2)*lay;

						_t_wg = (1000 * grav*(-1) * 2 * integral_b(func, 0, gR, z_xb[(bar - 1) + i * bar], z_xf[0 + (i + np*(ne - 1))*bar], gR)
							*(neloc_y_device[bar - 1 + i*bar] - neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar])
							- 1000 * grav * 2 * integral_f(func, 0, gR, z_xb[(bar - 1) + i * bar], z_xf[0 + (i + np*(ne - 1))*bar], gR)
							*(neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_y_device[0 + (i + (ne - 1)*np)*bar]))*lay;  //ok
						_t_wg = _t_wg + (1000 * grav*(-1) * 2 * integral_b(func, 0, gR, z_yb[i*bar], z_yf[(bar - 1) + (i + np*(ne - 1)) * bar], gR)
							*(neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_x_device[i*bar])
							- 1000 * grav * 2 * integral_f(func, 0, gR, z_yb[i*bar], z_yf[(bar - 1) + (i + np*(ne - 1)) * bar], gR)
							*(neloc_x_device[(bar - 1) + (i + (ne - 1)*np)*bar] - neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar]))*lay;  //ok

					}
					else if (sin(gang / 180 * PI) > 0 && cos(gang / 180 * PI) < 0){  //90-180
						_f_wgx = _f_wgx + 0.5 * 1000 * grav*(z_xb[i*bar] - z_xf[(bar - 1) + (i + np*(ne - 1))*bar])*3.14*pow(gR, 2)*lay;
						_f_wgy = _f_wgy + 0.5 * 1000 * grav*(z_yb[(bar - 1) + i * bar] - z_yf[0 + (i + np*(ne - 1))*bar])*3.14*pow(gR, 2)*lay;

						_t_wg = (1000 * grav*(-1) * 2 * integral_b(func, 0, gR, z_xb[(bar - 1) + (i + np*(ne - 1)) * bar], z_xf[i*bar], gR)
							*(neloc_y_device[(bar - 1) + (i + (ne - 1)*np)*bar] - neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar])
							- 1000 * grav * 2 * integral_f(func, 0, gR, z_xb[(bar - 1) + (i + np*(ne - 1)) * bar], z_xf[i*bar], gR)
							*(neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_y_device[i*bar]))*lay;  //ok
						_t_wg = _t_wg + (1000 * grav * 2 * integral_b(func, 0, gR, z_yb[(bar - 1) + i * bar], z_yf[0 + (i + np*(ne - 1))*bar], gR)
							*(neloc_x_device[bar - 1 + i*bar] - neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar])
							+ 1000 * grav * 2 * integral_f(func, 0, gR, z_yb[(bar - 1) + i * bar], z_yf[0 + (i + np*(ne - 1))*bar], gR)
							*(neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_x_device[0 + (i + (ne - 1)*np)*bar]))*lay;  //ok


					}
					else if (sin(gang / 180 * PI) < 0 && cos(gang / 180 * PI) < 0) {  //180-270
						_f_wgx = _f_wgx + 0.5 * 1000 * grav*(z_xb[0 + (i + np*(ne - 1))*bar] - z_xf[(bar - 1) + i * bar])*3.14*pow(gR, 2)*lay;
						_f_wgy = _f_wgy + 0.5 * 1000 * grav*(z_yb[(bar - 1) + (i + np*(ne - 1))*bar] - z_yf[i*bar])*3.14*pow(gR, 2)*lay;

						_t_wg = (1000 * grav*(-1) * 2 * integral_b(func, 0, gR, z_xb[0 + (i + np*(ne - 1))*bar], z_xf[(bar - 1) + i * bar], gR)
							*(neloc_y_device[0 + (i + (ne - 1)*np)*bar] - neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar])
							- 1000 * grav * 2 * integral_f(func, 0, gR, z_xb[0 + (i + np*(ne - 1))*bar], z_xf[(bar - 1) + i * bar], gR)
							*(neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_y_device[bar - 1 + i*bar]))*lay;  //ok

						_t_wg = _t_wg + (1000 * grav*(-1) * 2 * integral_b(func, 0, gR, z_yb[(bar - 1) + (i + np*(ne - 1)) * bar], z_yf[i*bar], gR)
							*(neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_x_device[(bar - 1) + (i + (ne - 1)*np)*bar])
							- 1000 * grav * 2 * integral_f(func, 0, gR, z_yb[i*bar], z_yf[(bar - 1) + (i + np*(ne - 1)) * bar], gR)
							*(neloc_x_device[i*bar] - neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar]))*lay;  //ok
					}

					else { //if (sin(gang / 180 * PI) < 0 && cos(gang / 180 * PI) > 0) { //270-360
						_f_wgx = _f_wgx + 0.5 * 1000 * grav*(z_xb[(bar - 1) + (i + np*(ne - 1))*bar] - z_xf[i*bar])*3.14*pow(gR, 2)*lay;
						_f_wgy = _f_wgy + 0.5 * 1000 * grav*(z_yb[0 + (i + np*(ne - 1))*bar] - z_yf[(bar - 1) + i * bar])*3.14*pow(gR, 2) *lay;


						_t_wg = (1000 * grav*(-1) * 2 * integral_b(func, 0, gR, z_xb[i*bar], z_xf[(bar - 1) + (i + np*(ne - 1)) * bar], gR)
							*(neloc_y_device[i*bar] - neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar])
							- 1000 * grav * 2 * integral_f(func, 0, gR, z_xb[i*bar], z_xf[(bar - 1) + (i + np*(ne - 1)) * bar], gR)
							*(neloc_y_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_y_device[(bar - 1) + (i + (ne - 1)*np)*bar]))*lay; //ok

						_t_wg = _t_wg + (1000 * grav * 2 * integral_b(func, 0, gR, z_yb[0 + (i + np*(ne - 1))*bar], z_yf[(bar - 1) + i * bar], gR)
							*(neloc_x_device[0 + (i + (ne - 1)*np)*bar] - neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar])
							+ 1000 * grav * 2 * integral_f(func, 0, gR, z_yb[(bar - 1) + i * bar], z_yf[0 + (i + np*(ne - 1))*bar], gR)
							*(neloc_x_device[(bar - 1) / 2 + (i + (ne - 1)*np / 2)*bar] - neloc_x_device[bar - 1 + i*bar]))*lay;  //ok


					}

					//----------------uc ini-----hydrodynamic---other force--part sidel					
					_f_wgx = _f_wgx + 0.5 * 1000 * grav*(z_xb[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar] - z_xf[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar])
						*3.14*pow(gR, 2) *(fabs(sin(gang / 180 * PI) / 2)*(ne - 1) + fabs(cos(gang / 180 * PI) / 2)
						*(bar - 1)
						)*lay;
					
					_f_wgy = _f_wgy + 0.5 * 1000 * grav*(z_yb[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar] - z_yf[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar])
						*3.14*pow(gR, 2) * (fabs(cos(gang / 180 * PI) / 2)*(ne - 1) + fabs(sin(gang / 180 * PI) / 2)
						*(bar - 1)
						)*lay;
					
					//--------------torque---------------------------
					t_sign = fabs(cos(gang / 180 * PI)*sin(gang / 180 * PI)) / (cos(gang / 180 * PI)*sin(gang / 180 * PI));

					_t_wg = _t_wg - t_sign*lay * (0.5 * 1000 * grav*(z_xb[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar] - z_xf[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar])
						*3.14*pow(gR, 2)*fabs(sin(gang / 180 * PI) / 2)*gR*(
						(1 - 0.5*fabs(sin(gang / 180 * PI)))*0.5*(ne + 1) + fabs(sin(gang / 180 * PI))*((ne * ne) - 1) / 8)
						+ 0.5 * 1000 * grav*(z_yb[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar] - z_yf[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar])
						*3.14*pow(gR, 2) * fabs(cos(gang / 180 * PI) / 2)*gR*(
						(1 - 0.5*fabs(cos(gang / 180 * PI)))*0.5*(ne + 1) + fabs(cos(gang / 180 * PI))*((ne * ne) - 1) / 8)
						+ 0.5 * 1000 * grav*(z_xb[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar] - z_xf[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar])
						*3.14*pow(gR, 2) * fabs(cos(gang / 180 * PI) / 2)*gR*(
						(1 - 0.5*fabs(cos(gang / 180 * PI)))*0.5*(bar + 1) + fabs(cos(gang / 180 * PI))*((bar * bar) - 1) / 8)
						+ 0.5 * 1000 * grav*(z_yb[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar] - z_yf[(bar - 1) / 2 + (i + np*(ne - 1) / 2) *bar])
						*3.14*pow(gR, 2) * fabs(sin(gang / 180 * PI) / 2)*gR*(
						(1 - 0.5*fabs(sin(gang / 180 * PI)))*0.5*(bar + 1) + fabs(sin(gang / 180 * PI))*((bar * bar) - 1) / 8)
						);	
				}
     
			} //end of u_g>u_c
			
			else {
				if (gvx != 0 || gvy != 0){
					_f_wgx = -(1 - h_min / _z_rela)*_mu*gm * grav*gvx / (sqrt(pow(gvx, 2) + pow(gvy, 2)));
					_f_wgy = -(1 - h_min / _z_rela)*_mu*gm * grav*gvy / (sqrt(pow(gvx, 2) + pow(gvy, 2)));
					_tau_wgx = 0;
					_tau_wgy = 0;
					_t_wg = 0.0;
			
				}
				else{
					_f_wgx = 0;
					_f_wgy = 0;
					_tau_wgx = 0;
					_tau_wgy = 0;
					_t_wg = 0.0;
					
				}
				
			}
		} // end of u_g<z_rela
		
		else {
			_f_wgx = 0;
			_tau_wgx = 0;

			_f_wgy = 0;
			_tau_wgy = 0;
			_t_wg = 0.0;
        }

// Part 7: Update all global variables
		g_gz_device[i] = g_gz;
        tau_wgx_device[i] = _tau_wgx;
        tau_wgy_device[i] = _tau_wgy;

		g_gfx_device[i] += _f_wgx;
		g_gfy_device[i] += _f_wgy;
		g_gt_device[i] += _t_wg;

		// update accelaration
		g_gax_device[i] = g_gfx_device[i] / gm;
		g_gay_device[i] = g_gfy_device[i] / gm;
		g_ganga_device[i] = g_gt_device[i] / g_gI_device[i];

		// update grain velocity and angle velocity
		g_gvx_device[i] += 0.5 * dt_g[0] * g_gax_device[i];
		g_gvy_device[i] += 0.5 * dt_g[0] * g_gay_device[i];
		g_gangv_device[i] += 0.5 * dt_g[0] * g_ganga_device[i];


		delete[] z_g;
		delete[] z_xff;
		delete[] z_yff;
		delete[] z_xbb;
		delete[] z_ybb;
		delete[] h_g;
		delete[] h_xff;
		delete[] h_yff;
		delete[] h_xbb;
		delete[] h_ybb;
	}
}


void loc_debris_hydforce_cuda(at::Tensor dx,
						at::Tensor dt_g, 
						at::Tensor maskIndex,
						at::Tensor num_x_device,
						at::Tensor num_y_device,
						at::Tensor zb_device,
						at::Tensor h_device,
						at::Tensor z_device,
						at::Tensor qx_device,
						at::Tensor qy_device, 
						at::Tensor g_gne_device,
						at::Tensor g_gbar_device,
						at::Tensor g_glay_device,
						at::Tensor g_gR_device, 
						at::Tensor g_gp_device,
						at::Tensor g_gt_device,
						at::Tensor g_gm_device,
						at::Tensor g_gz_device,
						at::Tensor g_gx_device,
						at::Tensor g_gy_device,
						at::Tensor neloc_x_device,
						at::Tensor neloc_y_device,
						at::Tensor mu,
						at::Tensor g_properties,
						at::Tensor g_grho_device,
						at::Tensor g_gvx_device,
						at::Tensor g_gvy_device,	
						at::Tensor g_gfx_device,
						at::Tensor g_gfy_device,
						at::Tensor cell_f_xP_device,
						at::Tensor cell_f_xN_device,
						at::Tensor cell_f_yP_device,
						at::Tensor cell_f_yN_device,
						at::Tensor tau_wgx_device,  
						at::Tensor tau_wgy_device, 
						at::Tensor g_gI_device,
						at::Tensor g_gax_device,
						at::Tensor g_gay_device,
						at::Tensor g_gang_device,
						at::Tensor g_gangv_device,
						at::Tensor g_ganga_device){

		at::cuda::CUDAGuard device_guard(g_gx_device.device());
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		const int np = g_gx_device.numel();
		const int ng = neloc_x_device.numel();

		int M, N;
  		M = maskIndex.size(0);
  		N = maskIndex.size(1);

		if (np == 0) {
			return;
		}

		int thread_0 = 512;
		int block_0 = (np + 512 - 1) / 512;

		AT_DISPATCH_FLOATING_TYPES(
			g_gx_device.type(), "loc_debris_hydforce_kernel", ([&] {
				loc_debris_hydforce_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
					M, N, np, ng,
					dx.data<scalar_t>(),
					dt_g.data<scalar_t>(), 
					maskIndex.data<int32_t>(),
					num_x_device.data<int32_t>(),
					num_y_device.data<int32_t>(),
					zb_device.data<scalar_t>(),
					h_device.data<scalar_t>(),
					z_device.data<scalar_t>(),
					qx_device.data<scalar_t>(),
					qy_device.data<scalar_t>(), 
					g_gne_device.data<int32_t>(),
					g_gbar_device.data<int32_t>(),
					g_glay_device.data<int32_t>(),
					g_gR_device.data<scalar_t>(), 
					g_gp_device.data<scalar_t>(),
					g_gt_device.data<scalar_t>(),
					g_gm_device.data<scalar_t>(),
					g_gz_device.data<scalar_t>(),
					g_gx_device.data<scalar_t>(),
					g_gy_device.data<scalar_t>(),
					neloc_x_device.data<scalar_t>(),
					neloc_y_device.data<scalar_t>(),
					mu.data<scalar_t>(),
					g_properties.data<scalar_t>(),
					g_grho_device.data<scalar_t>(),
					g_gvx_device.data<scalar_t>(),
					g_gvy_device.data<scalar_t>(),	
					g_gfx_device.data<scalar_t>(),
					g_gfy_device.data<scalar_t>(),
					cell_f_xP_device.data<scalar_t>(),
					cell_f_xN_device.data<scalar_t>(),
					cell_f_yP_device.data<scalar_t>(),
					cell_f_yN_device.data<scalar_t>(),
					tau_wgx_device.data<scalar_t>(), 
					tau_wgy_device.data<scalar_t>(), 
					g_gI_device.data<scalar_t>(),
					g_gax_device.data<scalar_t>(),
					g_gay_device.data<scalar_t>(),
					g_gang_device.data<scalar_t>(),
					g_gangv_device.data<scalar_t>(),
					g_ganga_device.data<scalar_t>());
		}));
		
		
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}

