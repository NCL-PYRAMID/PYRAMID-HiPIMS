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
__global__ void loc_debris_kernel(const int M, const int N, const int np, const int ng,
	scalar_t *__restrict__ dx,
	int32_t *__restrict__ maskIndex,
	int32_t *__restrict__ num_x_device, 
	int32_t *__restrict__ num_y_device,
	scalar_t *__restrict__ zb_device, 
	scalar_t *__restrict__ z_device, 
	scalar_t *__restrict__ qx_device, 
	scalar_t *__restrict__ qy_device, 
	scalar_t *__restrict__ z_rela, 
	scalar_t *__restrict__ r_rela, 
	int32_t *__restrict__ lay_sub, 
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
	scalar_t *__restrict__ g_gvx_device,
	scalar_t *__restrict__ g_gvy_device,
	scalar_t *__restrict__ g_gfx_device,
	scalar_t *__restrict__ g_gfy_device,
	scalar_t *__restrict__ qx_g, 
	scalar_t *__restrict__ qy_g, 
	scalar_t *__restrict__ h_g, 
	scalar_t *__restrict__ z_xf, 
	scalar_t *__restrict__ h_xf, 
	scalar_t *__restrict__ z_yf, 
	scalar_t *__restrict__ h_yf, 
	scalar_t *__restrict__ z_xb, 
	scalar_t *__restrict__ h_xb, 
	scalar_t *__restrict__ z_yb, 
	scalar_t *__restrict__ h_yb, 
	scalar_t *__restrict__ h_xff, 
	scalar_t *__restrict__ h_yff,
	scalar_t *__restrict__ h_xbb,
	scalar_t *__restrict__ h_ybb, 
	scalar_t *__restrict__ t_dem, 
	scalar_t *__restrict__ cell_f_xP_device, 
	scalar_t *__restrict__ cell_f_xN_device, 
	scalar_t *__restrict__ cell_f_yP_device, 
	scalar_t *__restrict__ cell_f_yN_device){

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i<np){
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

		// parameters
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

		// need to be updated after calculation
		scalar_t g_gz = g_gz_device[i];
		scalar_t _r_rela = r_rela[i];

		z_rela[i] = g_gm_device[i] / (1000 * gR * (bar + 1)*gR * (ne + 1));//rho * g * V
		lay_sub[i] = int(z_rela[i] / (2 * gR));
		_r_rela = (lay_sub[i] * 2 + 1)*gR - z_rela[i];

		//correct gz
		//the location of the central of debris
		deb_xc = int(floor(g_gx_device[i] / _dx));
		deb_yc = int(floor(g_gy_device[i] / _dx));
		g_gz = z_device[deb_xc + deb_yc * N] + lay * gR;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		scalar_t zb_g;
		scalar_t zb_xf;
		scalar_t zb_yf; 
		scalar_t zb_xb; 
		scalar_t zb_yb; 
		scalar_t zb_xff; 
		scalar_t zb_yff; 
		scalar_t zb_xbb;
		scalar_t zb_ybb;		

		scalar_t *z_g = new scalar_t[ng];
		scalar_t *z_xff = new scalar_t[ng];
		scalar_t *z_yff = new scalar_t[ng];
		scalar_t *z_xbb = new scalar_t[ng];
		scalar_t *z_ybb = new scalar_t[ng];

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

		for (int s = 0; s < ne; s++){
			for (int k = 0; k < bar; k++){
				// 为了计算水流作用力，和判断漂浮物前后左右四个方向上的zb，所以用了比较多的定位关系。
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
				h_g[pid] = z_g[pid] - zb_g;

				// f
				num_xf = int32_t(ceil((neloc_x_device[pid] + gR) / _dx)) + 2;
				num_yf = int32_t(ceil((neloc_y_device[pid] + gR) / _dx)) + 2;

				id_xf = maskIndex[num_xf + num_yc * N];
				zb_xf = zb_device[id_xf];
				z_xf[pid] = z_device[id_xf];
				h_xf[pid] = z_xf[pid] - zb_xf;

				id_yf = maskIndex[num_xc + num_yf * N];
				zb_yf = zb_device[id_yf];
				z_yf[pid] = z_device[id_yf];
				h_yf[pid] = z_yf[pid] - zb_yf;

				// b
				num_xb = int32_t(ceil((neloc_x_device[pid] - gR) / _dx)) - 2;
				num_yb = int32_t(ceil((neloc_y_device[pid] - gR) / _dx)) - 2;

				id_xb = maskIndex[num_xb + num_yc * N];
				zb_xb = zb_device[id_xb];
				z_xb[pid] = z_device[id_xb];
				h_xb[pid] = z_xb[pid] - zb_xb;
				
				id_yb = maskIndex[num_xc + num_yb * N];
				zb_yb = zb_device[id_yb];
				z_yb[pid] = z_device[id_yb];		
				h_yb[pid] = z_yb[pid] - zb_yb;

				// ff
				num_xff = int32_t(floor((neloc_x_device[pid] + gR) / _dx));
				num_yff = int32_t(floor((neloc_y_device[pid] + gR) / _dx));

				id_xff = maskIndex[num_xff + num_yc * N];
				zb_xff = zb_device[id_xff];
				z_xff[pid] = z_device[id_xff];
				
				id_yff = maskIndex[num_xc + num_yff * N];
				zb_yff = zb_device[id_yff];
				z_yff[pid] = z_device[id_yff];

				// bb
				num_xbb = int32_t(floor((neloc_x_device[pid] - gR) / _dx));			
				num_ybb = int32_t(floor((neloc_y_device[pid] - gR) / _dx));

				id_xbb = maskIndex[num_xbb + num_yc * N];
				zb_xbb = zb_device[id_xbb];
				z_xbb[pid] = z_device[id_xbb];
				h_xbb[pid] = z_xbb[pid] - zb_xbb;

				id_ybb = maskIndex[num_xc + num_ybb * N];
				zb_ybb = zb_device[id_ybb];
				z_ybb[pid] = z_device[id_ybb];
				h_ybb[pid] = z_ybb[pid] - zb_ybb;

				// xff
				xff = zb_xff;
				for (int ghost = num_ybb; ghost <= num_yff; ghost++){
					id_ghost = maskIndex[num_xff + ghost*N];
					if (zb_device[id_ghost] > xff){
						xff = zb_device[id_ghost];
					}
				}
				h_xff[pid] = z_xff[pid] - zb_xff;

				// yff
				yff = zb_yff;
				for (int ghost = num_xbb; ghost <= num_xff; ghost++){
					id_ghost = maskIndex[ghost + num_yff*N];
					if (zb_device[id_ghost] > yff){
						yff = zb_device[id_ghost];
					}
				}
				h_yff[pid] = z_yff[pid] - zb_yff;

				// xbb
				xbb = zb_xbb;
				for (int ghost = num_ybb; ghost <= num_yff; ghost++){
					id_ghost = maskIndex[num_xbb + ghost*N];
					if (zb_device[id_ghost] > xbb){
						xbb = zb_device[id_ghost];
					}
				}
				
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

				//----------------------------------------------------------------------------
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
					cell_f_xP_device[num_xff + num_yc * N] = obsp_r;
				}

				if (zb_xbb >= g_gz){
					obs_l = num_xbb * _dx + _dx;// / 2;
					obsp_l = compute_force_left_wall(i, 1, s, k, obs_l,  _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvx, gfx, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_xN_device[num_xbb + num_yc * N] = obsp_l;
					}
				else if (xbb >= g_gz && cell_l == 0 && (cell_u + cell_d) == 0){
					obs_l = num_xbb * _dx + _dx;// / 2;
					obsp_l = compute_force_left_wall(i, 1, s, k, obs_l,  _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvx, gfx, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_xN_device[num_xbb + num_yc * N] = obsp_l;
				}
				
				if (zb_yff >= g_gz){
					obs_u = num_yff * _dx; // - dy / 2;
					obsp_u = compute_force_upper_wall(i, 1, s, k, obs_u, _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvy, gfy, gp, gt, neloc_x_device, neloc_y_device);
				}
				else if (yff >= g_gz && cell_u == 0 && (cell_l + cell_r) == 0){
					obs_u = num_yff * _dx; // - dy / 2;
					obsp_u = compute_force_upper_wall(i, 1, s, k, obs_u, _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvy, gfy, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_yP_device[num_xc + num_yff * N] = obsp_u;					
				}
				
				if (zb_ybb >= g_gz){
					obs_d = num_ybb * _dx + _dx;// / 2;
					obsp_d = compute_force_lower_wall(i, 1, s, k, obs_d, _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvy, gfy, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_yN_device[num_xc + num_ybb * N] = obsp_d;
				}
				else if (ybb >= g_gz && cell_d == 0 && (cell_l + cell_r) == 0){
					obs_d = num_ybb * _dx + _dx;// / 2;
					obsp_d = compute_force_lower_wall(i, 1, s, k, obs_d, _mu, kn, nu, kt, 
						np, ne, bar, lay, gR, gvy, gfy, gp, gt, neloc_x_device, neloc_y_device);
					cell_f_yN_device[num_xc + num_ybb * N] = obsp_d;
				}	
			} //end of 1st loop of k++
		}// end of 1st loop of s++

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

		if (h_maxs > z_rela[i]){
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
		
		g_gz_device[i] = g_gz;
		r_rela[i] = _r_rela;

		delete[] z_g;
		delete[] z_xff;
		delete[] z_yff;
		delete[] z_xbb;
		delete[] z_ybb;
	}
}


void loc_debris_cuda(at::Tensor dx, 
	at::Tensor maskIndex, 
	at::Tensor num_x_device, 
	at::Tensor num_y_device,
	at::Tensor zb_device, 
	at::Tensor z_device, 
	at::Tensor qx_device, 
	at::Tensor qy_device, 
	at::Tensor z_rela, 
	at::Tensor r_rela, 
	at::Tensor lay_sub, 
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
	at::Tensor g_gv_device, 
	at::Tensor g_gf_device,
	at::Tensor qx_g, 
	at::Tensor qy_g, 
	at::Tensor h_g, 
	at::Tensor z_xf, 
	at::Tensor h_xf, 
	at::Tensor z_yf,
	at::Tensor h_yf, 
	at::Tensor z_xb, 
	at::Tensor h_xb,
	at::Tensor z_yb, 
	at::Tensor h_yb,
	at::Tensor h_xff, 
	at::Tensor h_yff, 
	at::Tensor h_xbb,
	at::Tensor h_ybb, 
	at::Tensor t_dem, 
	at::Tensor cell_f_xP_device, 
	at::Tensor cell_f_xN_device, 
	at::Tensor cell_f_yP_device, 
	at::Tensor cell_f_yN_device){

		at::cuda::CUDAGuard device_guard(g_gx_device.device());
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		const int np = g_gx_device.numel();
		const int ng = h_g.numel();
		const int M = maskIndex.numel();
		const int N = maskIndex.numel();
		if (np == 0) {
			return;
		}

		int thread_0 = 512;
		int block_0 = (np + 512 - 1) / 512;

		AT_DISPATCH_FLOATING_TYPES(
			g_gx_device.type(), "loc_debris_cuda", ([&] {
				loc_debris_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
					M, N, np, ng,
					dx.data<scalar_t>(),
					maskIndex.data<int32_t>(),
					num_x_device.data<int32_t>(),
					num_y_device.data<int32_t>(),
					zb_device.data<scalar_t>(),
					z_device.data<scalar_t>(),
					qx_device.data<scalar_t>(),
					qy_device.data<scalar_t>(), 
					z_rela.data<scalar_t>(),
					r_rela.data<scalar_t>(),
					lay_sub.data<int32_t>(), 
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
					g_gv_device.data<scalar_t>(),	
					g_gf_device.data<scalar_t>(),
					qx_g.data<scalar_t>(), 
					qy_g.data<scalar_t>(), 
					h_g.data<scalar_t>(), 
					z_xf.data<scalar_t>(), 
					h_xf.data<scalar_t>(), 
					z_yf.data<scalar_t>(), 
					h_yf.data<scalar_t>(), 
					z_xb.data<scalar_t>(),
					h_xb.data<scalar_t>(), 
					z_yb.data<scalar_t>(), 
					h_yb.data<scalar_t>(), 
					h_xff.data<scalar_t>(), 
					h_yff.data<scalar_t>(), 
					h_xbb.data<scalar_t>(), 
					h_ybb.data<scalar_t>(), 
					t_dem.data<scalar_t>(), 
					cell_f_xP_device.data<scalar_t>(),
					cell_f_xN_device.data<scalar_t>(),
					cell_f_yP_device.data<scalar_t>(),
					cell_f_yN_device.data<scalar_t>());
		}));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
