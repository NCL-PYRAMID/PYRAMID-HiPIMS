#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

////////////////////////////////////////////////////////////////////////////////////////////////
// particle neighbor:
//    c       |       d      c       |       d      c       |       d      c       |       d
//    3       |       2      0    *  |       1      4       |       3      5       |   *   0  
//    --------+--------      --------+--------      --------+--------      --------+--------
//    0   *   |       1      7       |       8      5       |   *   0      6       |       7                         
//    a       |       b      a       |       b      a       |       b      a       |       b      
//  (xp>x_cell,y<y_cell)   (xp>x_cell,y>y_cell)    (x<x_cell,y<y_cell)    (x<x_cell,y>y_cell)
////////////////////////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__device__ scalar_t distance(scalar_t x1, scalar_t x2, scalar_t y1, scalar_t y2){
    return sqrt( pow((x1 - x2),2.0) + pow((y1 - y2),2.0) );
}

// interpolote velocity just based on distance
template <typename scalar_t>
__device__ void velocity_interpolation(int32_t *nei_id, scalar_t *nei_x, scalar_t *nei_y,
                                            scalar_t *nei_u, scalar_t *nei_v, 
                                            scalar_t xp, scalar_t yp,scalar_t &up, scalar_t &vp){
    scalar_t u_neighbor[4], v_neighbor[4];
    scalar_t dis_neighbor[4], Pji[4];
    int32_t j_neighbor[4];

    scalar_t x_cell = nei_x[0];
    scalar_t y_cell = nei_y[0];
    scalar_t dis_cell = distance(xp, x_cell, yp, y_cell);

    // find 4 nearest neighbor cells, index: 0-a, 1-b, 2-c, 3-d
    if (dis_cell<=10e-6){
        up = nei_u[0];
        vp = nei_v[0];
    }
    else{
        if (xp>x_cell && yp>y_cell) {
            for (int j=0; j<4; j++) {
                j_neighbor[j] = j < 2 ? j + 7 : 3 - j; // 7, 8 , 1, 0
            }
        }
        else if (xp>x_cell && yp<=y_cell) {
            for (int j=0; j<4; j++) {
                j_neighbor[j] = j;
            }
        }
        else if (xp<=x_cell && yp<=y_cell) {
            j_neighbor[0] = 5;
            j_neighbor[1] = 0;
            j_neighbor[2] = 3;
            j_neighbor[3] = 4;
        }
        else{
            j_neighbor[0] = 6;
            j_neighbor[1] = 7;
            j_neighbor[2] = 0;
            j_neighbor[3] = 5;
        }
    
        scalar_t Pji_tol = 0.0;
    
        // calculate weight velocity
        for (int j=0; j<4; j++) {
    
            int32_t j_id = j_neighbor[j];
    
            if (nei_id[j_id] == -1) {
                Pji[j] = 0.0;
                u_neighbor[j] = 0.0;
                v_neighbor[j] = 0.0;
            }
            else {
                dis_neighbor[j] = distance(nei_x[j_id], xp, nei_y[j_id], yp);
                Pji[j] = 1.0 / dis_neighbor[j];
                u_neighbor[j] = nei_u[j_id] * Pji[j];
                v_neighbor[j] = nei_v[j_id] * Pji[j];
            }
    
            Pji_tol = Pji_tol + Pji[j];
        }

        up = (u_neighbor[0] + u_neighbor[1] + u_neighbor[2] + u_neighbor[3]) / Pji_tol;
        vp = (v_neighbor[0] + v_neighbor[1] + v_neighbor[2] + v_neighbor[3]) / Pji_tol;
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////
// particle neighbor:
//    d       |       c      a       |       b      c       |       d      b       |       a
//    3       |       2      0    *  |       1      4       |       3      5       |   *   0  
//    --------+--------      --------+--------      --------+--------      --------+--------
//    0   *   |       1      7       |       8      5       |   *   0      6       |       7                         
//    a       |       b      d       |       c      b       |       a      c       |       d      
//  (xp>x_cell,y<y_cell)   (xp>x_cell,y>y_cell)    (x<x_cell,y<y_cell)    (x<x_cell,y>y_cell)
//   a:particle cellid   b:same y    d: same x
////////////////////////////////////////////////////////////////////////////////////////////////
// interpolate velocity based on jingchun's method
template <typename scalar_t>
__device__ void velocity_interpolation_jingchun(int32_t *nei_id, 
                                            scalar_t *nei_x, scalar_t *nei_y,
                                            scalar_t *nei_u, scalar_t *nei_v, 
                                            scalar_t xp, scalar_t yp,scalar_t &up, scalar_t &vp,
                                            scalar_t dx){
    scalar_t u_neighbor[4], v_neighbor[4];
    scalar_t x_neighbor[4], y_neighbor[4];
    int32_t j_neighbor[4];
    int32_t Mask_neighbor[4];
    scalar_t x_cell = nei_x[0];
    scalar_t y_cell = nei_y[0];

    if (abs(nei_u[0])<=10e-6 && abs(nei_v[0]<=10e-6)){
        up = 0.0;
        vp = 0.0;
    }
    else{
        // cell 0: (i,j);  cell 1: (i+1,j)/(i-1,j);   cell 2: (i+1,j+1)/(i-1,j-1);  cell 3: (i,j+1)/(i,j-1)
        j_neighbor[0] = 0;
        if (xp>x_cell && yp>y_cell) {
            j_neighbor[1] = 1;
            j_neighbor[2] = 8;
            j_neighbor[3] = 7;
        }
        else if (xp>x_cell && yp<=y_cell) {
            j_neighbor[1] = 1;
            j_neighbor[2] = 2;
            j_neighbor[3] = 3;
        }
        else if (xp<=x_cell && yp<=y_cell) {
            j_neighbor[1] = 5;
            j_neighbor[2] = 4;
            j_neighbor[3] = 3;
        }
        else{
            j_neighbor[1] = 5;
            j_neighbor[2] = 6;
            j_neighbor[3] = 7;
        }

        int32_t tol_wetMask=0;
        for(int j=0;j<4;j++){
            int32_t j_id = j_neighbor[j];

            if (nei_id[j_id] == -1) {
                u_neighbor[j] = 0.0;
                v_neighbor[j] = 0.0;
                Mask_neighbor[j] = 0;
            }
            else {
                u_neighbor[j] = nei_u[j_id];
                v_neighbor[j] = nei_v[j_id];
                scalar_t uv_neighbor = sqrt( pow(u_neighbor[j],2.) + pow(v_neighbor[j],2.0) );
                if (uv_neighbor<=10e-6){
                    Mask_neighbor[j] = 0;
                }
                else{
                    x_neighbor[j] = nei_x[j_id];
                    y_neighbor[j] = nei_y[j_id];
                    Mask_neighbor[j] = 1;
                }
            }
            tol_wetMask += Mask_neighbor[j];
        }


        if(tol_wetMask==1){ //only 1 cell available, particles' velocity equal to cell a 
            up = u_neighbor[0];
            vp = v_neighbor[0];
        }
        else if (tol_wetMask==2){  //2 cells available, particles' velocity equal to cell a&b or a&d
            if(Mask_neighbor[1]==1){ // cell 0&1
                up = ( u_neighbor[0] * abs(x_neighbor[1] - xp) + u_neighbor[1] * abs(x_neighbor[0] - xp) ) / dx;
                vp = ( v_neighbor[0] * abs(x_neighbor[1] - xp) + v_neighbor[1] * abs(x_neighbor[0] - xp) ) / dx;
            }
            else{   // cell 0&3
                up = ( u_neighbor[0] * abs(y_neighbor[3] - yp) + u_neighbor[3] * abs(y_neighbor[0] - yp) ) / dx;
                vp = ( v_neighbor[0] * abs(y_neighbor[3] - yp) + v_neighbor[3] * abs(y_neighbor[0] - yp) ) / dx;
            }
            
        }
        else if (tol_wetMask==3){   //3 cells available, particles' velocity equal to cell a&b&c or a&d&c or a&b&d
            int32_t j_id1, j_id2, j_id3;
            if(Mask_neighbor[1]==0){
                j_id1 = 2;
                j_id2 = 3;
                j_id3 = 0;
            }
            else if(Mask_neighbor[2]==0){
                j_id1 = 1;
                j_id2 = 0;
                j_id3 = 3;
            }
            if(Mask_neighbor[3]==0){
                j_id1 = 0;
                j_id2 = 1;
                j_id3 = 2;
            }

            scalar_t uup,vvp;
            if (x_neighbor[j_id1]==x_neighbor[j_id2]){
                scalar_t xxp = x_neighbor[j_id1];
                uup = ( u_neighbor[j_id1] * abs(y_neighbor[j_id2] - yp) + u_neighbor[j_id2] * abs(y_neighbor[j_id1] - yp) ) / dx;
                vvp = ( v_neighbor[j_id1] * abs(y_neighbor[j_id2] - yp) + v_neighbor[j_id2] * abs(y_neighbor[j_id1] - yp) ) / dx; 

                up = ( u_neighbor[j_id3] * abs(xxp - xp) + uup * abs(x_neighbor[j_id3] - xp) ) / dx;
                vp = ( v_neighbor[j_id3] * abs(xxp - xp) + vvp * abs(x_neighbor[j_id3] - xp) ) / dx; 
            }
            else{
                scalar_t yyp = y_neighbor[j_id1];
                uup = ( u_neighbor[j_id1] * abs(x_neighbor[j_id2] - xp) + u_neighbor[j_id2] * abs(x_neighbor[j_id1] - xp) ) / dx;
                vvp = ( v_neighbor[j_id1] * abs(x_neighbor[j_id2] - xp) + v_neighbor[j_id2] * abs(x_neighbor[j_id1] - xp) ) / dx; 

                up = ( u_neighbor[j_id3] * abs(yyp - yp) + uup * abs(y_neighbor[j_id3] - yp) ) / dx;
                vp = ( v_neighbor[j_id3] * abs(yyp - yp) + vvp * abs(y_neighbor[j_id3] - yp) ) / dx; 
            }
        }
        else{   //4 cells available
            scalar_t uup1, vvp1, uup2, vvp2;
            uup1 = ( u_neighbor[0] * abs(x_neighbor[1] - xp) + u_neighbor[1] * abs(x_neighbor[0] - xp) ) / dx;
            vvp1 = ( v_neighbor[0] * abs(x_neighbor[1] - xp) + v_neighbor[1] * abs(x_neighbor[0] - xp) ) / dx; 

            uup2 = ( u_neighbor[2] * abs(x_neighbor[3] - xp) + u_neighbor[3] * abs(x_neighbor[2] - xp) ) / dx;
            vvp2 = ( v_neighbor[2] * abs(x_neighbor[3] - xp) + v_neighbor[3] * abs(x_neighbor[2] - xp) ) / dx; 

            up = ( uup1 * abs(y_neighbor[3] - yp) + uup2 * abs(y_neighbor[0] - yp) ) / dx;
            vp = ( vvp1 * abs(y_neighbor[3] - yp) + vvp2 * abs(y_neighbor[0] - yp) ) / dx; 
        }
    }
}


template <typename scalar_t>
__device__ void update_CellId(int32_t *nei_id, scalar_t *nei_x, scalar_t *nei_y, scalar_t xp, scalar_t yp, int32_t &cellid, scalar_t dx){
    scalar_t dxp = xp - nei_x[0];
    scalar_t dyp = yp - nei_y[0];
    int32_t c_id;

    // cell 2,1,8
    if (dxp > 0.5*dx){
        if (dyp > 0.5*dx){
            c_id = 8;
        }
        else if (dyp <= -0.5*dx){
            c_id = 2;
        }
        else{
            c_id = 1;
        }
    }
    else if (dxp <= -0.5*dx){
        if (dyp > 0.5*dx){
            c_id = 6;
        }
        else if (dyp <= -0.5*dx){
            c_id = 4;
        }
        else{
            c_id = 5;
        }
    }
    else{
        if (dyp > 0.5*dx){
            c_id = 7;
        }
        else if (dyp <= -0.5*dx){
            c_id = 3;
        }
        else{
            c_id = 0;
        }
    }

    cellid = nei_id[c_id];
}

//////////////////////////////////////////////////////////////////////////
// nei_id:
//                   4     3     2  
//                         |                 
//                   5  -- 0 --  1
//                         |                                      
//                   6     7     8               
//////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void transport_kernel(const int N, const int PNN, int32_t *__restrict__ index, scalar_t *__restrict__ x, scalar_t *__restrict__ y, 
                        scalar_t *__restrict__ u, scalar_t *__restrict__ v, 
                        scalar_t *__restrict__ xp, scalar_t *__restrict__ yp,
                        int32_t *__restrict__ cellid, int32_t *__restrict__ layer, 
                        scalar_t *__restrict__ dx, scalar_t *__restrict__ dt){

    int32_t r_id, rr_id;
    int32_t nei_id[9];
    scalar_t nei_x[9], nei_y[9], nei_u[9], nei_v[9];
    int32_t rot_dir;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<PNN) {
        int32_t _layer = layer[i];
        
        if (_layer == 1){
            scalar_t _xp = xp[i];
            scalar_t _yp = yp[i];
            int32_t c_id = cellid[i]; 

            // parameters for runge-kutta
            scalar_t kxn[5], kyn[5];
            kxn[0] = 0.0;
            kyn[0] = 0.0;
            scalar_t ark4[4] = {0.0, 0.5, 0.5, 1.0};
            scalar_t up, vp;

            // D8 
            // centre cell
            nei_id[0] = c_id;
            nei_x[0] = x[c_id];
            nei_y[0] = y[c_id];
            nei_u[0] = u[c_id];
            nei_v[0] = v[c_id];

            // find neigbhor cell index
            for (int j=0; j<4; j++) {
                r_id = j + 1;
                rot_dir = r_id  > 3 ? r_id - 3 : r_id + 1;
                
                r_id = index[r_id * N + c_id];
                rr_id = index[rot_dir * N + r_id];

                nei_id[j*2+1] = r_id;   //1,3,5,7
                nei_id[j*2+2] = rr_id;  //2,4,6,8
            }

            // find whether neighbor cell 2,4,6,8 is null
            for (int j=0; j<3; j++){
                int32_t l_id = nei_id[j*2 + 1];
                int32_t ll_id = nei_id[j*2 + 3];
                int32_t corner = j*2 + 2;
                if (l_id == -1){
                    if (ll_id==-1){
                        nei_id[corner] = -1;
                    }
                    else{
                        // rot_dir = j < 2 ? 2-j : j+1;
                        rot_dir = j+1;
                    nei_id[corner] = index[rot_dir*N + ll_id];
                    }
                }
            }
            if (nei_id[7] == -1){
                if(nei_id[1] == -1){
                    nei_id[8] = -1;
                }
                else{
                    nei_id[8] = index[4*N + nei_id[1]];
                }
            }

            for (int j=0; j<9; j++){
                r_id = nei_id[j];
                nei_x[j] = x[r_id];
                nei_y[j] = y[r_id];
                nei_u[j] = u[r_id];
                nei_v[j] = v[r_id];
            }
                
            // 4th runge-kutta, interpoltation particles' velocity
            for (int k=0; k<4; k++) {
                scalar_t kdxp = ark4[k] * kxn[k];
                scalar_t kdyp = ark4[k] * kyn[k];

                // cid for runge-kutta
                int32_t cid_new = c_id; 
        
                scalar_t xp1 = _xp + kdxp;
                scalar_t yp1 = _yp + kdyp;

                update_CellId(nei_id, nei_x, nei_y, xp1, yp1, cid_new, dx[0]);

                if (cid_new < 0){   //particles transported out of domain
                    up = u[c_id];
                    vp = v[c_id];
                }
                else if (cid_new == c_id) { //particles cellid not changed
                    velocity_interpolation_jingchun(nei_id, nei_x, nei_y, nei_u, nei_v, xp1, yp1, up, vp, dx[0]);
                }
                else{   //particles transported into another cell
                    scalar_t uvp = sqrt( pow(up,2.) + pow(vp,2.) );
                    scalar_t uvp_cell = sqrt( pow(u[cid_new],2.) + pow(v[cid_new],2.) );
                    if (uvp>=10e-6 && uvp_cell>=10e-6){
                        int32_t kr_id, krr_id;
                        int32_t knei_id[9];
                        scalar_t knei_x[9], knei_y[9], knei_u[9], knei_v[9];
                        int32_t rot_dirk;

                        knei_id[0] = cid_new;
                        knei_x[0] = x[cid_new];
                        knei_y[0] = y[cid_new];
                        knei_u[0] = u[cid_new];
                        knei_v[0] = v[cid_new];

                        for (int j=0; j<4; j++){
                            kr_id = j + 1;
                            rot_dirk = kr_id  > 3 ? kr_id - 3 : kr_id + 1;
                        
                            kr_id = index[kr_id * N + cid_new];
                            krr_id = index[rot_dirk * N + kr_id];

                            knei_id[j*2+1] = kr_id;   //1,3,5,7
                            knei_id[j*2+2] = krr_id;  //2,4,6,8

                            knei_x[j*2+1] = x[kr_id];
                            knei_x[j*2+2] = x[krr_id];

                            knei_y[j*2+1] = y[kr_id];
                            knei_y[j*2+2] = y[krr_id];

                            knei_u[j*2+1] = u[kr_id];
                            knei_u[j*2+2] = u[krr_id];

                            knei_v[j*2+1] = v[kr_id];
                            knei_v[j*2+2] = v[krr_id];
                        }
                        
                        // find whether neighbor cell 2,4,6,8 is null
                        for (int j=0; j<3; j++){
                            int32_t kl_id = knei_id[j*2 + 1];
                            int32_t kll_id = knei_id[j*2 + 3];
                            int32_t cornerk = j*2 + 2;
                            if (kl_id == -1) {
                                if (kll_id==-1){
                                    knei_id[cornerk] = -1;
                                }
                                else{
                                    rot_dirk = j+1;
                                    knei_id[cornerk] = index[rot_dirk*N + kll_id];
                                }
                            }
                        }
                        if (knei_id[7] == -1){
                            if(knei_id[1] == -1){
                                knei_id[8] = -1;
                            }
                            else{
                                knei_id[8] = index[4*N + knei_id[1]];
                            }
                        }

                        velocity_interpolation_jingchun(knei_id, knei_x, knei_y, knei_u, knei_v, xp1, yp1, up, vp, dx[0]);    
                    }
                }

                kxn[k+1] = dt[0] * up;
                kyn[k+1] = dt[0] * vp;
            }

            scalar_t dxp = (kxn[1] + 2.0 * (kxn[2] + kxn[3]) + kxn[4]) / 6.0;
            scalar_t dyp = (kyn[1] + 2.0 * (kyn[2] + kyn[3]) + kyn[4]) / 6.0;

            _xp = _xp + dxp;
            _yp = _yp + dyp;

            update_CellId(nei_id, nei_x, nei_y, _xp, _yp, c_id, dx[0]);

            if(c_id< 0){
                c_id = -9999;
                _layer = -1;
                _xp = -9999;
                _yp = -9999;
            }

            // update particle positions and cellid
            xp[i] = _xp;
            yp[i] = _yp;
            cellid[i] = c_id;
            layer[i] = _layer;
        }
    }
}

//////////////////////////////////////////////////// For Test //////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void transport_kernel_test(const int N, const int PNN, int32_t *__restrict__ index, scalar_t *__restrict__ x, scalar_t *__restrict__ y, 
                        scalar_t *__restrict__ un1, scalar_t *__restrict__ vn1, 
                        scalar_t *__restrict__ un2, scalar_t *__restrict__ vn2, 
                        scalar_t *__restrict__ xp, scalar_t *__restrict__ yp,
                        int32_t *__restrict__ cellid, int32_t *__restrict__ layer, 
                        scalar_t *__restrict__ dx, scalar_t *__restrict__ dt){

    int32_t r_id, rr_id;
    int32_t nei_id[9];
    scalar_t nei_x[9], nei_y[9], nei_u[9], nei_v[9];
    int32_t rot_dir;
    scalar_t up, vp;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<PNN) {
        int32_t _layer = layer[i];
        
        if (_layer == 1){
            scalar_t _xp = xp[i];
            scalar_t _yp = yp[i];
            int32_t c_id = cellid[i];
            
            // parameters for runge-kutta
            scalar_t kxn[5], kyn[5];
            kxn[0] = 0.0;
            kyn[0] = 0.0;
            scalar_t ark4[4] = {0.0, 0.5, 0.5, 1.0};

            // D8 
            // centre cell
            nei_id[0] = c_id;
            nei_x[0] = x[c_id];
            nei_y[0] = y[c_id];
            nei_u[0] = un1[c_id];
            nei_v[0] = vn1[c_id];

            // find neigbhor cell index
            for (int j=0; j<4; j++) {
                r_id = j + 1;
                rot_dir = r_id  > 3 ? r_id - 3 : r_id + 1;
                
                r_id = index[r_id * N + c_id];
                rr_id = index[rot_dir * N + r_id];

                nei_id[j*2+1] = r_id;   //1,3,5,7
                nei_id[j*2+2] = rr_id;  //2,4,6,8
            }

            // find whether neighbor cell 2,4,6,8 is null
            for (int j=0; j<3; j++){
                int32_t l_id = nei_id[j*2 + 1];
                int32_t ll_id = nei_id[j*2 + 3];
                int32_t corner = j*2 + 2;
                if (l_id == -1) {
                    if (ll_id==-1){
                        nei_id[corner] = -1;
                    }
                    else{
                        // rot_dir = j < 2 ? 2-j : j+1;
                        rot_dir = j+1;
                        nei_id[corner] = index[rot_dir*N + ll_id];
                    }
                }
            }
            if (nei_id[7] == -1){
                if(nei_id[1] == -1){
                    nei_id[8] = -1;
                }
                else{
                    nei_id[8] = index[4*N + nei_id[1]];
                }
            }

            for (int j=0; j<9; j++){
                r_id = nei_id[j];
                nei_x[j] = x[r_id];
                nei_y[j] = y[r_id];
                nei_u[j] = un1[r_id];
                nei_v[j] = vn1[r_id];
            }
            
            // 2nd runge-kutta, interpoltation particles' velocity
            // kn1
            for (int k=0; k<4; k++) {
                scalar_t kdxp = ark4[k] * kxn[k];
                scalar_t kdyp = ark4[k] * kyn[k];

                // cid for runge-kutta
                int32_t cid_new = c_id; 
        
                scalar_t xp1 = _xp + kdxp;
                scalar_t yp1 = _yp + kdyp;

                update_CellId(nei_id, nei_x, nei_y, xp1, yp1, cid_new, dx[0]);

                if (cid_new < 0){   //particles transported out of domain
                    up = un1[c_id];
                    vp = vn1[c_id];
                }
                else if (cid_new == c_id) { //particles cellid not changed
                    velocity_interpolation_jingchun(nei_id, nei_x, nei_y, nei_u, nei_v, xp1, yp1, up, vp, dx[0]);
                }
                else{   //particles transported into another cell
                    scalar_t uvp = sqrt( pow(up,2.) + pow(vp,2.) );
                    scalar_t uvp_cell = sqrt( pow(un1[cid_new],2.) + pow(vn1[cid_new],2.) );
                    if (uvp>=10e-6 && uvp_cell>=10e-6){
                        int32_t kr_id, krr_id;
                        int32_t knei_id[9];
                        scalar_t knei_x[9], knei_y[9], knei_u[9], knei_v[9];
                        int32_t rot_dirk;

                        knei_id[0] = cid_new;
                        knei_x[0] = x[cid_new];
                        knei_y[0] = y[cid_new];
                        knei_u[0] = un1[cid_new];
                        knei_v[0] = vn1[cid_new];

                        for (int j=0; j<4; j++){
                            kr_id = j + 1;
                            rot_dirk = kr_id  > 3 ? kr_id - 3 : kr_id + 1;
                        
                            kr_id = index[kr_id * N + cid_new];
                            krr_id = index[rot_dirk * N + kr_id];

                            knei_id[j*2+1] = kr_id;   //1,3,5,7
                            knei_id[j*2+2] = krr_id;  //2,4,6,8

                            knei_x[j*2+1] = x[kr_id];
                            knei_x[j*2+2] = x[krr_id];

                            knei_y[j*2+1] = y[kr_id];
                            knei_y[j*2+2] = y[krr_id];

                            knei_u[j*2+1] = un1[kr_id];
                            knei_u[j*2+2] = un1[krr_id];

                            knei_v[j*2+1] = vn1[kr_id];
                            knei_v[j*2+2] = vn1[krr_id];
                        }
                        
                        // find whether neighbor cell 2,4,6,8 is null
                        for (int j=0; j<3; j++){
                            int32_t kl_id = knei_id[j*2 + 1];
                            int32_t kll_id = knei_id[j*2 + 3];
                            int32_t cornerk = j*2 + 2;
                            if (kl_id == -1) {
                                if (kll_id==-1){
                                    knei_id[cornerk] = -1;
                                }
                                else{
                                    rot_dirk = j+1;
                                    knei_id[cornerk] = index[rot_dirk*N + kll_id];
                                }
                            }
                        }
                        if (knei_id[7] == -1){
                            if(knei_id[1] == -1){
                                knei_id[8] = -1;
                            }
                            else{
                                knei_id[8] = index[4*N + knei_id[1]];
                            }
                        }

                        velocity_interpolation_jingchun(knei_id, knei_x, knei_y, knei_u, knei_v, xp1, yp1, up, vp, dx[0]);    
                    }
                }

                kxn[k+1] = dt[0] * up;
                kyn[k+1] = dt[0] * vp;
            }

            scalar_t dxp1 = (kxn[1] + 2.0 * (kxn[2] + kxn[3]) + kxn[4]) / 6.0;
            scalar_t dyp1 = (kyn[1] + 2.0 * (kyn[2] + kyn[3]) + kyn[4]) / 6.0;

            // time n+1
            // kn2
            int32_t cid_new; 
            scalar_t xp1 = _xp + dxp1;
            scalar_t yp1 = _yp + dyp1;
            update_CellId(nei_id, nei_x, nei_y, xp1, yp1, cid_new, dx[0]);

            if (cid_new < 0){
                up = up;
                vp = vp;
            }
            else if (cid_new == c_id) { //particles cellid not changed
                for (int j=0; j<9; j++){
                    r_id = nei_id[j];
                    nei_u[j] = un2[r_id];
                    nei_v[j] = vn2[r_id];
                } 
                velocity_interpolation_jingchun(nei_id, nei_x, nei_y, nei_u, nei_v, xp1, yp1, up, vp, dx[0]);
            }
            else {   //particles transported into another cell
                int32_t kr_id, krr_id;
                int32_t knei_id[9];
                scalar_t knei_x[9], knei_y[9], knei_u[9], knei_v[9];
                int32_t rot_dirk;

                knei_id[0] = cid_new;
                knei_x[0] = x[cid_new];
                knei_y[0] = y[cid_new];
                knei_u[0] = un2[cid_new];
                knei_v[0] = vn2[cid_new];

                for (int j=0; j<4; j++){
                    kr_id = j + 1;
                    rot_dirk = kr_id  > 3 ? kr_id - 3 : kr_id + 1;
                
                    kr_id = index[kr_id * N + cid_new];
                    krr_id = index[rot_dirk * N + kr_id];

                    knei_id[j*2+1] = kr_id;   //1,3,5,7
                    knei_id[j*2+2] = krr_id;  //2,4,6,8

                    knei_x[j*2+1] = x[kr_id];
                    knei_x[j*2+2] = x[krr_id];

                    knei_y[j*2+1] = y[kr_id];
                    knei_y[j*2+2] = y[krr_id];

                    knei_u[j*2+1] = un2[kr_id];
                    knei_u[j*2+2] = un2[krr_id];

                    knei_v[j*2+1] = vn2[kr_id];
                    knei_v[j*2+2] = vn2[krr_id];
                }
                
                // find whether neighbor cell 2,4,6,8 is null
                for (int j=0; j<3; j++){
                    int32_t kl_id = knei_id[j*2 + 1];
                    int32_t kll_id = knei_id[j*2 + 3];
                    int32_t cornerk = j*2 + 2;
                    if (kl_id == -1) {
                        if (kll_id==-1){
                            knei_id[cornerk] = -1;
                        }
                        else{
                            rot_dirk = j+1;
                            knei_id[cornerk] = index[rot_dirk*N + kll_id];
                        }
                    }
                }
                if (knei_id[7] == -1){
                    if(knei_id[1] == -1){
                        knei_id[8] = -1;
                    }
                    else{
                        knei_id[8] = index[4*N + knei_id[1]];
                    }
                }
                velocity_interpolation_jingchun(knei_id, knei_x, knei_y, knei_u, knei_v, xp1, yp1, up, vp, dx[0]);    
            }

            // 2nd runge-kutta, interpoltation particles' velocity - time
            for (int k=0; k<4; k++) {
                scalar_t kdxp = ark4[k] * kxn[k];
                scalar_t kdyp = ark4[k] * kyn[k];

                // cid for runge-kutta
                int32_t cid_new = c_id; 
        
                scalar_t xp1 = _xp + kdxp;
                scalar_t yp1 = _yp + kdyp;

                update_CellId(nei_id, nei_x, nei_y, xp1, yp1, cid_new, dx[0]);

                if (cid_new < 0){   //particles transported out of domain
                    up = un2[c_id];
                    vp = vn2[c_id];
                }
                else if (cid_new == c_id) { //particles cellid not changed
                    velocity_interpolation_jingchun(nei_id, nei_x, nei_y, nei_u, nei_v, xp1, yp1, up, vp, dx[0]);
                }
                else{   //particles transported into another cell
                    scalar_t uvp = sqrt( pow(up,2.) + pow(vp,2.) );
                    scalar_t uvp_cell = sqrt( pow(un2[cid_new],2.) + pow(vn2[cid_new],2.) );
                    if (uvp>=10e-6 && uvp_cell>=10e-6){
                        int32_t kr_id, krr_id;
                        int32_t knei_id[9];
                        scalar_t knei_x[9], knei_y[9], knei_u[9], knei_v[9];
                        int32_t rot_dirk;

                        knei_id[0] = cid_new;
                        knei_x[0] = x[cid_new];
                        knei_y[0] = y[cid_new];
                        knei_u[0] = un2[cid_new];
                        knei_v[0] = vn2[cid_new];

                        for (int j=0; j<4; j++){
                            kr_id = j + 1;
                            rot_dirk = kr_id  > 3 ? kr_id - 3 : kr_id + 1;
                        
                            kr_id = index[kr_id * N + cid_new];
                            krr_id = index[rot_dirk * N + kr_id];

                            knei_id[j*2+1] = kr_id;   //1,3,5,7
                            knei_id[j*2+2] = krr_id;  //2,4,6,8

                            knei_x[j*2+1] = x[kr_id];
                            knei_x[j*2+2] = x[krr_id];

                            knei_y[j*2+1] = y[kr_id];
                            knei_y[j*2+2] = y[krr_id];

                            knei_u[j*2+1] = un2[kr_id];
                            knei_u[j*2+2] = un2[krr_id];

                            knei_v[j*2+1] = vn2[kr_id];
                            knei_v[j*2+2] = vn2[krr_id];
                        }
                        
                        // find whether neighbor cell 2,4,6,8 is null
                        for (int j=0; j<3; j++){
                            int32_t kl_id = knei_id[j*2 + 1];
                            int32_t kll_id = knei_id[j*2 + 3];
                            int32_t cornerk = j*2 + 2;
                            if (kl_id == -1) {
                                if (kll_id==-1){
                                    knei_id[cornerk] = -1;
                                }
                                else{
                                    rot_dirk = j+1;
                                    knei_id[cornerk] = index[rot_dirk*N + kll_id];
                                }
                            }
                        }
                        if (knei_id[7] == -1){
                            if(knei_id[1] == -1){
                                knei_id[8] = -1;
                            }
                            else{
                                knei_id[8] = index[4*N + knei_id[1]];
                            }
                        }

                        velocity_interpolation_jingchun(knei_id, knei_x, knei_y, knei_u, knei_v, xp1, yp1, up, vp, dx[0]);    
                    }
                }

                kxn[k+1] = dt[0] * up;
                kyn[k+1] = dt[0] * vp;
            }

            scalar_t dxp2 = (kxn[1] + 2.0 * (kxn[2] + kxn[3]) + kxn[4]) / 6.0;
            scalar_t dyp2 = (kyn[1] + 2.0 * (kyn[2] + kyn[3]) + kyn[4]) / 6.0;

            _xp = _xp + (dxp1 + dxp2)/2.;
            _yp = _yp + (dyp1 + dyp2)/2.;
            
            update_CellId(nei_id, nei_x, nei_y, _xp, _yp, c_id, dx[0]);

            if(c_id< 0){
                c_id = -9999;
                _layer = -1;
                _xp = -9999;
                _yp = -9999;
            }

            // update particle positions and cellid
            xp[i] = _xp;
            yp[i] = _yp;
            cellid[i] = c_id;
            layer[i] = _layer;
        }
    }
}


void transport_cuda(at::Tensor index, at::Tensor x, at::Tensor y,
                    at::Tensor un1, at::Tensor vn1, 
                    at::Tensor u, at::Tensor v,
                    at::Tensor xp, at::Tensor yp,
                    at::Tensor cellid, at::Tensor layer, 
                    at::Tensor dx, at::Tensor dt) {
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = x.numel();
    const int PNN = cellid.numel();

    int thread_0 = 512;
    int block_0 = (PNN + 512 - 1) / 512;
    
    // AT_DISPATCH_FLOATING_TYPES(
    //     x.type(), "transport_cuda", ([&] {
    //         transport_kernel<scalar_t><<<block_0, thread_0, 0, stream>>>(
    //             N, PNN, index.data<int32_t>(), x.data<scalar_t>(), y.data<scalar_t>(), 
    //             u.data<scalar_t>(), v.data<scalar_t>(), 
    //             xp.data<scalar_t>(),yp.data<scalar_t>(), 
    //             cellid.data<int32_t>(), layer.data<int32_t>(), 
    //             dx.data<scalar_t>(), dt.data<scalar_t>());
    //         }));

    ////////////////////////////////////////////////////////// 2nd Runge-Kutta ////////////////////////////////////////////////
    AT_DISPATCH_FLOATING_TYPES(
        x.type(), "transport_cuda", ([&] {
            transport_kernel_test<scalar_t><<<block_0, thread_0, 0, stream>>>(
                N, PNN, index.data<int32_t>(), x.data<scalar_t>(), y.data<scalar_t>(), 
                un1.data<scalar_t>(), vn1.data<scalar_t>(), 
                u.data<scalar_t>(), v.data<scalar_t>(), 
                xp.data<scalar_t>(),yp.data<scalar_t>(), 
                cellid.data<int32_t>(), layer.data<int32_t>(), 
                dx.data<scalar_t>(), dt.data<scalar_t>());
            }));
    //////////////////////////////////////////////////////////       End      ////////////////////////////////////////////////

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}

