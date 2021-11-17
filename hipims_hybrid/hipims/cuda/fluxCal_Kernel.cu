#include "gpu.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>
// #include <c10/cuda/CUDACachingAllocator.h>

// enum FIELD_TYPE { H = 0, WL = 1, QX = 2, QY = 3, Z = 4 };
// enum BOUNDARY_TYPE = {WALL_NON_SLIP = 3, WALL_SLIP = 4, OPEN = 5, HQ_GIVEN =
// 6};
// field
// 0: h, 1: wl, 2: qx, 3: qy, 4: z
// #define H 0
// #define WL 1
// #define QX 2
// #define QY 3
// #define Z 4

// // boundary type
// #define WALL_NON_SLIP 3
// #define WALL_SLIP 4
// #define OPEN 5
// #define HQ_GIVEN 6

namespace {

// update:
// the field_c, field_l, field_r are chose to represent the central, left and
// right field values

template <typename scalar_t>
__device__ __forceinline__ void
hydroStaticReconstruction(scalar_t *field_l, scalar_t *field_r,
                          const scalar_t h_small) {
  auto z_bm = max(field_l[1] - field_l[0], field_r[1] - field_r[0]);
  z_bm = min(field_l[1], z_bm);

  auto ux_l = field_l[0] < h_small ? 0.0 : field_l[2] / field_l[0];
  auto uy_l = field_l[0] < h_small ? 0.0 : field_l[3] / field_l[0];

  auto ux_r = field_r[0] < h_small ? 0.0 : field_r[2] / field_r[0];
  auto uy_r = field_r[0] < h_small ? 0.0 : field_r[3] / field_r[0];

  field_l[0] = field_l[1] - z_bm;
  field_r[0] = max(0.0, field_r[1] - z_bm) - max(0.0, field_r[4] - z_bm);

  field_l[2] = ux_l * field_l[0];
  field_r[2] = ux_r * field_r[0];

  field_l[3] = uy_l * field_l[0];
  field_r[3] = uy_r * field_r[0];

  field_l[4] = z_bm;
  field_r[4] = z_bm;

  field_l[1] = z_bm + field_l[0];
  field_r[1] = z_bm + field_r[0];
}

template <typename scalar_t>
__device__ __forceinline__ void
hllcFluxSolver(scalar_t *field_c, scalar_t *field_l, scalar_t *field_r,
               const scalar_t *normal, const scalar_t *tangential,
               scalar_t &flux_h, scalar_t &flux_qx, scalar_t &flux_qy,
               const scalar_t h_small, const scalar_t g, const scalar_t dx,
               const scalar_t dt) {
  auto ux_l = field_l[0] < h_small ? 0.0 : field_l[2] / field_l[0];
  auto uy_l = field_l[0] < h_small ? 0.0 : field_l[3] / field_l[0];

  auto ux_r = field_r[0] < h_small ? 0.0 : field_r[2] / field_r[0];
  auto uy_r = field_r[0] < h_small ? 0.0 : field_r[3] / field_r[0];
  hydroStaticReconstruction(field_l, field_r, h_small);

  if (field_l[0] < h_small && field_r[0] < h_small) {
    flux_qx -= ((0.5 * g * (field_l[0] + field_c[0]) *
                 (field_l[4] - field_c[4]) * normal[0]) /
                dx) *
               dt;
    flux_qy -= ((0.5 * g * (field_l[0] + field_c[0]) *
                 (field_l[4] - field_c[4]) * normal[1]) /
                dx) *
               dt;
  } else {
    auto u_L = ux_l * normal[0] + uy_l * normal[1];
    auto u_R = ux_r * normal[0] + uy_r * normal[1];

    auto v_L = -ux_l * normal[1] + uy_l * normal[0];
    auto v_R = -ux_r * normal[1] + uy_r * normal[0];

    auto qx_L = u_L * field_l[0]; // acturally the normal discharge
    auto qy_L = v_L * field_l[0]; // acturally the tan discharge
    auto qx_R = u_R * field_r[0];
    auto qy_R = v_R * field_r[0];

    scalar_t a_L = sqrt(g * field_l[0]);
    scalar_t a_R = sqrt(g * field_r[0]);

    scalar_t h_star = pow((a_L + a_R) / 2.0 + (u_L - u_R) / 4.0, 2.0) / g;
    scalar_t u_star = (u_L + u_R) / 2.0 + a_L - a_R;
    scalar_t a_star = sqrt(g * h_star);

    scalar_t s_L, s_R;

    s_L = field_l[0] <= h_small ? u_R - 2.0 * a_R
                                : min(u_L - a_L, u_star - a_star);
    s_R = field_r[0] <= h_small ? u_L + 2.0 * a_L
                                : max(u_R + a_R, u_star + a_star);

    scalar_t s_M =
        (s_L * field_r[0] * (u_R - s_R) - s_R * field_l[0] * (u_L - s_L)) /
        (field_r[0] * (u_R - s_R) - field_l[0] * (u_L - s_L));

    scalar_t h_flux_L = qx_L;
    scalar_t qx_flux_L = u_L * qx_L + 0.5 * g * field_l[0] * field_l[0];
    scalar_t qy_flux_L = u_L * qy_L;

    scalar_t h_flux_R = qx_R;
    scalar_t qx_flux_R = u_R * qx_R + 0.5 * g * field_r[0] * field_r[0];
    scalar_t qy_flux_R = u_R * qy_R;

    scalar_t h_flux_M = (s_R * h_flux_L - s_L * h_flux_R +
                         s_L * s_R * (field_r[0] - field_l[0])) /
                        (s_R - s_L);
    scalar_t qx_flux_M =
        (s_R * qx_flux_L - s_L * qx_flux_R + s_L * s_R * (qx_R - qx_L)) /
        (s_R - s_L);

    scalar_t h_flux, qx_flux, qy_flux;
    if (0.0 <= s_L) {
      h_flux = h_flux_L;
      qx_flux = qx_flux_L;
      qy_flux = qy_flux_L;
    } else if (s_L < 0.0 && 0.0 <= s_M) {
      h_flux = h_flux_M;
      qx_flux = qx_flux_M;
      qy_flux = h_flux_M * v_L;
    } else if (s_M < 0.0 && 0.0 <= s_R) {
      h_flux = h_flux_M;
      qx_flux = qx_flux_M;
      qy_flux = h_flux_M * v_R;
    } else {
      h_flux = h_flux_R;
      qx_flux = qx_flux_R;
      qy_flux = qy_flux_R;
    }

    flux_h -= (h_flux / dx) * dt;
    flux_qx -= ((qx_flux * normal[0] + qy_flux * tangential[0] +
                 0.5 * g * (field_l[0] + field_c[0]) *
                     (field_l[4] - field_c[4]) * normal[0]) /
                dx) *
               dt;
    flux_qy -= ((qx_flux * normal[1] + qy_flux * tangential[1] +
                 0.5 * g * (field_l[0] + field_c[0]) *
                     (field_l[4] - field_c[4]) * normal[1]) /
                dx) *
               dt;
  }
}

template <typename scalar_t>
__device__ __forceinline__ void boundary_WallNonSlip(scalar_t *field,
                                                     scalar_t *field_b) {
  field_b[0] = field[0];
  field_b[1] = field[1];
  field_b[4] = field[4];
  field_b[2] = 0.0;
  field_b[3] = 0.0;
}

template <typename scalar_t>
__device__ __forceinline__ void
boundary_WallSlip(scalar_t *field, scalar_t *field_b, const scalar_t *normal) {
  field_b[0] = field[0];
  field_b[1] = field[1];
  field_b[4] = field[4];
  // field_b[2] = (normal[1] * normal[1] - normal[0] * normal[0]) * field[2];
  // field_b[3] = (normal[0] * normal[0] - normal[1] * normal[1]) * field[3];
  field_b[2] = (normal[0] * normal[0] - normal[1] * normal[1]) * field[2];
  field_b[3] = (normal[1] * normal[1] - normal[0] * normal[0]) * field[3];
}

template <typename scalar_t>
__device__ __forceinline__ void boundary_Open(scalar_t *field,
                                              scalar_t *field_b) {
  field_b[0] = field[0];
  field_b[1] = field[1];
  field_b[4] = field[4];
  field_b[2] = field[2];
  field_b[3] = field[3];
}

template <typename scalar_t>
__device__ __forceinline__ void
boundary_HQ_given(scalar_t *field, scalar_t *field_b, const scalar_t *normal,
                  const scalar_t given_depth, const scalar_t given_qx,
                  const scalar_t given_qy) {
  field_b[0] = given_depth;
  field_b[4] = field[4];
  field_b[1] = field_b[4] + field_b[0];
  field_b[2] = given_qx;
  field_b[3] = given_qy;
}

template <typename scalar_t>
__device__ __forceinline__ void
boundary_conditions(const int boundaryType, scalar_t *field, scalar_t *field_b,
                    const scalar_t *normal, const scalar_t given_depth,
                    const scalar_t given_qx, const scalar_t given_qy) {
  switch (boundaryType) {
  case 3: {
    boundary_WallNonSlip(field, field_b);
    break;
  }
  case 4: {
    boundary_WallSlip(field, field_b, normal);
    break;
  }
  case 5: {
    boundary_Open(field, field_b);
    break;
  }
  case 6: {
    boundary_HQ_given(field, field_b, normal, given_depth, given_qx, given_qy);
    break;
  }
  }
}

template <typename scalar_t>
__device__ __forceinline__ void
MUSCL_reconstruction(scalar_t *field_c, scalar_t *field_l, scalar_t *field_r,
                     scalar_t *field_edge, const scalar_t h_small) {
  for (int i = 0; i < 4; i++) {
    field_edge[i] =
        abs(field_r[i] - field_c[i]) > h_small
            ? field_c[i] +
                  0.5 * max(0.0, min(1.0, (field_c[i] - field_l[i]) /
                                              (field_r[i] - field_c[i]))) *
                      (field_r[i] - field_c[i])
            : field_c[i];
  }
  field_edge[4] = field_edge[1] - field_edge[0];
}

template <typename scalar_t>
__global__ void fluxCalculation_kernel(
    const int N, const int depth_w, const int depth_h, const int discharge_w,
    const int discharge_h, scalar_t *__restrict__ h, scalar_t *__restrict__ wl,
    scalar_t *__restrict__ z, scalar_t *__restrict__ qx,
    scalar_t *__restrict__ qy, const int32_t *__restrict__ index,
    const scalar_t *__restrict__ normal, scalar_t *__restrict__ given_depth,
    scalar_t *__restrict__ given_discharge, scalar_t *__restrict__ dx,
    scalar_t *__restrict__ t, scalar_t *__restrict__ dt,
    scalar_t *__restrict__ h_flux, scalar_t *__restrict__ qx_flux,
    scalar_t *__restrict__ qy_flux) {

  scalar_t h_small = 1.0e-6;
  scalar_t g = 9.81;
  int timelevel_depth = 0;
  int timelevel_discharge = 0;

  while ((t[0] > given_depth[timelevel_depth * depth_w]) &&
         (timelevel_depth < depth_h - 1)) {
    timelevel_depth++;
  }
  while ((t[0] > given_discharge[timelevel_discharge * discharge_w]) &&
         (timelevel_discharge < discharge_h - 1)) {
    timelevel_discharge++;
  }

  scalar_t field_c[5];
  scalar_t field_l[5];
  scalar_t field_r[5];
  scalar_t field_rr[5];
  scalar_t field_edge_l[5];
  scalar_t field_edge_r[5];
  scalar_t _normal[2];
  scalar_t _tangential[2];

  scalar_t normal_l[2];

  int32_t c_type, c_id, r_id, rr_id, l_id;
  int32_t boundaryType, boundaryID;
  int n;
  scalar_t h_boundary, qx_boundary, qy_boundary;

  // get the index of cell

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    c_type = index[i];
    c_id = i;

    h_flux[i] = 0.0;
    qx_flux[i] = 0.0;
    qy_flux[i] = 0.0;

    field_c[0] = h[c_id];
    field_c[1] = wl[c_id];
    field_c[2] = qx[c_id];
    field_c[3] = qy[c_id];
    field_c[4] = z[c_id];

    for (int j = 0; j < 4; j++) {
      // _normal[2] = {normal[j * 2], normal[j * 2 + 1]};
      // _tangential[2] = {-_normal[1], _normal[0]};
      _normal[0] = normal[j * 2];
      _normal[1] = normal[j * 2 + 1];

      _tangential[0] = -_normal[1];
      _tangential[1] = _normal[0];

      /**
      index structure:
      1~8: the index of neighboring cells
      0: the property of considering cell-> internal cell will be 10 or sth,
      boundary cell will be 3**, 4**, 5**, 6**
                        6
                        2
                7 | 3 | 0 | 1 | 5
                        4
                        8
      */

      r_id = j + 1;
      rr_id = r_id + 4;
      l_id = r_id + 2 > 4 ? r_id - 2 : r_id + 2;

      // begin to get the id of neighbor cells
      r_id = index[r_id * N + c_id];
      rr_id = index[rr_id * N + c_id];
      // get the neighbor boundary id (types)
      l_id = index[l_id * N + c_id];
      boundaryType = c_type;
      boundaryID = c_type;
      if (r_id == -1) {
        boundaryType = c_type;
        boundaryID = c_type;
        n = 0;
        while (boundaryType >= 10) {
          boundaryType = (boundaryType - (boundaryType % 10)) / 10;
          n++;
        }
        boundaryID = boundaryID - boundaryType * powf(10, n);
        // boundaryID = index[r_id] - boundaryType * exp10f(n);

        h_boundary = given_depth[timelevel_depth * depth_w + boundaryID + 1];
        qx_boundary = given_discharge[timelevel_discharge * discharge_w + 1 +
                                      2 * boundaryID];
        qy_boundary = given_discharge[timelevel_discharge * discharge_w + 2 +
                                      2 * boundaryID];
        boundary_conditions(boundaryType, field_c, field_r, _normal, h_boundary,
                            qx_boundary, qy_boundary);
        for (int v_i = 0; v_i < 5; v_i++) {
          field_rr[v_i] = field_r[v_i];
        }
      } else {
        field_r[0] = h[r_id];
        field_r[1] = wl[r_id];
        field_r[2] = qx[r_id];
        field_r[3] = qy[r_id];
        field_r[4] = z[r_id];

        if (rr_id == -1) {
          boundaryType = index[r_id];
          boundaryID = index[r_id];
          n = 0;
          while (boundaryType >= 10) {
            boundaryType = (boundaryType - (boundaryType % 10)) / 10;
            n++;
          }
          // boundaryID = index[r_id] - boundaryType * powf(10, n);
          boundaryID = boundaryID - boundaryType * exp10f(n);

          h_boundary = given_depth[timelevel_depth * depth_w + boundaryID + 1];
          qx_boundary = given_discharge[timelevel_discharge * discharge_w + 1 +
                                        2 * boundaryID];
          qy_boundary = given_discharge[timelevel_discharge * discharge_w + 2 +
                                        2 * boundaryID];
          boundary_conditions(boundaryType, field_r, field_rr, _normal,
                              h_boundary, qx_boundary, qy_boundary);

        } else {
          field_rr[0] = h[rr_id];
          field_rr[1] = wl[rr_id];
          field_rr[2] = qx[rr_id];
          field_rr[3] = qy[rr_id];
          field_rr[4] = z[rr_id];
        }
      }

      if (l_id == -1) {
        boundaryType = c_type;
        boundaryID = c_type;
        n = 0;
        while (boundaryType >= 10) {
          boundaryType = (boundaryType - (boundaryType % 10)) / 10;
          n++;
        }
        // boundaryID = c_type - boundaryType * powf(10, n);
        boundaryID = boundaryID - boundaryType * exp10f(n);

        normal_l[0] = -_normal[0];
        normal_l[1] = -_normal[1];
        h_boundary = given_depth[timelevel_depth * depth_w + boundaryID + 1];
        qx_boundary = given_discharge[timelevel_discharge * discharge_w + 1 +
                                      2 * boundaryID];
        qy_boundary = given_discharge[timelevel_discharge * discharge_w + 2 +
                                      2 * boundaryID];
        boundary_conditions(boundaryType, field_c, field_l, normal_l,
                            h_boundary, qx_boundary, qy_boundary);
      } else {
        field_l[0] = h[l_id];
        field_l[1] = wl[l_id];
        field_l[2] = qx[l_id];
        field_l[3] = qy[l_id];
        field_l[4] = z[l_id];
      }

      MUSCL_reconstruction(field_c, field_l, field_r, field_edge_l, h_small);
      MUSCL_reconstruction(field_r, field_rr, field_c, field_edge_r, h_small);

      hllcFluxSolver(field_c, field_edge_l, field_edge_r, _normal, _tangential,
                     h_flux[i], qx_flux[i], qy_flux[i], h_small, g, dx[0],
                     dt[0]);
    }
  }
  // here we may need to synchronous the thread
  // __syncthreads();
  // // update
  // if (i < N) {
  //   h[i] -= (h_flux[i] * dt[0]);
  //   qx[i] -= (qx_flux[i] * dt[0]);
  //   qy[i] -= (qy_flux[i] * dt[0]);
  //   wl[i] = h[i] + z[i];
  // }
}
}

void fluxCalculation_cuda(at::Tensor h_flux, at::Tensor qx_flux,
                          at::Tensor qy_flux, torch::Tensor h, torch::Tensor wl,
                          torch::Tensor z, torch::Tensor qx, torch::Tensor qy,
                          torch::Tensor index, torch::Tensor normal,
                          torch::Tensor given_depth,
                          torch::Tensor given_discharge, torch::Tensor dx,
                          torch::Tensor t, torch::Tensor dt) {
  at::cuda::CUDAGuard device_guard(h.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // auto h_flux = torch::zeros_like(h);
  // auto qx_flux = torch::zeros_like(qx);
  // auto qy_flux = torch::zeros_like(qy);
  const int N = h.size(0);

  const int depth_h = given_depth.size(0);
  const int depth_w = given_depth.size(1);
  const int discharge_h = given_discharge.size(0);
  const int discharge_w = given_discharge.size(1);

  int thread = 512;
  int block = (N + 512 - 1) / 512;

  AT_DISPATCH_FLOATING_TYPES(
      h.type(), "fluxCalculation_cuda", ([&] {
        fluxCalculation_kernel<scalar_t><<<block, thread, 0, stream>>>(
            N, depth_w, depth_h, discharge_w, discharge_h, h.data<scalar_t>(),
            wl.data<scalar_t>(), z.data<scalar_t>(), qx.data<scalar_t>(),
            qy.data<scalar_t>(), index.data<int32_t>(), normal.data<scalar_t>(),
            given_depth.data<scalar_t>(), given_discharge.data<scalar_t>(),
            dx.data<scalar_t>(), t.data<scalar_t>(), dt.data<scalar_t>(),
            h_flux.data<scalar_t>(), qx_flux.data<scalar_t>(),
            qy_flux.data<scalar_t>());
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error in load_textures: %s\n", cudaGetErrorString(err));
}
