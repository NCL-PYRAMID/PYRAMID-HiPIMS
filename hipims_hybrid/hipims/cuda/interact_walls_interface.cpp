#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
void interact_walls_cuda(at::Tensor wleft, at::Tensor wright, at::Tensor wup, at::Tensor wdown, 
							at::Tensor wp_up, at::Tensor wp_down, at::Tensor wp_left, at::Tensor wp_right,
							at::Tensor mu, at::Tensor g_gkn_device, at::Tensor g_gnu_device, at::Tensor g_grho_device, at::Tensor g_gkt_device,
							at::Tensor dt_g, at::Tensor g_gne_device, at::Tensor g_gbar_device, at::Tensor g_glay_device, at::Tensor g_gR_device,
							at::Tensor g_gx_device, at::Tensor g_gy_device, at::Tensor g_gvx_device, at::Tensor g_gvy_device,
							at::Tensor g_gax_device, at::Tensor g_gay_device, at::Tensor g_gang_device, at::Tensor g_gangv_device, at::Tensor g_ganga_device,
							at::Tensor g_gfx_device, at::Tensor g_gfy_device, at::Tensor g_gfz_device, at::Tensor g_gp_device, at::Tensor g_gt_device,
							at::Tensor neloc_x_device, at::Tensor neloc_y_device, at::Tensor t);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void interact_walls(at::Tensor wleft, at::Tensor wright, at::Tensor wup, at::Tensor wdown, 
                        at::Tensor wp_up, at::Tensor wp_down, at::Tensor wp_left, at::Tensor wp_right,
                        at::Tensor mu, at::Tensor g_gkn_device, at::Tensor g_gnu_device, at::Tensor g_grho_device, at::Tensor g_gkt_device,
                        at::Tensor dt_g, at::Tensor g_gne_device, at::Tensor g_gbar_device, at::Tensor g_glay_device, at::Tensor g_gR_device,
                        at::Tensor g_gx_device, at::Tensor g_gy_device, at::Tensor g_gvx_device, at::Tensor g_gvy_device,
                        at::Tensor g_gax_device, at::Tensor g_gay_device, at::Tensor g_gang_device, at::Tensor g_gangv_device, at::Tensor g_ganga_device,
                        at::Tensor g_gfx_device, at::Tensor g_gfy_device, at::Tensor g_gfz_device, at::Tensor g_gp_device, at::Tensor g_gt_device,
                        at::Tensor neloc_x_device, at::Tensor neloc_y_device, at::Tensor t) {
  CHECK_INPUT(wleft);
  CHECK_INPUT(wright);
  CHECK_INPUT(wup);
  CHECK_INPUT(wdown);
  CHECK_INPUT(wp_up);
  CHECK_INPUT(wp_down);
  CHECK_INPUT(wp_left);
  CHECK_INPUT(wp_right);
  CHECK_INPUT(mu);
  CHECK_INPUT(g_gkn_device);
  CHECK_INPUT(g_gnu_device);
  CHECK_INPUT(g_grho_device);
  CHECK_INPUT(g_gkt_device);
  CHECK_INPUT(dt_g);
  CHECK_INPUT(g_gne_device);
  CHECK_INPUT(g_gbar_device);
  CHECK_INPUT(g_glay_device);
  CHECK_INPUT(g_gR_device);
  CHECK_INPUT(g_gx_device);
  CHECK_INPUT(g_gy_device);
  CHECK_INPUT(g_gvx_device);
  CHECK_INPUT(g_gvy_device);
  CHECK_INPUT(g_gax_device);
  CHECK_INPUT(g_gay_device);
  CHECK_INPUT(g_gang_device);
  CHECK_INPUT(g_gangv_device);
  CHECK_INPUT(g_ganga_device);
  CHECK_INPUT(g_gfx_device);
  CHECK_INPUT(g_gfy_device);
  CHECK_INPUT(g_gfz_device);
  CHECK_INPUT(g_gp_device);
  CHECK_INPUT(g_gt_device);
  CHECK_INPUT(neloc_x_device);
  CHECK_INPUT(neloc_y_device);
  CHECK_INPUT(t);

  interact_walls_cuda(wleft, wright, wup, wdown, 
                        wp_up, wp_down, wp_left, wp_right,
                        mu, g_gkn_device, g_gnu_device, g_grho_device, g_gkt_device,
                        dt_g, g_gne_device, g_gbar_device, g_glay_device, g_gR_device,
                        g_gx_device, g_gy_device, g_gvx_device, g_gvy_device,
                        g_gax_device, g_gay_device, g_gang_device, g_gangv_device, g_ganga_device,
                        g_gfx_device, g_gfy_device, g_gfz_device, g_gp_device, g_gt_device,
                        neloc_x_device, neloc_y_device, t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &interact_walls,
        "calculate interaction between debris and walls, CUDA version");
}