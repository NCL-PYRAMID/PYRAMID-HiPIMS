#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
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
	at::Tensor cell_f_yN_device);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void loc_debris(at::Tensor dx, at::Tensor maskIndex, 
                  at::Tensor num_x_device,  at::Tensor num_y_device,
                  at::Tensor zb_device,  at::Tensor z_device, at::Tensor qx_device,  at::Tensor qy_device, 
                  at::Tensor z_rela, at::Tensor r_rela, at::Tensor lay_sub, 
                  at::Tensor g_gne_device, at::Tensor g_gbar_device, at::Tensor g_glay_device, 
                  at::Tensor g_gR_device, at::Tensor g_gp_device, at::Tensor g_gt_device,
                  at::Tensor g_gm_device, at::Tensor g_gz_device, 
                  at::Tensor g_gx_device, at::Tensor g_gy_device, 
                  at::Tensor neloc_x_device, at::Tensor neloc_y_device,
                  at::Tensor mu, at::Tensor g_properties, 
                  at::Tensor g_gv_device, at::Tensor g_gf_device,
                  at::Tensor qx_g, at::Tensor qy_g, at::Tensor h_g, 
                  at::Tensor z_xf, at::Tensor h_xf, 
                  at::Tensor z_yf, at::Tensor h_yf, 
                  at::Tensor z_xb, at::Tensor h_xb,
                  at::Tensor z_yb, at::Tensor h_yb,
                  at::Tensor h_xff, at::Tensor h_yff, 
                  at::Tensor h_xbb,at::Tensor h_ybb, 
                  at::Tensor t_dem, at::Tensor cell_f_xP_device, 
                  at::Tensor cell_f_xN_device, at::Tensor cell_f_yP_device, at::Tensor cell_f_yN_device) {
                    
  CHECK_INPUT(dx);
  CHECK_INPUT(maskIndex);
  CHECK_INPUT(num_x_device);
  CHECK_INPUT(num_y_device);
  CHECK_INPUT(zb_device);
  CHECK_INPUT(z_device);
  CHECK_INPUT(qx_device);
  CHECK_INPUT(qy_device);
  CHECK_INPUT(z_rela);
  CHECK_INPUT(r_rela);
  CHECK_INPUT(lay_sub);
  CHECK_INPUT(g_gne_device);
  CHECK_INPUT(g_gbar_device);
  CHECK_INPUT(g_glay_device);
  CHECK_INPUT(g_gR_device);
  CHECK_INPUT(g_gp_device);
	CHECK_INPUT(g_gt_device);
  CHECK_INPUT(g_gm_device);
  CHECK_INPUT(g_gz_device);
  CHECK_INPUT(g_gx_device);
  CHECK_INPUT(g_gy_device);
  CHECK_INPUT(neloc_x_device);
  CHECK_INPUT(neloc_y_device);
  CHECK_INPUT(mu);
  CHECK_INPUT(g_properties);
  CHECK_INPUT(g_gv_device);
  CHECK_INPUT(g_gf_device);
  CHECK_INPUT(qx_g);
  CHECK_INPUT(qy_g);
  CHECK_INPUT(h_g);
  CHECK_INPUT(z_xf);
  CHECK_INPUT(h_xf);
  CHECK_INPUT(z_yf);
  CHECK_INPUT(h_yf);
  CHECK_INPUT(z_xb);
  CHECK_INPUT(h_xb);
  CHECK_INPUT(z_yb);
  CHECK_INPUT(h_yb);
  CHECK_INPUT(h_xff);
  CHECK_INPUT(h_yff);
  CHECK_INPUT(h_xbb);
  CHECK_INPUT(h_ybb);
  CHECK_INPUT(t_dem);
  CHECK_INPUT(cell_f_xP_device);
  CHECK_INPUT(cell_f_xN_device);
  CHECK_INPUT(cell_f_yP_device);
  CHECK_INPUT(cell_f_yN_device);

  loc_debris_cuda(dx, 
                  maskIndex, 
                  num_x_device, 
                  num_y_device,
                  zb_device, 
                  z_device, 
                  qx_device, 
                  qy_device, 
                  z_rela, 
                  r_rela, 
                  lay_sub, 
                  g_gne_device, 
                  g_gbar_device, 
                  g_glay_device, 
                  g_gR_device, 
                  g_gp_device,
	                g_gt_device,
                  g_gm_device, 
                  g_gz_device, 
                  g_gx_device, 
                  g_gy_device, 
                  neloc_x_device, 
                  neloc_y_device,
                  mu, 
                  g_properties, 
                  g_gv_device, 
                  g_gf_device,
                  qx_g, 
                  qy_g, 
                  h_g, 
                  z_xf, 
                  h_xf, 
                  z_yf,
                  h_yf, 
                  z_xb, 
                  h_xb,
                  z_yb, 
                  h_yb,
                  h_xff, 
                  h_yff, 
                  h_xbb,
                  h_ybb, 
                  t_dem, 
                  cell_f_xP_device, 
                  cell_f_xN_device, 
                  cell_f_yP_device, 
                  cell_f_yN_device);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &loc_debris,
        "calculate interaction between debris and walls, CUDA version");
}