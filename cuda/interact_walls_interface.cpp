#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
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
							at::Tensor neloc_y_device);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void interact_walls(at::Tensor walls,
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
      
  CHECK_INPUT(walls);
  CHECK_INPUT(mu);
  CHECK_INPUT(g_properties);
  CHECK_INPUT(g_gne_device);
  CHECK_INPUT(g_gbar_device);
  CHECK_INPUT(g_glay_device);
  CHECK_INPUT(g_gR_device);
  CHECK_INPUT(g_gvx_device);
  CHECK_INPUT(g_gvy_device);
  CHECK_INPUT(g_gfx_device);
  CHECK_INPUT(g_gfy_device);
  CHECK_INPUT(g_gp_device);
  CHECK_INPUT(g_gt_device);
  CHECK_INPUT(neloc_x_device);
  CHECK_INPUT(neloc_y_device);

  interact_walls_cuda(walls, mu, g_properties, g_gne_device, g_gbar_device, g_glay_device, 
							g_gR_device, g_gvx_device, g_gvy_device, 
							g_gfx_device, g_gfy_device,
							g_gp_device, g_gt_device, 
							neloc_x_device, neloc_y_device);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &interact_walls,
        "calculate interaction between debris and walls, CUDA version");
}