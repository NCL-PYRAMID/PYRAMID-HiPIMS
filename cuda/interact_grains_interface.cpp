#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
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
        at::Tensor ey);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void interact_grains(at::Tensor mu, 
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

    CHECK_INPUT(mu);            
    CHECK_INPUT(g_properties);
    CHECK_INPUT(dt_g);
    CHECK_INPUT(ne);
    CHECK_INPUT(bar);
    CHECK_INPUT(lay);
    CHECK_INPUT(gR);
    CHECK_INPUT(gvx);
    CHECK_INPUT(gvy);
    CHECK_INPUT(gang);
    CHECK_INPUT(gangv);
    CHECK_INPUT(gfx);
    CHECK_INPUT(gfy);
    CHECK_INPUT(gp);
    CHECK_INPUT(gt);
    CHECK_INPUT(ex);
    CHECK_INPUT(ey);

    interact_grains_cuda(mu, g_properties,
							dt_g, ne, bar, lay, gR,
							gvx, gvy, gang, gangv,
							gfx, gfy, gp, gt, ex, ey);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &interact_grains,
        "Calculate interact of grains, CUDA version");
}