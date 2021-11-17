#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void element_location_update_cuda(at::Tensor eMask, at::Tensor ex, at::Tensor ey,
                       at::Tensor gx, at::Tensor gy,
                       at::Tensor gang, at::Tensor gR, 
                       at::Tensor ne, at::Tensor bar);
 


// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void element_location_update(at::Tensor eMask, at::Tensor ex, at::Tensor ey,
                       at::Tensor gx, at::Tensor gy,
                       at::Tensor gang, at::Tensor gR, 
                       at::Tensor ne, at::Tensor bar) {
  CHECK_INPUT(eMask);
  CHECK_INPUT(ex);
  CHECK_INPUT(ey);
  CHECK_INPUT(gx);
  CHECK_INPUT(gy);
  CHECK_INPUT(gang);
  CHECK_INPUT(gR);
  CHECK_INPUT(ne);
  CHECK_INPUT(bar);

  element_location_update_cuda(eMask, ex, ey, gx, gy, gang, gR, ne, bar);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &element_location_update, "Element Location Updating, CUDA version");
}