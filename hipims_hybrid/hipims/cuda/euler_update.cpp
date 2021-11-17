#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void euler_update_cuda(at::Tensor updateMask, at::Tensor h_update,
                       at::Tensor qx_update, at::Tensor qy_update,
                       at::Tensor z_update, at::Tensor h, at::Tensor wl,
                       at::Tensor z, at::Tensor qx, at::Tensor qy);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void euler_update(at::Tensor updateMask, at::Tensor h_update,
                  at::Tensor qx_update, at::Tensor qy_update,
                  at::Tensor z_update, at::Tensor h, at::Tensor wl,
                  at::Tensor z, at::Tensor qx, at::Tensor qy) {
  CHECK_INPUT(updateMask);
  CHECK_INPUT(h);
  CHECK_INPUT(qx);
  CHECK_INPUT(qy);
  CHECK_INPUT(wl);
  CHECK_INPUT(z);
  CHECK_INPUT(qx_update);
  CHECK_INPUT(qy_update);
  CHECK_INPUT(h_update);
  CHECK_INPUT(z_update);

  euler_update_cuda(updateMask, h_update, qx_update, qy_update, z_update, h, wl,
                    z, qx, qy);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &euler_update, "Friction Updating, CUDA version");
}