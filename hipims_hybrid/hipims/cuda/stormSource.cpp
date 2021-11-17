#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void Storm_cuda(at::Tensor wetMask, at::Tensor x, at::Tensor y, at::Tensor h,
                at::Tensor wl, at::Tensor qx, at::Tensor qy, at::Tensor z,
                at::Tensor stormData, at::Tensor dt);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void stormSource(at::Tensor wetMask, at::Tensor x, at::Tensor y, at::Tensor h,
                 at::Tensor wl, at::Tensor qx, at::Tensor qy, at::Tensor z,
                 at::Tensor stormData, at::Tensor dt) {
  CHECK_INPUT(wetMask);
  CHECK_INPUT(h);
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  CHECK_INPUT(qx);
  CHECK_INPUT(z);
  CHECK_INPUT(wl);
  CHECK_INPUT(qy);
  CHECK_INPUT(dt);
  CHECK_INPUT(stormData);

  Storm_cuda(wetMask, x, y, h, wl, qx, qy, z, stormData, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("addStormFriction", &stormSource,
        "Storm Friction Updating, CUDA version");
}