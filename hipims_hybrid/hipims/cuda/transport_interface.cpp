#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
void transport_cuda(at::Tensor index, at::Tensor x, at::Tensor y, 
                    at::Tensor un1, at::Tensor vn1,
                    at::Tensor u, at::Tensor v,
                    at::Tensor xp, at::Tensor yp,
                    at::Tensor cellid, at::Tensor layer, 
                    at::Tensor dx, at::Tensor dt);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void PTM(at::Tensor index, at::Tensor x, at::Tensor y,
                    at::Tensor un1, at::Tensor vn1, 
                    at::Tensor u, at::Tensor v,
                    at::Tensor xp, at::Tensor yp,
                    at::Tensor cellid, at::Tensor layer,
                    at::Tensor dx, at::Tensor dt) {
  CHECK_INPUT(index);
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  CHECK_INPUT(un1);
  CHECK_INPUT(vn1);
  CHECK_INPUT(u);
  CHECK_INPUT(v);
  CHECK_INPUT(xp);
  CHECK_INPUT(yp);
  CHECK_INPUT(cellid);
  CHECK_INPUT(layer);
  CHECK_INPUT(dx);
  CHECK_INPUT(dt);

  transport_cuda(index, x, y, un1, vn1, u, v, xp, yp, cellid, layer, dx, dt);
  // transport_cuda(index, x, y, u, v, xp, yp, cellid, layer, dx, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("particle_tracking", &PTM,
        "Particles Transporting, CUDA version");
}