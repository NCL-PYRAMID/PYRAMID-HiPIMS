#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void parInit_cuda(at::Tensor x, 
                    at::Tensor y, 
                    at::Tensor dx,
                    at::Tensor Ms_cum, 
                    at::Tensor Mg_cum,
                    at::Tensor cellid, 
                    at::Tensor xp, 
                    at::Tensor yp, 
                    at::Tensor layer,
                    at::Tensor particlesIncell,
                    at::Tensor np_max);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void parInit(at::Tensor x, 
                at::Tensor y, 
                at::Tensor dx,
                at::Tensor Ms_cum, 
                at::Tensor Mg_cum,
                at::Tensor cellid, 
                at::Tensor xp, 
                at::Tensor yp,
                at::Tensor layer,
                at::Tensor particlesIncell,
                at::Tensor np_max) {
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  CHECK_INPUT(dx);
  CHECK_INPUT(Ms_cum);
  CHECK_INPUT(Mg_cum);
  CHECK_INPUT(cellid);
  CHECK_INPUT(xp);
  CHECK_INPUT(yp);
  CHECK_INPUT(layer);
  CHECK_INPUT(particlesIncell);
  CHECK_INPUT(np_max);

  parInit_cuda(x, y, dx, Ms_cum, Mg_cum, cellid, xp, yp, layer, particlesIncell, np_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("initParticles", &parInit,
        "Particles Initializing, CUDA version");
}