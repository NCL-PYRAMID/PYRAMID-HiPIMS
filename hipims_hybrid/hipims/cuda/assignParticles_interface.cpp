#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
void assignPar_cuda(at::Tensor dMMask, at::Tensor dM_num, at::Tensor pid_unassigned,
                    at::Tensor x, at::Tensor y, 
                    at::Tensor xp, at::Tensor yp,
                    at::Tensor cellid, at::Tensor layer);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void assignPar(at::Tensor dMMask, at::Tensor dM_num, at::Tensor pid_unassigned,
                    at::Tensor x, at::Tensor y, 
                    at::Tensor xp, at::Tensor yp,
                    at::Tensor cellid, at::Tensor layer) {
  CHECK_INPUT(dMMask);
  CHECK_INPUT(dM_num);
  CHECK_INPUT(pid_unassigned);
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  CHECK_INPUT(xp);
  CHECK_INPUT(yp);
  CHECK_INPUT(cellid);
  CHECK_INPUT(layer);
  
  assignPar_cuda(dMMask, dM_num, pid_unassigned,
                    x, y, 
                    xp, yp,
                    cellid, layer);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &assignPar,
        "Update After washoff, CUDA version");
}