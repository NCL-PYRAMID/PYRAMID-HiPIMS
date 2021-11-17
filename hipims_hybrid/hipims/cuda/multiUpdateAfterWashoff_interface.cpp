#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
void update_after_washoff_cuda(at::Tensor pid_assigned, at::Tensor layer, 
                                at::Tensor pclass, at::Tensor cellid, 
                                at::Tensor Ms_num, at::Tensor Mg_num, 
                                at::Tensor dMs_num, at::Tensor dMg_num, at::Tensor x);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                        \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)


void update_after_washoff(at::Tensor pid_assigned, at::Tensor layer, 
                            at::Tensor pclass, at::Tensor cellid,
                            at::Tensor Ms_num, at::Tensor Mg_num, 
                            at::Tensor dMs_num, at::Tensor dMg_num, at::Tensor x) {
  CHECK_INPUT(pid_assigned);
  CHECK_INPUT(layer);
  CHECK_INPUT(pclass);
  CHECK_INPUT(cellid);
  CHECK_INPUT(Ms_num);
  CHECK_INPUT(Mg_num);
  CHECK_INPUT(dMs_num);
  CHECK_INPUT(dMg_num);
  CHECK_INPUT(x);

  update_after_washoff_cuda(pid_assigned, layer, pclass, cellid, Ms_num, Mg_num, dMs_num, dMg_num, x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &update_after_washoff,
        "Update Particles After Washoff, CUDA version");
}