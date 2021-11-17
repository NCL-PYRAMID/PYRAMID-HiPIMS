#include <torch/extension.h>
#include <vector>

// // CUDA forward declaration
// void update_after_transport_cuda(at::Tensor x, at::Tensor cellid, at::Tensor Ms_num);

// // C++ interface
// #define CHECK_CUDA(x)                                                          
//   TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
// #define CHECK_CONTIGUOUS(x)                                                    
//   TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
// #define CHECK_INPUT(x)                                                         
//   CHECK_CUDA(x);                                                               
//   CHECK_CONTIGUOUS(x)

// void update_after_transport(at::Tensor x, at::Tensor cellid, at::Tensor Ms_num) {
//   CHECK_INPUT(x);
//   CHECK_INPUT(cellid);
//   CHECK_INPUT(Ms_num);

//   update_after_transport_cuda(x, cellid, Ms_num);
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("update", &update_after_transport,
//         "Update Particles After Transport, CUDA version");
// }

// CUDA forward declaration
void update_after_transport_cuda(at::Tensor Ms, at::Tensor Mrs, at::Tensor Ms_num, at::Tensor cellid, at::Tensor layer, at::Tensor p_mass);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void update_after_transport(at::Tensor Ms, at::Tensor Mrs, at::Tensor Ms_num, at::Tensor cellid, at::Tensor layer, at::Tensor p_mass) {
  CHECK_INPUT(Ms);
  CHECK_INPUT(Mrs);
  CHECK_INPUT(Ms_num);
  CHECK_INPUT(cellid);
  CHECK_INPUT(layer);
  CHECK_INPUT(p_mass);

  update_after_transport_cuda(Ms, Mrs, Ms_num, cellid, layer, p_mass);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &update_after_transport,
        "Update Particles After Transport, CUDA version");
}
