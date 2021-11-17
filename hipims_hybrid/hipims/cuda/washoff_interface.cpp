#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
void washoff_cuda(at::Tensor wetMaskIndex, 
              at::Tensor h, at::Tensor qx, at::Tensor qy, 
              at::Tensor Ms, at::Tensor Mg, 
              at::Tensor Ms_num, at::Tensor Mg_num,  at::Tensor dM_num,
              at::Tensor Mrs, at::Tensor Mrg,
              at::Tensor p_mass, at::Tensor pid, at::Tensor cellid, at::Tensor layer, 
              at::Tensor P, at::Tensor manning, 
              at::Tensor polAttributes, at::Tensor dt);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void washoff(at::Tensor wetMaskIndex, 
              at::Tensor h, at::Tensor qx, at::Tensor qy, 
              at::Tensor Ms, at::Tensor Mg, 
              at::Tensor Ms_num, at::Tensor Mg_num, at::Tensor dM_num,
              at::Tensor Mrs, at::Tensor Mrg,
              at::Tensor p_mass, at::Tensor pid, at::Tensor cellid, at::Tensor layer, 
              at::Tensor P, at::Tensor manning, 
              at::Tensor polAttributes, at::Tensor dt) {
  CHECK_INPUT(wetMaskIndex);
  CHECK_INPUT(h);
  CHECK_INPUT(qx);
  CHECK_INPUT(qy);
  CHECK_INPUT(Ms);
  CHECK_INPUT(Mg);
  CHECK_INPUT(Ms_num);
  CHECK_INPUT(Mg_num);
  CHECK_INPUT(dM_num);
  CHECK_INPUT(Mrs);
  CHECK_INPUT(Mrg);
  CHECK_INPUT(p_mass);
  CHECK_INPUT(pid);
  CHECK_INPUT(cellid);
  CHECK_INPUT(layer);
  CHECK_INPUT(P);
  CHECK_INPUT(manning);
  CHECK_INPUT(polAttributes);
  CHECK_INPUT(dt);

  washoff_cuda(wetMaskIndex, h, qx, qy, 
              Ms, Mg, 
              Ms_num, Mg_num, dM_num,
              Mrs, Mrg,
              p_mass, pid, cellid, layer,
              P, manning,
              polAttributes, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update", &washoff,
        "Update After washoff, CUDA version");
}