#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
void washoff_cuda(at::Tensor Ms, at::Tensor Mg, 
                    at::Tensor Ms_num, at::Tensor Mg_num, at::Tensor dM_num,
                    at::Tensor Mrs, at::Tensor Mrg, 
                    at::Tensor h,at::Tensor qx, at::Tensor qy,
                    at::Tensor manning, at::Tensor P,
                    at::Tensor polAttributes, at::Tensor vs, 
                    at::Tensor ratio,
                    at::Tensor dx, at::Tensor dt);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void washoff(at::Tensor Ms, at::Tensor Mg, 
                    at::Tensor Ms_num, at::Tensor Mg_num, at::Tensor dM_num,
                    at::Tensor Mrs, at::Tensor Mrg, 
                    at::Tensor h,at::Tensor qx, at::Tensor qy,
                    at::Tensor manning, at::Tensor P,
                    at::Tensor polAttributes, at::Tensor vs, 
                    at::Tensor ratio,
                    at::Tensor dx, at::Tensor dt) {
  CHECK_INPUT(Ms);
  CHECK_INPUT(Mg);
  CHECK_INPUT(Ms_num);
  CHECK_INPUT(Mg_num);
  CHECK_INPUT(Mrs);
  CHECK_INPUT(Mrg);
  CHECK_INPUT(h);
  CHECK_INPUT(qx);
  CHECK_INPUT(qy);
  CHECK_INPUT(manning);
  CHECK_INPUT(P);
  CHECK_INPUT(polAttributes);
  CHECK_INPUT(vs);
  CHECK_INPUT(ratio);
  CHECK_INPUT(dx);
  CHECK_INPUT(dt);

  washoff_cuda(Ms, Mg, Ms_num, Mg_num, Mrs, Mrg, h, qx, qy,
                    manning, P, polAttributes, vs, ratio, dx, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("updating", &washoff,
        "Pollutant Washoff, CUDA version");
}