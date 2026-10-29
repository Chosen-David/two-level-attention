#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
at::Tensor sparse_softmax(at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("sparse_softmax",&sparse_softmax);}
