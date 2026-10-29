#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
at::Tensor virtual_compact_softmax(at::Tensor,at::Tensor,int64_t,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("virtual_compact_softmax",&virtual_compact_softmax);}
