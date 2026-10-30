#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
at::Tensor virtual_compact_softmax(at::Tensor,at::Tensor,int64_t,int64_t,bool,at::Tensor);
at::Tensor virtual_compact_softmax_live(at::Tensor,at::Tensor,int64_t,int64_t,bool,at::Tensor,at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("virtual_compact_softmax",&virtual_compact_softmax);
 m.def("virtual_compact_softmax_live",&virtual_compact_softmax_live);
}
