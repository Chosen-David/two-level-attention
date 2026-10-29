// Reuse the exact pinned PyTorch TopK final small-sort, including tie order.
#include <ATen/ATen.h>
#include <ATen/native/cuda/Sort.h>
#include <torch/csrc/utils/pybind.h>
void sort_rank(at::Tensor values, at::Tensor ids) {
  TORCH_CHECK(values.is_cuda() && ids.is_cuda() && values.device()==ids.device(), "same CUDA device required");
  TORCH_CHECK(values.dim()==2 && values.sizes()==ids.sizes() && values.is_contiguous() && ids.is_contiguous(), "contiguous equal matrices required");
  TORCH_CHECK(ids.scalar_type()==at::kLong && values.size(1)<=4096, "int64 indices, K<=4096 required");
  at::native::sortKeyValueInplace(values,ids,1,true);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("sort_rank",&sort_rank);}
