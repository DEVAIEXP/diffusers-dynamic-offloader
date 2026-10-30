// Benchmark-only Windows registration adapter. Not part of the DDO package.
#include <torch/library.h>
#include "torch_binding.h"

TORCH_LIBRARY(external_qk_rope_bench, ops) {
  ops.def("rms_norm_split_rope(Tensor x, Tensor weight, Tensor cos, Tensor sin, int heads, float eps) -> Tensor");
  ops.def("rms_norm_split_rope_qk(Tensor q, Tensor k, Tensor weight_q, Tensor weight_k, Tensor cos, Tensor sin, int heads, float eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(external_qk_rope_bench, CUDA, ops) {
  ops.impl("rms_norm_split_rope", &rms_norm_split_rope);
  ops.impl("rms_norm_split_rope_qk", &rms_norm_split_rope_qk);
}
