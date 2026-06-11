/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

{#
// @lint-ignore LINTIGNORE
// @lint-ignore-every CLANGFORMAT
// clang-format off
// Note: clang-format off doesn't work with this templaterized code,
// so we need to keep lint-ignore-every.
// See https://fburl.com/dw9ljh4h
#}

{%- set mdesc = "dense" if dense else ("ssd" if ssd else "split") %}
{%- set locs_or_addrs_tensor = "ssd_row_addrs" if ssd else "lxu_cache_locations" %}
{%- set locs_or_addrs_type = "int64_t" if ssd else "int32_t" %}
{%- set locs_or_addrs_idx = "row_idx" if ssd else "cache_idx" %}
{%- set kernel_class = "DenseEmbeddingNobagCodegenForwardUnweightedSmallKernel" if dense else "SplitEmbeddingNoBagCodegenForwardUnweightedSmallKernel" %}
{#- Accessor namespace: 'at' for SYCL (standard PyTorch), 'pta' for CUDA (packed variant) #}
{%- set pta = "at" if is_sycl else "pta" %}
{#- Unified name for the shuffled cache-location value in the inner loop #}
{%- set cache_ref_j = "cache_idx_j" if is_sycl else (locs_or_addrs_idx + "_j") %}

{%- if is_sycl %}
#include <sycl/sycl.hpp>
#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/StridedRandomAccessor.h>

#include "fbgemm_gpu/utils/xpu/fbgemm_utils/utils.h"
#include "fbgemm_gpu/utils/xpu/fbgemm_utils/vec4.h"
#include "fbgemm_gpu/utils/xpu/fbgemm_utils/weight_row.h"

using at::native::RestrictPtrTraits;
using namespace fbgemm_xpu;
namespace fbgemm {
{%- else %}
#include "fbgemm_gpu/embedding_forward_template_helpers.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;
{%- endif %}

template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    typename index_t,
    size_t kThreadGroupSize
    >
{%- if is_sycl %}
// SYCL: kernel expressed as a functor class submitted via queue.submit
class {{ kernel_class }} {
public:
    {{ kernel_class }}(
{%- else %}
__launch_bounds__(kForwardMaxThreads) __global__ void
{%- if is_index_select %}
batch_index_select_dim0_codegen_forward_small_kernel(
{%- else %}
{{ mdesc }}_embedding_nobag_codegen_forward_unweighted_small_kernel(
{%- endif %}
{%- endif %}
    // ---- shared parameter list (accessor prefix controlled by 'pta' variable) ----
    const {{ pta }}::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const {{ pta }}::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const {{ pta }}::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const {{ pta }}::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const {{ pta }}::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if is_index_select and not is_sycl %}
    const {{ pta }}::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    FixedDivisor fd_B,
    const {{ pta }}::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    {%- if not is_index_select or is_sycl %}
    const {{ pta }}::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {%- endif %}
    {%- if not dense %}
    const {{ pta }}::PackedTensorAccessor32<{{ locs_or_addrs_type }}, 1, at::RestrictPtrTraits> {{ locs_or_addrs_tensor }},
    {%- endif %}
    {%- if is_index_select and not is_sycl %}
    const {{ pta }}::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    const {{ pta }}::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> total_L_offsets,
    const int32_t fixed_L_per_warp,
    const bool permute_output_dim_0_1,
    {%- endif %}
    {{ pta }}::PackedTensorAccessor64<output_t, {{ "1" if (is_index_select and not is_sycl) else "2" }}, at::RestrictPtrTraits> output
{%- if is_sycl %}
    // SYCL: constructor stores all args as class members, body lives in operator()
    ) : dev_weights(dev_weights),
{%- if not dense %}
        uvm_weights(uvm_weights),
        lxu_cache_weights(lxu_cache_weights),
        weights_placements(weights_placements),
{%- endif %}
        weights_offsets(weights_offsets),
        D(D),
        fd_B(fd_B),
        indices(indices),
        offsets(offsets),
{%- if not dense %}
        {{ locs_or_addrs_tensor }}({{ locs_or_addrs_tensor }}),
{%- endif %}
        output(output) {}

    void operator()(const sycl::nd_item<2>& item) const {
    // SYCL thread indexing
    const auto b_t = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (static_cast<int64_t>(b_t) >= offsets.size(0) - 1) {
        return;
    }
{%- else %}
    // CUDA thread indexing
    ) {
    int32_t T = weights_offsets.size(0);
    auto b_t = blockIdx.x * blockDim.y + threadIdx.y;
    {%- if not is_index_select %}
    if (b_t >= offsets.size(0) - 1) {
        return;
    }
    {%- endif %}
{%- endif %}
    int32_t t;
    int32_t b;

    fd_B.DivMod(b_t, &t, &b);

    {%- if is_index_select %}
    index_t indices_start;
    int32_t L;
    int32_t L_start;
    if (t >= T) {
        return;
    }
    const auto total_L_start = total_L_offsets[t];
    const auto total_L = total_L_offsets[t + 1] - total_L_start;
    L_start = b * fixed_L_per_warp;
    if (L_start >= total_L) {
        return;
    }
    indices_start = total_L_start + L_start;
    L = (total_L - L_start >= fixed_L_per_warp) ? fixed_L_per_warp : (total_L - L_start);
    {%- else %}
    const auto indices_start = offsets[b_t];
    const auto L = offsets[b_t + 1] - indices_start;
    {%- endif %}

    {%- if is_index_select %}
    const auto D_start = D_offsets[t];
    const auto D_end = D_offsets[t + 1];
    const auto D = D_end - D_start;

    // Check D in the kernel to avoid iterating through the list on host
    CUDA_KERNEL_ASSERT(D % 4 == 0 && "The column size must be multiple of 4");
    const auto output_offset = permute_output_dim_0_1 ? D_start : output_offsets[t];
    const auto output_stride = permute_output_dim_0_1 ? D_offsets[T] : D;
    {%- endif %} // dense

    int64_t weights_offset = weights_offsets[t];
    const emb_t* __restrict__ weights;
    {%- if not dense %}
    const auto placement = static_cast<PlacementType>(weights_placements[t]);
    if (placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset];
    } else {
        weights = {{ "nullptr" if ssd else "&uvm_weights[weights_offset]" }};
    }
    {%- else %}
    weights = &dev_weights[weights_offset];
    {%- endif %}

    int32_t D_emb = D;
{%- if not is_sycl %}
    // INT8 quantization params are appended to the row in CUDA only
    if constexpr (std::is_same_v<emb_t, uint8_t>) {
        D_emb += kINT8QparamsBytes;
    }
{%- endif %}

{%- if is_sycl %}
    // SYCL: lane within the sub-group; sub-group replaces CUDA warp
    const int32_t lane_id = static_cast<int32_t>(item.get_local_id(1));
    const int32_t group_start =
        (lane_id / static_cast<int32_t>(kThreadGroupSize)) *
        static_cast<int32_t>(kThreadGroupSize);
    const int32_t group_end = group_start + static_cast<int32_t>(kThreadGroupSize);
    const int32_t d = (lane_id % static_cast<int32_t>(kThreadGroupSize)) * 4;
    // Sub-group size is queried at runtime; replaces compile-time kWarpSize
    const uint32_t kSubGroupSize = item.get_sub_group().get_local_range().get(0);
{%- else %}
    const auto group_start = threadIdx.x / kThreadGroupSize * kThreadGroupSize;
    const int32_t group_end = group_start + kThreadGroupSize;
    const auto d = threadIdx.x % kThreadGroupSize * 4;
{%- endif %}

    for (int32_t l_start = 0; l_start < L; l_start += {{ "kSubGroupSize" if is_sycl else "kWarpSize" }}) {
{%- if is_sycl %}
        const int32_t l = l_start + lane_id;
{%- else %}
        auto l = l_start + threadIdx.x;
{%- endif %}
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        {%- if not dense %}
        const {{ locs_or_addrs_type }} {{ locs_or_addrs_idx }} =
          (placement == PlacementType::MANAGED_CACHING && l < L)
            ? {{ locs_or_addrs_tensor }}[indices_start + l] : 0;
        {%- endif %}
        for (auto j = group_start; j < group_end && l_start + j < L; ++j) {
{%- if is_sycl %}
            // SYCL: broadcast idx from lane j across the sub-group
            int64_t idx_j = sycl::select_from_group(item.get_sub_group(), idx, j);
{%- else %}
            int64_t idx_j = shfl_sync(idx, j);
{%- endif %}
            {%- if is_index_select %}
            int64_t output_j = L_start + l_start + j;
            {%- else %}
            int64_t output_j = indices_start + l_start + j;
            {%- endif %}
            {%- if not dense %}
{%- if is_sycl %}
            // SYCL: broadcast cache location from lane j (always int32_t; SSD not supported on XPU)
            const int32_t {{ cache_ref_j }} =
                sycl::select_from_group(item.get_sub_group(), {{ locs_or_addrs_idx }}, j);
{%- else %}
            const {{ locs_or_addrs_type }} {{ cache_ref_j }} =
              shfl_sync({{ locs_or_addrs_idx }}, j);
{%- endif %}
            {%- endif %}

            auto weight_row_emb = WeightRowAccessor<emb_t, cache_t>(
                &weights[idx_j * D_emb],
                D
            );

            if (d < D) {
                {%- if not dense %}
                if (placement == PlacementType::MANAGED_CACHING &&
                    {{ cache_ref_j }} != kCacheLocationMissing) {
                    const cache_t* cache_weights;
                    {%- if ssd %}
                    cache_weights = reinterpret_cast<const cache_t*>(
                        *reinterpret_cast<const uint64_t*>(&{{ cache_ref_j }}));
                    {%- else %}
                    cache_weights = reinterpret_cast<const cache_t*>(
                        &lxu_cache_weights[{{ cache_ref_j }}][0]);
                    {%- endif  %}

                    auto weight_row_cache = WeightRowAccessor<cache_t, cache_t>(cache_weights, D);
                    Vec4T<cache_t> weight = weight_row_cache.load(d);
                    weight.store(&output[output_j][d]);
                } else {
                    Vec4T<cache_t> weight = weight_row_emb.load(d);
                    weight.store(&output[output_j][d]);
                }
                {%- else %}
                    Vec4T<cache_t> weight = weight_row_emb.load(d);
                    {%- if is_index_select and not is_sycl %}
                    // output is 1D (because the stride can be irregular)
                    weight.store(&output[output_offset + output_j * output_stride + d]);
                    {%- else %}
                    // output is 2D
                    weight.store(&output[output_j][d]);
                    {%- endif %}
                {%- endif %}
            }
        }
    }
{%- if is_sycl %}
    }  // operator()

private:
    // ---- class members mirror the kernel parameter list ----
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights;
{%- if not dense %}
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights;
    const at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights;
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements;
{%- endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets;
    int64_t D;
    FixedDivisor fd_B;
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices;
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets;
{%- if not dense %}
    const at::PackedTensorAccessor32<{{ locs_or_addrs_type }}, 1, at::RestrictPtrTraits> {{ locs_or_addrs_tensor }};
{%- endif %}
    mutable at::PackedTensorAccessor64<output_t, 2, at::RestrictPtrTraits> output;
};

} // namespace fbgemm
{%- else %}
}
{%- endif %}

/*
    Explicitly instantiate the kernel function template.  The instantiations are
    based on the types enumerated by DISPATCH_EMB_GRAD_CACHE_TYPES macro used in
    embedding_forward_split_template.cu
*/

{%- for output_type in ['float', 'at::Half', 'at::BFloat16'] %}
{#- Float8 is not supported on XPU hardware; omit from SYCL instantiations #}
{%- for emb_type in (['float', 'at::Half'] + ([] if is_sycl else (['at::Float8_e4m3fnuz'] if is_rocm else ['at::Float8_e4m3fn']))) %}
{%- for cache_type in ['float', 'at::Half'] %}
{%- for kEmbeddingSize in [4, 8, 16, 32] %}
{%- for index_type in ['int32_t', 'int64_t'] %}

{%- if is_sycl %}
// SYCL: explicit class instantiation
template class {{ kernel_class }}<
  {{ emb_type }},
  {{ cache_type }},
  {{ output_type }},
  {{ index_type }},
  {{ kEmbeddingSize // 4 }}
>;
{%- else %}
template __launch_bounds__(kForwardMaxThreads) __global__ void
{%- if is_index_select %}
batch_index_select_dim0_codegen_forward_small_kernel
{%- else %}
{{ mdesc }}_embedding_nobag_codegen_forward_unweighted_small_kernel
{%- endif %}
<
  {{ emb_type }},
  {{ cache_type }},
  {{ output_type }},
  {{ index_type }},
  {{ kEmbeddingSize // 4 }}
> (
    const pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> uvm_weights,
    const pta::PackedTensorAccessor64<{{ cache_type }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    FixedDivisor fd_B,
    const pta::PackedTensorAccessor32<{{ index_type }}, 1, at::RestrictPtrTraits> indices,
    {%- if not is_index_select %}
    const pta::PackedTensorAccessor32<{{ index_type }}, 1, at::RestrictPtrTraits> offsets,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<{{ locs_or_addrs_type }}, 1, at::RestrictPtrTraits> {{ locs_or_addrs_tensor }},
    {%- endif %}
    {%- if is_index_select %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> total_L_offsets,
    const int32_t fixed_L_per_warp,
    const bool permute_output_dim_0_1,
    {%- endif %}
    pta::PackedTensorAccessor64<{{ output_type }}, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> output);
{%- endif %}

{%- endfor %}
{%- endfor %}
{%- endfor %}
{%- endfor %}
{%- endfor %}
