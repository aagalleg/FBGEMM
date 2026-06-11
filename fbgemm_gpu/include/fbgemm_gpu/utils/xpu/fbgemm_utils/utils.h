/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/xpu/XPUStream.h>
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <cstdint>

#include "dispatch_macros.h"

namespace fbgemm_xpu {

constexpr int VEC_WIDTH = 4;
constexpr size_t kThreadGroupSize = 32;
constexpr int32_t kCacheLocationMissing = -1;
constexpr size_t kMaxThreads = 1024;
constexpr size_t kForwardMaxThreads = 512;
constexpr size_t kBackwardMaxThreads = 512;

using overflow_safe_int_t = int64_t;

// These values are adjusted in backward based on B and T
constexpr int DEFAULT_INFO_NUM_BITS = 32;
constexpr int DEFAULT_INFO_B_NUM_BITS = 26;
constexpr uint32_t DEFAULT_INFO_B_MASK = (1u << DEFAULT_INFO_B_NUM_BITS) - 1;
constexpr uint32_t MAX_T =
    (1u << (DEFAULT_INFO_NUM_BITS - DEFAULT_INFO_B_NUM_BITS)) - 1;
constexpr uint32_t MAX_B = (1u << DEFAULT_INFO_B_NUM_BITS) - 1;

enum class SparseType : uint8_t {
  FP32 = 0,
  FP16 = 1,
  INT8 = 2,
  INT4 = 3,
  INT2 = 4,
  BF16 = 5,
  FP8 = 6,
  INVALID = 7,
  MX4 = 8,
  NFP8 = 9,
};

// SYCL atomic add (equivalent to CUDA atomicAdd)
template <typename T>
inline T xpuAtomicAdd(T* address, T val) {
  sycl::atomic_ref<
      T,
      sycl::memory_order::relaxed,
      sycl::memory_scope::device,
      sycl::access::address_space::global_space>
      atomic_val(*address);
  return atomic_val.fetch_add(val);
}

template <typename T>
inline T div_round_up(T numerator, T denominator) {
    return (numerator + denominator - 1) / denominator;
}

inline at::ScalarType getScalarType(SparseType dtype) {
  switch (dtype) {
    case SparseType::FP32:
      return at::kFloat;
    case SparseType::FP16:
      return at::kHalf;
    case SparseType::INT8:
      return at::kByte;
    case SparseType::BF16:
      return at::kBFloat16;
    case SparseType::INT4:
      return at::kQUInt4x2;
    case SparseType::INT2:
      return at::kQUInt2x4;
    case SparseType::NFP8:
      return at::kFloat8_e4m3fn;
    default:
      return at::ScalarType::Undefined;
  }
};

enum class PoolingMode : uint8_t { SUM = 0, MEAN = 1, NONE = 2 };

// Keep in sync with EmbeddingLocation in split_table_batched_embeddings_ops.py
enum class PlacementType : uint8_t {
  DEVICE = 0,
  MANAGED = 1,
  MANAGED_CACHING = 2,
  HOST = 3,
};

std::tuple<int32_t, uint32_t> adjust_info_B_num_bits(int32_t B, int32_t T);


// PhiloxXpuState is not available in PyTorch 2.8
// Use native PhiloxState in future implementations
struct PhiloxXpuState {
    uint64_t seed = 0;
    uint64_t offset = 0;
};

using fint32 = union fint32 {
  uint32_t I;
  float F;
};

class FixedDivisor {
 public:
  explicit FixedDivisor(const int32_t d) : d_(d) {
    CalcSignedMagic();
  }

  /// Calculates `q = n / d`.
  int32_t Div(const int32_t n) const {
    // In lieu of a mulhi instruction being available, perform the
    // work in uint64
    return (int32_t)((magic_ * (uint64_t)n) >> shift_);
  }

  /// Calculates `r = n % d`.
  int32_t Mod(const int32_t n) const {
    return n - d_ * Div(n);
  }

  /// Calculates `q = n / d` and `r = n % d` together.
  void DivMod(const int32_t n, int32_t* q, int32_t* r) const {
    *q = Div(n);
    *r = n - d_ * *q;
  }
  int32_t D() const {
    return d_;
  }

 private:
  // Calculates magic multiplicative value and shift amount for calculating `q =
  // n / d` for signed 32-bit integers.
  // Implementation taken from Hacker's Delight section 10.
  void CalcSignedMagic() {
    if (d_ == 1) {
      magic_ = UINT64_C(0x1) << 32;
      shift_ = 32;
      return;
    }

    const uint32_t two31 = UINT32_C(0x80000000);
    const uint32_t ad = std::abs(d_);
    const uint32_t t = two31 + ((uint32_t)d_ >> 31);
    const uint32_t anc = t - 1 - t % ad; // Absolute value of nc.
    uint32_t p = 31; // Init. p.
    uint32_t q1 = two31 / anc; // Init. q1 = 2**p/|nc|.
    uint32_t r1 = two31 - q1 * anc; // Init. r1 = rem(2**p, |nc|).
    uint32_t q2 = two31 / ad; // Init. q2 = 2**p/|d|.
    uint32_t r2 = two31 - q2 * ad; // Init. r2 = rem(2**p, |d|).
    uint32_t delta = 0;
    do {
      ++p;
      q1 <<= 1; // Update q1 = 2**p/|nc|.
      r1 <<= 1; // Update r1 = rem(2**p, |nc|).
      if (r1 >= anc) { // (Must be an unsigned comparison here).
        ++q1;
        r1 -= anc;
      }
      q2 <<= 1; // Update q2 = 2**p/|d|.
      r2 <<= 1; // Update r2 = rem(2**p, |d|).
      if (r2 >= ad) { // (Must be an unsigned comparison here).
        ++q2;
        r2 -= ad;
      }
      delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));
    int32_t magic = q2 + 1;
    if (d_ < 0) {
      magic = -magic;
    }
    shift_ = p;
    magic_ = (uint64_t)(uint32_t)magic;
  }
  int32_t d_ = 1;
  uint64_t magic_;
  int shift_;
};


// Based on the empirical study, max grid size that is 64x larger than the
// number of compute units gives good performance across the board
constexpr int32_t MAX_WORK_GROUPS_FACTOR = 64;

inline int32_t get_max_work_groups_() {
  auto device = c10::xpu::getCurrentXPUStream().queue().get_device();
  return MAX_WORK_GROUPS_FACTOR *
      device.get_info<sycl::info::device::max_compute_units>();
}

#define SYCL_DEVICE_GUARD(TENSOR)          \
  c10::OptionalDeviceGuard device_guard;  \
  device_guard.reset_device(TENSOR.device())

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
#define DISPATCH_OPTIMAL_KERNEL(MAX_D, ...)                                   \
  [&] {                                                                       \
    if (MAX_D <= 32) {                               \
             [[ maybe_unused ]] const int max_vecs_per_thread =               \
               1;                                      \
             constexpr int kFixedMaxVecsPerThread = 1; \
             [[ maybe_unused ]] constexpr int kThreadGroupSize =              \
               8;                                            \
             [[ maybe_unused ]] constexpr bool kUseVecBlocking =              \
               false;                                             \
             return __VA_ARGS__();                                            \
           }                                                                  \
        if (MAX_D <= 64) {                               \
             [[ maybe_unused ]] const int max_vecs_per_thread =               \
               1;                                      \
             constexpr int kFixedMaxVecsPerThread = 1; \
             [[ maybe_unused ]] constexpr int kThreadGroupSize =              \
               16;                                            \
             [[ maybe_unused ]] constexpr bool kUseVecBlocking =              \
               false;                                             \
             return __VA_ARGS__();                                            \
           }                                                                  \
        if (MAX_D <= 128) {                               \
             [[ maybe_unused ]] const int max_vecs_per_thread =               \
               1;                                      \
             constexpr int kFixedMaxVecsPerThread = 1; \
             [[ maybe_unused ]] constexpr int kThreadGroupSize =              \
               32;                                            \
             [[ maybe_unused ]] constexpr bool kUseVecBlocking =              \
               false;                                             \
             return __VA_ARGS__();                                            \
           }                                                                  \
        if (MAX_D <= 256) {                               \
             [[ maybe_unused ]] const int max_vecs_per_thread =               \
               2;                                      \
             constexpr int kFixedMaxVecsPerThread = 2; \
             [[ maybe_unused ]] constexpr int kThreadGroupSize =              \
               32;                                            \
             [[ maybe_unused ]] constexpr bool kUseVecBlocking =              \
               false;                                             \
             return __VA_ARGS__();                                            \
           }                                                                  \
        if (MAX_D > 256) {                                     \
         [[ maybe_unused ]] const int max_vecs_per_thread =                  \
           (MAX_D + 128 - 1) / 128;                \
         constexpr int kFixedMaxVecsPerThread = 2; \
         [[ maybe_unused ]] constexpr int kThreadGroupSize = fbgemm_xpu::kThreadGroupSize;      \
         [[ maybe_unused ]] constexpr bool kUseVecBlocking = true;           \
         return __VA_ARGS__();                                               \
       }                                                                     \
    }()

#else
#define DISPATCH_OPTIMAL_KERNEL(MAX_D, ...)                                   \
  [&] {                                                                       \
    if (MAX_D <= 128) {                               \
             [[ maybe_unused ]] const int max_vecs_per_thread =               \
               1;                                      \
             constexpr int kFixedMaxVecsPerThread = 1; \
             [[ maybe_unused ]] constexpr int kThreadGroupSize =              \
               32;                                            \
             [[ maybe_unused ]] constexpr bool kUseVecBlocking =              \
               false;                                             \
             return __VA_ARGS__();                                            \
           }                                                                  \
        if (MAX_D <= 256) {                               \
             [[ maybe_unused ]] const int max_vecs_per_thread =               \
               2;                                      \
             constexpr int kFixedMaxVecsPerThread = 2; \
             [[ maybe_unused ]] constexpr int kThreadGroupSize =              \
               32;                                            \
             [[ maybe_unused ]] constexpr bool kUseVecBlocking =              \
               false;                                             \
             return __VA_ARGS__();                                            \
           }                                                                  \
        if (MAX_D > 256) {                                     \
         [[ maybe_unused ]] const int max_vecs_per_thread =                  \
           (MAX_D + 128 - 1) / 128;                \
         constexpr int kFixedMaxVecsPerThread = 2; \
         [[ maybe_unused ]] constexpr int kThreadGroupSize = fbgemm_xpu::kThreadGroupSize;      \
         [[ maybe_unused ]] constexpr bool kUseVecBlocking = true;           \
         return __VA_ARGS__();                                               \
       }                                                                     \
    }()
#endif

inline void validate_local_mem_size(
    sycl::queue& q,
    const int32_t local_mem_bytes) {
  
  auto device = q.get_device();
  auto max_local_mem = device.get_info<sycl::info::device::local_mem_size>();
  TORCH_CHECK(
      local_mem_bytes <= max_local_mem,
      "Attempted to allocate ",
      local_mem_bytes / 1024,
      " KB of local memory but only ",
      max_local_mem / 1024,
      " KB is available");
}

template<typename func_t>
int32_t compute_num_groups_and_dynamic_smem_bytes(
    int32_t* num_groups,
    const func_t compute_smem_bytes_fn,
    const int32_t used_shared_bytes) {
  int32_t smem_bytes = 0;
  while (
      (smem_bytes = compute_smem_bytes_fn(*num_groups))
      >= used_shared_bytes
  ) {
    *num_groups /= 2;
  }
  TORCH_CHECK_GE(*num_groups, 1);

  return smem_bytes;
}

#define DISPATCH_OPTIMAL_NOBAG_FORWARD_KERNEL(DD_, ...)                     \
    [&] {                                                                       \
        if (DD_ <= 4) {                                                         \
            constexpr int kEmbeddingSize = 4;                                   \
            return __VA_ARGS__();                                               \
        }                                                                       \
        if (DD_ <= 8) {                                                         \
            constexpr int kEmbeddingSize = 8;                                   \
            return __VA_ARGS__();                                               \
        }                                                                       \
        if (DD_ <= 16) {                                                        \
            constexpr int kEmbeddingSize = 16;                                  \
            return __VA_ARGS__();                                               \
        }                                                                       \
        if (DD_ <= 32) {                                                        \
            constexpr int kEmbeddingSize = 32;                                  \
            return __VA_ARGS__();                                               \
        }                                                                       \
        return;                                                                 \
    }()

} // namespace fbgemm_xpu
