// =============================================================================
// Copyright 2024 Enflame. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
// RUN: bash %cc_wrapper.sh %cc -std=c++17 %flags %target %s -o %t -lgtest \
// RUN: -lpthread
// RUN: %t

#include <stdint.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include "krt_benchmark/krt_benchmark.h"

#include "dte/fixture.h"
#include "krt/dte.h"
#include "krt/misc.h"
#include "krt_benchmark/clock.h"
#include "krt_benchmark/utils.h"

constexpr int LOOP_COUNT = 8;

#if __GCU_ARCH__ >= 400
#define DEFINE_DTE_CTX(use_cdte) tops_dte_ctx_t ctx;
#else
#define DEFINE_DTE_CTX(use_cdte)                                               \
  tops_dte_ctx_t sdte_ctx;                                                     \
  __private_dte__ tops_dte_ctx_t cdte_ctx;                                     \
  auto &ctx = use_cdte ? cdte_ctx : sdte_ctx;
#endif

__global__ void
KernelDteConfigLinearCopy(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(total_size);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_linear_copy(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                                   dst_addr, direction, total_size);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigSlice(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(src_offsets);
  DTE_FIXTURE_PARAM(dst_dims);
  DTE_FIXTURE_PARAM(value);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_slice(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                             dst_addr, direction, bpe, rank, src_dims,
                             src_offsets, dst_dims, value);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigSliceTranspose(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(src_offsets);
  DTE_FIXTURE_PARAM(transpose_layout);
  DTE_FIXTURE_PARAM(dst_dims);
  DTE_FIXTURE_PARAM(value);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_slice_transpose(
          __DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr, dst_addr, direction, bpe,
          rank, src_dims, src_offsets, transpose_layout, dst_dims, value);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}
__global__ void
KernelDteConfigDeslice(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(dst_offsets);
  DTE_FIXTURE_PARAM(dst_dims);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_deslice(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                               dst_addr, direction, bpe, rank, src_dims,
                               dst_offsets, dst_dims);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void KernelDteConfigTransposeDeslice(
    krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(transpose_layout);
  DTE_FIXTURE_PARAM(dst_offsets);
  DTE_FIXTURE_PARAM(dst_dims);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_transpose_deslice(
          __DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr, dst_addr, direction, bpe,
          rank, src_dims, transpose_layout, dst_offsets, dst_dims);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigBroadcast(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(dst_dims);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_broadcast(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                                 dst_addr, direction, bpe, rank, src_dims,
                                 dst_dims);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigTranspose(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(transpose_layout);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_transpose(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                                 dst_addr, direction, bpe, rank, src_dims,
                                 transpose_layout);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigPad(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(pad_low);
  DTE_FIXTURE_PARAM(pad_high);
  DTE_FIXTURE_PARAM(pad_mid);
  DTE_FIXTURE_PARAM(value);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_pad(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                           dst_addr, direction, bpe, rank, src_dims, pad_low,
                           pad_high, pad_mid, value);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigMemset(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(total_size);
  DTE_FIXTURE_PARAM(value);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_memset(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, dst_addr,
                              direction, bpe, total_size, value);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigMirrorLr(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_mirror_lr(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                                 dst_addr, direction, bpe, rank, src_dims);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigMirrorTb(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_mirror_tb(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                                 dst_addr, direction, bpe, rank, src_dims);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigShrink(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(total_size);
  DTE_FIXTURE_PARAM(phase);
  DTE_FIXTURE_PARAM(ratio);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_shrink(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                              dst_addr, direction, bpe, total_size, phase,
                              ratio);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigExpand(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(total_size);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(phase);
  DTE_FIXTURE_PARAM(ratio);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_expand(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                              dst_addr, direction, bpe, total_size, rank,
                              src_dims, phase, ratio);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteTrigger(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(total_size);
  DTE_FIXTURE_PARAM(value);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    tops::dte_config_memset(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, dst_addr,
                            direction, bpe, total_size, value);

    auto t0 = krt_benchmark::InvokeWithClock(
        [&]() { tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl); });

    result += t0;

    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

#if __GCU_ARCH__ >= 300

__global__ void
KernelDteConfigSlicePad(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(src_offsets);
  DTE_FIXTURE_PARAM(slice_dims);
  DTE_FIXTURE_PARAM(pad_low);
  DTE_FIXTURE_PARAM(pad_high);
  DTE_FIXTURE_PARAM(pad_mid);
  DTE_FIXTURE_PARAM(value);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_slice_pad(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                                 dst_addr, direction, bpe, rank, src_dims,
                                 src_offsets, slice_dims, pad_low, pad_high,
                                 pad_mid, value);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigSliceDeslice(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(src_offsets);
  DTE_FIXTURE_PARAM(slice_dims);
  DTE_FIXTURE_PARAM(dst_dims);
  DTE_FIXTURE_PARAM(dst_offsets);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_slice_deslice(
          __DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr, dst_addr, direction, bpe,
          rank, src_dims, src_offsets, slice_dims, dst_dims, dst_offsets);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigSliceBroadcast(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(src_offsets);
  DTE_FIXTURE_PARAM(slice_dims);
  DTE_FIXTURE_PARAM(dst_dims);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_slice_broadcast(
          __DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr, dst_addr, direction, bpe,
          rank, src_dims, src_offsets, slice_dims, dst_dims);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigSliceExpand(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(src_offsets);
  DTE_FIXTURE_PARAM(slice_dims);
  DTE_FIXTURE_PARAM(phase);
  DTE_FIXTURE_PARAM(ratio);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_slice_expand(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                                    dst_addr, direction, bpe, rank, src_dims,
                                    src_offsets, slice_dims, phase, ratio);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigShrinkDeslice(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(phase);
  DTE_FIXTURE_PARAM(ratio);
  DTE_FIXTURE_PARAM(dst_dims);
  DTE_FIXTURE_PARAM(dst_offsets);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_shrink_deslice(
          __DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr, dst_addr, direction, bpe,
          rank, src_dims, phase, ratio, dst_dims, dst_offsets);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigMemsetDeslice(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(value);
  DTE_FIXTURE_PARAM(dst_dims);
  DTE_FIXTURE_PARAM(dst_offsets);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_memset_deslice(__DTU_KRT_DTE_CTX_T_AS & ctx.impl,
                                      dst_addr, direction, bpe, rank, src_dims,
                                      value, dst_dims, dst_offsets);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigMirrorLrPad(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(pad_low);
  DTE_FIXTURE_PARAM(pad_high);
  DTE_FIXTURE_PARAM(pad_mid);
  DTE_FIXTURE_PARAM(value);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_mirror_lr_pad(
          __DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr, dst_addr, direction, bpe,
          rank, src_dims, pad_low, pad_high, pad_mid, value);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigMirrorTbPad(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(pad_low);
  DTE_FIXTURE_PARAM(pad_high);
  DTE_FIXTURE_PARAM(pad_mid);
  DTE_FIXTURE_PARAM(value);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_mirror_tb_pad(
          __DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr, dst_addr, direction, bpe,
          rank, src_dims, pad_low, pad_high, pad_mid, value);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigMirrorLrDeslice(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(dst_dims);
  DTE_FIXTURE_PARAM(dst_offsets);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_mirror_lr_deslice(__DTU_KRT_DTE_CTX_T_AS & ctx.impl,
                                         src_addr, dst_addr, direction, bpe,
                                         rank, src_dims, dst_dims, dst_offsets);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigMirrorTbDeslice(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(dst_dims);
  DTE_FIXTURE_PARAM(dst_offsets);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_mirror_tb_deslice(__DTU_KRT_DTE_CTX_T_AS & ctx.impl,
                                         src_addr, dst_addr, direction, bpe,
                                         rank, src_dims, dst_dims, dst_offsets);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

__global__ void
KernelDteConfigSubSample(krt_benchmark::CountersDeviceData counter_data) {
  krt_benchmark::CounterProxy counters(counter_data);

  DTE_FIXTURE_PARAM(use_cdte);
  DTE_FIXTURE_PARAM(src_addr);
  DTE_FIXTURE_PARAM(dst_addr);
  DTE_FIXTURE_PARAM(direction);
  DTE_FIXTURE_PARAM(bpe);
  DTE_FIXTURE_PARAM(rank);
  DTE_FIXTURE_PARAM(src_dims);
  DTE_FIXTURE_PARAM(dim_stride);

  DEFINE_DTE_CTX(use_cdte)
  tops::dte_scope s(ctx);

  float result = 0;
  for (int loop = 0; loop < LOOP_COUNT; loop++) {
    auto t0 = krt_benchmark::InvokeWithClock([&]() {
      tops::dte_config_sub_sample(__DTU_KRT_DTE_CTX_T_AS & ctx.impl, src_addr,
                                  dst_addr, direction, bpe, rank, src_dims,
                                  dim_stride);
    });

    result += t0;

    tops::dte_trigger(__DTU_KRT_DTE_CTX_T_AS & ctx.impl);
    tops::task_wait_thread_node(__DTU_KRT_TASK_NODE_T_AS & ctx.ev.impl);
  }
  counters[0] = result / LOOP_COUNT;
}

#endif // __GCU_ARCH__ >= 300

std::vector<dte_fixture::DteTestParam> dte_config_test_set = {
    {
        .name = "dte_config_linear_copy_sdte",
        .kernel = KernelDteConfigLinearCopy,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 60},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .direction = 0,
    },
#if 0
    {
        .name = "dte_config_linear_copy_cdte",
        .kernel = KernelDteConfigLinearCopy,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 63},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .direction = 0,
    },
    {
        .name = "dte_config_slice_sdte",
        .kernel = KernelDteConfigSlice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {4, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 231},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {4, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .src_offsets = {0, 0, 0, 0},
        .value = 0,
        .direction = 0,
    },
    {
        .name = "dte_config_slice_cdte",
        .kernel = KernelDteConfigSlice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {4, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 231},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {4, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .src_offsets = {0, 0, 0, 0},
        .value = 0,
        .direction = 0,
    },
    {
        .name = "dte_config_slice_transpose_sdte",
        .kernel = KernelDteConfigSliceTranspose,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {4, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 8, 64, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 390},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {4, 64, 8, 32},
        .dst_dims = {2, 8, 64, 32},
        .src_offsets = {2, 0, 0, 0},
        .transpose_layout = {0, 2, 1, 3},
        .value = 0,
        .direction = 0,
    },
    {
        .name = "dte_config_slice_transpose_cdte",
        .kernel = KernelDteConfigSliceTranspose,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {4, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 8, 64, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 390},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {4, 64, 8, 32},
        .dst_dims = {2, 8, 64, 32},
        .src_offsets = {2, 0, 0, 0},
        .transpose_layout = {0, 2, 1, 3},
        .value = 0,
        .direction = 0,
    },
    {
        .name = "dte_config_deslice_sdte",
        .kernel = KernelDteConfigDeslice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::ConcatenateTestData(
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32},
                                                            123),
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 0)),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 137},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {4, 64, 8, 32},
        .dst_offsets = {0, 0, 0, 0},
        .direction = 0,
    },
    {
        .name = "dte_config_deslice_cdte",
        .kernel = KernelDteConfigDeslice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::ConcatenateTestData(
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32},
                                                            123),
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 0)),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 137},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {4, 64, 8, 32},
        .dst_offsets = {0, 0, 0, 0},
        .direction = 0,
    },
    {
        .name = "dte_config_transpose_deslice_sdte",
        .kernel = KernelDteConfigTransposeDeslice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::ConcatenateTestData(
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 8, 64, 32},
                                                            123),
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 8, 64, 32}, 0)),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 294},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {4, 8, 64, 32},
        .dst_offsets = {0, 0, 0, 0},
        .transpose_layout = {0, 2, 1, 3},
        .direction = 0,
    },
    {
        .name = "dte_config_transpose_deslice_cdte",
        .kernel = KernelDteConfigTransposeDeslice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::ConcatenateTestData(
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 8, 64, 32},
                                                            123),
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 8, 64, 32}, 0)),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 291},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {4, 8, 64, 32},
        .dst_offsets = {0, 0, 0, 0},
        .transpose_layout = {0, 2, 1, 3},
        .direction = 0,
    },
    {
        .name = "dte_config_broadcast_cdte",
        .kernel = KernelDteConfigBroadcast,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {1, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 166},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {1, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .direction = 0,
    },
    {
        .name = "dte_config_broadcast_sdte",
        .kernel = KernelDteConfigBroadcast,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {1, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 166},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {1, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .direction = 0,
    },
    {
        .name = "dte_config_transpose_cdte",
        .kernel = KernelDteConfigTranspose,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {64, 2, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 422},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {64, 2, 8, 32},
        .transpose_layout = {1, 0, 2, 3},
        .direction = 0,
    },
    {
        .name = "dte_config_transpose_sdte",
        .kernel = KernelDteConfigTranspose,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {64, 2, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 421},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {64, 2, 8, 32},
        .transpose_layout = {1, 0, 2, 3},
        .direction = 0,
    },
    {
        .name = "dte_config_pad_cdte",
        .kernel = KernelDteConfigPad,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 16, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {8, 16, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 320},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 16, 8, 32},
        .dst_dims = {8, 16, 8, 32},
        .pad_low = {2, 0, 0, 0},
        .pad_high = {2, 0, 0, 0},
        .pad_mid = {2, 0, 0, 0},
        .value = 123,
        .direction = 0,
    },
    {
        .name = "dte_config_pad_sdte",
        .kernel = KernelDteConfigPad,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 16, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {8, 16, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 320},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 16, 8, 32},
        .dst_dims = {8, 16, 8, 32},
        .pad_low = {2, 0, 0, 0},
        .pad_high = {2, 0, 0, 0},
        .pad_mid = {2, 0, 0, 0},
        .value = 123,
        .direction = 0,
    },
    {
        .name = "dte_config_memset_cdte",
        .kernel = KernelDteConfigMemset,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 76},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .value = 123,
        .direction = 0,
    },
    {
        .name = "dte_config_memset_sdte",
        .kernel = KernelDteConfigMemset,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 74},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .value = 123,
        .direction = 0,
    },
    {
        .name = "dte_config_mirror_lr_cdte",
        .kernel = KernelDteConfigMirrorLr,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 98},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .direction = 0,
    },
    {
        .name = "dte_config_mirror_lr_sdte",
        .kernel = KernelDteConfigMirrorLr,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 98},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .direction = 0,
    },
    {
        .name = "dte_config_mirror_tb_cdte",
        .kernel = KernelDteConfigMirrorTb,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 98},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .direction = 0,
    },
    {
        .name = "dte_config_mirror_tb_sdte",
        .kernel = KernelDteConfigMirrorTb,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 98},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .direction = 0,
    },
    {.name = "dte_config_shrink_sdte",
     .kernel = KernelDteConfigShrink,
     .input_type = dte_fixture::UIntType32,
     .output_type = dte_fixture::UIntType16,
     .input_data = krt_benchmark::GenerateUniformTestData<uint32_t>(
         {2, 64, 8, 32}, 0x12345678u),
     .ref_data = krt_benchmark::GenerateUniformTestData<uint16_t>(
         {2, 64, 8, 32}, 0x5678u),
     .use_cdte = false,
     .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
         {{200, 99999},
          {210, 99999},
          {300, 93},
          {400, 99999}}), // FIXME: golden cycle.

     .src_dims = {2, 64, 8, 32},
     .dst_dims = {2, 64, 8, 32},
     .phase = 0,
     .direction = 0,
     .ratio = 0},
    {.name = "dte_config_shrink_cdte",
     .kernel = KernelDteConfigShrink,
     .input_type = dte_fixture::UIntType32,
     .output_type = dte_fixture::UIntType16,
     .input_data = krt_benchmark::GenerateUniformTestData<uint32_t>(
         {2, 64, 8, 32}, 0x12345678u),
     .ref_data = krt_benchmark::GenerateUniformTestData<uint16_t>(
         {2, 64, 8, 32}, 0x5678u),
     .use_cdte = true,
     .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
         {{200, 99999},
          {210, 99999},
          {300, 95},
          {400, 99999}}), // FIXME: golden cycle.

     .src_dims = {2, 64, 8, 32},
     .dst_dims = {2, 64, 8, 32},
     .phase = 0,
     .direction = 0,
     .ratio = 0},
    {.name = "dte_config_expand_sdte",
     .kernel = KernelDteConfigExpand,
     .input_type = dte_fixture::UIntType16,
     .output_type = dte_fixture::UIntType32,
     .input_data = krt_benchmark::GenerateUniformTestData<uint16_t>(
         {2, 64, 8, 32}, 0x5678u),
     .ref_data = krt_benchmark::GenerateUniformTestData<uint32_t>(
         {2, 64, 8, 32}, 0x56780000u),
     .use_cdte = false,
     .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
         {{200, 99999},
          {210, 99999},
          {300, 151},
          {400, 99999}}), // FIXME: golden cycle.

     .src_dims = {2, 64, 8, 32},
     .dst_dims = {2, 64, 8, 32},
     .phase = 1,
     .ratio = 0},
    {.name = "dte_config_expand_cdte",
     .kernel = KernelDteConfigExpand,
     .input_type = dte_fixture::UIntType16,
     .output_type = dte_fixture::UIntType32,
     .input_data = krt_benchmark::GenerateUniformTestData<uint16_t>(
         {2, 64, 8, 32}, 0x5678u),
     .ref_data = krt_benchmark::GenerateUniformTestData<uint32_t>(
         {2, 64, 8, 32}, 0x56780000u),
     .use_cdte = true,
     .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
         {{200, 99999},
          {210, 99999},
          {300, 153},
          {400, 99999}}), // FIXME: golden cycle.

     .src_dims = {2, 64, 8, 32},
     .dst_dims = {2, 64, 8, 32},
     .phase = 1,
     .ratio = 0},
    {
        .name = "dte_trigger_cdte",
        .kernel = KernelDteTrigger,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 74},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .value = 123,
        .direction = 0,
    },

#if __GCU_ARCH__ >= 300

    {
        .name = "dte_config_slice_pad_sdte",
        .kernel = KernelDteConfigSlicePad,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 290},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .slice_dims = {2, 2, 8, 32},
        .src_offsets = {0, 0, 0, 0},
        .pad_low = {0, 31, 0, 0},
        .pad_high = {0, 31, 0, 0},
        .pad_mid = {0, 0, 0, 0},
        .value = 123,
        .direction = 0,
    },
    {
        .name = "dte_config_slice_pad_cdte",
        .kernel = KernelDteConfigSlicePad,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 290},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .slice_dims = {2, 2, 8, 32},
        .src_offsets = {0, 0, 0, 0},
        .pad_low = {0, 31, 0, 0},
        .pad_high = {0, 31, 0, 0},
        .pad_mid = {0, 0, 0, 0},
        .value = 123,
        .direction = 0,
    },
    {
        .name = "dte_config_slice_deslice_sdte",
        .kernel = KernelDteConfigSliceDeslice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 288},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .slice_dims = {2, 64, 8, 32},
        .src_offsets = {0, 0, 0, 0},
        .dst_offsets = {0, 0, 0, 0},
        .value = 0,
        .direction = 0,
    },
    {
        .name = "dte_config_slice_deslice_cdte",
        .kernel = KernelDteConfigSliceDeslice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 288},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .slice_dims = {2, 64, 8, 32},
        .src_offsets = {0, 0, 0, 0},
        .dst_offsets = {0, 0, 0, 0},
        .value = 0,
        .direction = 0,
    },
    {
        .name = "dte_config_slice_broadcast_sdte",
        .kernel = KernelDteConfigSliceBroadcast,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 227},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .slice_dims = {2, 1, 8, 32},
        .src_offsets = {0, 0, 0, 0},
        .direction = 0,
    },
    {
        .name = "dte_config_slice_broadcast_cdte",
        .kernel = KernelDteConfigSliceBroadcast,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 227},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .slice_dims = {2, 1, 8, 32},
        .src_offsets = {0, 0, 0, 0},
        .direction = 0,
    },
    {.name = "dte_config_slice_expand_sdte",
     .kernel = KernelDteConfigSliceExpand,
     .input_type = dte_fixture::UIntType16,
     .output_type = dte_fixture::UIntType32,
     .input_data = krt_benchmark::GenerateUniformTestData<uint16_t>(
         {1, 64, 8, 32}, 0x1234),
     .ref_data = krt_benchmark::GenerateUniformTestData<uint32_t>(
         {1, 64, 8, 32}, 0x1234),
     .use_cdte = false,
     .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
         {{200, 99999},
          {210, 99999},
          {300, 144},
          {400, 99999}}), // FIXME: golden cycle.

     .src_dims = {1, 64, 8, 32},
     .dst_dims = {1, 64, 8, 32},
     .slice_dims = {1, 64, 8, 32},
     .src_offsets = {0, 0, 0, 0},
     .phase = 0,
     .direction = 0,
     .ratio = 0},
    {.name = "dte_config_slice_expand_cdte",
     .kernel = KernelDteConfigSliceExpand,
     .input_type = dte_fixture::UIntType8,
     .output_type = dte_fixture::UIntType32,
     .input_data =
         krt_benchmark::GenerateUniformTestData<uint8_t>({1, 64, 8, 32}, 1),
     .ref_data = krt_benchmark::GenerateUniformTestData<uint32_t>(
         {1, 64, 8, 32}, 0x01000000),
     .use_cdte = true,
     .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
         {{200, 99999},
          {210, 99999},
          {300, 144},
          {400, 99999}}), // FIXME: golden cycle.

     .src_dims = {1, 64, 8, 32},
     .dst_dims = {1, 64, 8, 32},
     .slice_dims = {1, 64, 8, 32},
     .src_offsets = {0, 0, 0, 0},
     .phase = 3,
     .direction = 0,
     .ratio = 1},
    {.name = "dte_config_shrink_deslice_sdte",
     .kernel = KernelDteConfigShrinkDeslice,
     .input_type = dte_fixture::UIntType32,
     .output_type = dte_fixture::UIntType16,
     .input_data = krt_benchmark::GenerateUniformTestData<uint32_t>(
         {1, 64, 8, 32}, 0x12345678),
     .ref_data = krt_benchmark::GenerateUniformTestData<uint16_t>(
         {1, 64, 8, 32}, 0x5678),
     .use_cdte = false,
     .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
         {{200, 99999},
          {210, 99999},
          {300, 165},
          {400, 99999}}), // FIXME: golden cycle.

     .src_dims = {1, 64, 8, 32},
     .dst_dims = {1, 64, 8, 32},
     .slice_dims = {1, 64, 8, 32},
     .dst_offsets = {0, 0, 0, 0},
     .phase = 0,
     .direction = 0,
     .ratio = 0},
    {.name = "dte_config_shrink_deslice_cdte",
     .kernel = KernelDteConfigShrinkDeslice,
     .input_type = dte_fixture::UIntType32,
     .output_type = dte_fixture::UIntType8,
     .input_data = krt_benchmark::GenerateUniformTestData<uint32_t>(
         {1, 64, 8, 32}, 0x12345678),
     .ref_data =
         krt_benchmark::GenerateUniformTestData<uint8_t>({1, 64, 8, 32}, 0x78),
     .use_cdte = true,
     .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
         {{200, 99999},
          {210, 99999},
          {300, 165},
          {400, 99999}}), // FIXME: golden cycle.

     .src_dims = {1, 64, 8, 32},
     .dst_dims = {1, 64, 8, 32},
     .slice_dims = {1, 64, 8, 32},
     .dst_offsets = {0, 0, 0, 0},
     .phase = 0,
     .direction = 0,
     .ratio = 1},
    {
        .name = "dte_config_memset_deslice_sdte",
        .kernel = KernelDteConfigMemsetDeslice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 1),
        .ref_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 1),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 147},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .dst_offsets = {0, 0, 0, 0},
        .value = 1,
        .direction = 0,
    },
    {
        .name = "dte_config_memset_deslice_cdte",
        .kernel = KernelDteConfigMemsetDeslice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 1),
        .ref_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 1),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 147},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .dst_offsets = {0, 0, 0, 0},
        .value = 1,
        .direction = 0,
    },
    {
        .name = "dte_config_mirror_lr_pad_sdte",
        .kernel = KernelDteConfigMirrorLrPad,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 1),
        .ref_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 66, 8, 32}, 1),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 346},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 66, 8, 32},
        .pad_low = {0, 1, 0, 0},
        .pad_high = {0, 1, 0, 0},
        .pad_mid = {0, 0, 0, 0},
        .value = 1,
        .direction = 0,
    },
    {
        .name = "dte_config_mirror_lr_pad_cdte",
        .kernel = KernelDteConfigMirrorLrPad,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 1),
        .ref_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 66, 8, 32}, 1),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 347},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 66, 8, 32},
        .pad_low = {0, 1, 0, 0},
        .pad_high = {0, 1, 0, 0},
        .pad_mid = {0, 0, 0, 0},
        .value = 1,
        .direction = 0,
    },
    {
        .name = "dte_config_mirror_tb_pad_sdte",
        .kernel = KernelDteConfigMirrorTbPad,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 1),
        .ref_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 66, 8, 32}, 1),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 344},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 66, 8, 32},
        .pad_low = {0, 1, 0, 0},
        .pad_high = {0, 1, 0, 0},
        .pad_mid = {0, 0, 0, 0},
        .value = 1,
        .direction = 0,
    },
    {
        .name = "dte_config_mirror_tb_pad_cdte",
        .kernel = KernelDteConfigMirrorTbPad,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 1),
        .ref_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 66, 8, 32}, 1),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 345},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 66, 8, 32},
        .pad_low = {0, 1, 0, 0},
        .pad_high = {0, 1, 0, 0},
        .pad_mid = {0, 0, 0, 0},
        .value = 1,
        .direction = 0,
    },
    {
        .name = "dte_config_mirror_lr_deslice_sdte",
        .kernel = KernelDteConfigMirrorLrDeslice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 124},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .dst_offsets = {0, 0, 0, 0},
        .direction = 0,
    },
    {
        .name = "dte_config_mirror_lr_deslice_cdte",
        .kernel = KernelDteConfigMirrorLrDeslice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 124},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .dst_offsets = {0, 0, 0, 0},
        .direction = 0,
    },
    {
        .name = "dte_config_mirror_tb_deslice_sdte",
        .kernel = KernelDteConfigMirrorTbDeslice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 124},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .dst_offsets = {0, 0, 0, 0},
        .direction = 0,
    },
    {
        .name = "dte_config_mirror_tb_deslice_cdte",
        .kernel = KernelDteConfigMirrorTbDeslice,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .ref_data = krt_benchmark::GenerateUniformTestData<int32_t>(
            {2, 64, 8, 32}, 123),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 124},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .dst_offsets = {0, 0, 0, 0},
        .direction = 0,
    },
    {
        .name = "dte_config_sub_sample_sdte",
        .kernel = KernelDteConfigSubSample,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 1),
        .ref_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 1),
        .use_cdte = false,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 138},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .direction = 0,
        .dim_stride = 1,
    },
    {
        .name = "dte_config_sub_sample_cdte",
        .kernel = KernelDteConfigSubSample,
        .input_type = dte_fixture::IntType32,
        .output_type = dte_fixture::IntType32,
        .input_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 1),
        .ref_data =
            krt_benchmark::GenerateUniformTestData<int32_t>({2, 64, 8, 32}, 1),
        .use_cdte = true,
        .golden_cycle = krt_benchmark::GoldenCyclesByGcuArch(
            {{200, 99999},
             {210, 99999},
             {300, 138},
             {400, 99999}}), // FIXME: golden cycle.

        .src_dims = {2, 64, 8, 32},
        .dst_dims = {2, 64, 8, 32},
        .direction = 0,
        .dim_stride = 1,
    },

#endif // __GCU_ARCH__ >= 300
#endif
};

using BenchmarkDteConfig = dte_fixture::BenchmarkDteFixture;
TEST_P(BenchmarkDteConfig, Test) { Run(); }
INSTANTIATE_TEST_SUITE_P(Benchmark, BenchmarkDteConfig,
                         ::testing::ValuesIn(dte_config_test_set));
