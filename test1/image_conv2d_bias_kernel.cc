/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include "conv2d_bias_kernel.h"

#include "c_src/conv2d_bias_fp16_ci4_rs3_stride1.h"
#include "c_src/conv2d_bias_fp16_ci3_rs3_stride1.h"

template <typename T>
__global__ void image_conv2d_bias_kernel(T* out, T* lhs, T* rhs, T* bias,
                                         CONV2D_OP_PARAS conv_args) {
  int32_t sip_id = threadIdx.z * (blockDim.x * blockDim.y) +
                   threadIdx.y * blockDim.x + threadIdx.x;
  int32_t clu_id = blockIdx.z * (gridDim.x * gridDim.y) +
                   blockIdx.y * gridDim.x + blockIdx.x;

  int32_t kernel_id = conv_args.kernel_index - CONV2D_GENERAL_KERNEL_NUM;
  int32_t bpe = 4;
  int32_t reduce_align = 16;
  if (std::is_same<T, float>::value) {
    bpe = 4;
    reduce_align = 16;
  } else if (std::is_same<T, tops::half>::value) {
    bpe = 2;
    reduce_align = 32;
  } else if (std::is_same<T, tops::bfloat>::value) {
    bpe = 2;
    reduce_align = 32;
  }

  // configure conv2d parameters
  int32_t padding_top = conv_args.padding_top;
  int32_t padding_bottom = conv_args.padding_bottom;
  int32_t padding_left = conv_args.padding_left;
  int32_t padding_right = conv_args.padding_right;
  int32_t stride_h = conv_args.stride_h;
  int32_t stride_w = conv_args.stride_w;
  int32_t window_dilation_h = conv_args.window_dilation_h;
  int32_t window_dilation_w = conv_args.window_dilation_w;
  int32_t activation_mode = conv_args.activation_mode;
  float coef = conv_args.coef;
  using u_t = typename cc_kernel::UnderlyingType<T>::type;

  // configure l3 parameters
  int32_t l3_n_dim = conv_args.l3_n_dim;
  int32_t l3_hi_dim = conv_args.l3_hi_dim;
  int32_t l3_wi_dim = conv_args.l3_wi_dim;
  int32_t l3_ci_dim = conv_args.l3_ci_dim;
  int32_t l3_r_dim = conv_args.l3_r_dim;
  int32_t l3_s_dim = conv_args.l3_s_dim;
  int32_t l3_co_dim = conv_args.l3_co_dim;
  int32_t l3_ho_dim = conv_args.l3_ho_dim;
  int32_t l3_wo_dim = conv_args.l3_wo_dim;
  int32_t l3_wixci_dim = l3_wi_dim * l3_ci_dim;

  // configure l2 parameters
  int32_t l2_n_dim = conv_args.l2_n_dim[clu_id];
  int32_t l2_ho_dim = conv_args.l2_ho_dim[clu_id];
  int32_t l2_wo_dim = conv_args.l2_wo_dim[clu_id];
  int32_t l2_co_dim = conv_args.l2_co_dim[clu_id];
  int32_t l2_n_offset = conv_args.l2_n_offset[clu_id];
  int32_t l2_ho_offset = conv_args.l2_ho_offset[clu_id];
  int32_t l2_wo_offset = conv_args.l2_wo_offset[clu_id];
  int32_t l2_co_offset = conv_args.l2_co_offset[clu_id];

  // configure l1 parameters
  int32_t l1_n_step = conv_args.l1_n_step;
  int32_t l1_hi_step = conv_args.l1_hi_step;
  int32_t l1_wi_step = conv_args.l1_wi_step;
  int32_t l1_ci_step = conv_args.l1_ci_step;
  int32_t l1_r_step = conv_args.l1_r_step;
  int32_t l1_s_step = conv_args.l1_s_step;
  int32_t l1_co_step = conv_args.l1_co_step;
  int32_t l1_ho_step = conv_args.l1_ho_step;
  int32_t l1_wo_step = conv_args.l1_wo_step;
  int32_t l1_sxci_padding = ALIGN_UP((l1_s_step * l1_ci_step), reduce_align);
  int32_t l1_wixci_step = ALIGN_UP(((l1_wo_step - 1) * stride_w * l1_ci_step +
                          l1_sxci_padding), reduce_align);

  // configure thread offset view on l2_buffer
  int32_t l1_n_offset = conv_args.l1_n_offset[sip_id];
  int32_t l1_ho_offset = conv_args.l1_ho_offset[sip_id];
  int32_t l1_wo_offset = conv_args.l1_wo_offset[sip_id];
  int32_t l1_co_offset = conv_args.l1_co_offset[sip_id];

  // thread offset view on l3_buffer
  int32_t l3_n_offset = l2_n_offset + l1_n_offset;
  int32_t l3_ho_offset = l2_ho_offset + l1_ho_offset;
  int32_t l3_wo_offset = l2_wo_offset + l1_wo_offset;
  int32_t l3_co_offset = l2_co_offset + l1_co_offset;
  // real loop num on thread
  int32_t l1_n_dim = std::max(0, std::min(l2_n_dim - l1_n_offset,
                              conv_args.l1_n_dim[sip_id]));
  int32_t l1_ho_dim = std::max(0, std::min(l2_ho_dim - l1_ho_offset,
                               conv_args.l1_ho_dim[sip_id]));
  int32_t l1_wo_dim = std::max(0, std::min(l2_wo_dim - l1_wo_offset,
                               conv_args.l1_wo_dim[sip_id]));
  int32_t l1_co_dim = std::max(0, std::min(l2_co_dim - l1_co_offset,
                               conv_args.l1_co_dim[sip_id]));

  bool merge_woco = false;
  if (l3_co_dim <= l1_co_step) {
    merge_woco = true;
  }

  // configure l3 buffer
  int32_t l3_lhs_shape[3] = {l3_n_dim,  l3_hi_dim, l3_wixci_dim};
  int32_t l3_rhs_shape[4] = {l3_co_dim, l3_r_dim,  l3_s_dim,  l3_ci_dim};
  int32_t l3_out_shape_3d[3] = {l3_n_dim, l3_ho_dim, l3_wo_dim * l3_co_dim};
  int32_t l3_out_shape_4d[4] = {l3_n_dim, l3_ho_dim, l3_wo_dim, l3_co_dim};
  int32_t l3_bias_shape[1] = {l3_co_dim};

  tops::mdspan l3_lhs(tops::Global, lhs, l3_lhs_shape);
  tops::mdspan l3_rhs(tops::Global, rhs, l3_rhs_shape);
  tops::mdspan l3_out_3d(tops::Global, out, l3_out_shape_3d);
  tops::mdspan l3_out_4d(tops::Global, out, l3_out_shape_4d);
  tops::mdspan l3_bias(tops::Global, bias, l3_bias_shape);

  // configure l2 buffer
  int32_t l3_co_dim_align = ALIGN_UP(l3_co_dim, l1_co_step);
  int32_t l3_co_pad = l3_co_dim_align - l3_co_dim;
  int32_t l2_rhs_shape[4] = {l3_co_dim_align, l3_r_dim, l3_s_dim, l3_ci_dim};
  int32_t l2_rhs_cpy_shape[4] = {l3_co_dim, l3_r_dim, l3_s_dim, l3_ci_dim};
  int32_t l2_rhs_set_shape[4] = {l3_co_pad, l3_r_dim, l3_s_dim, l3_ci_dim};
  int32_t l2_rhs_layout[4] = {1, 2, 3, 0};
  // configure buffer size
  int32_t l2_rhs_cpy_num = l2_rhs_cpy_shape[0] * l2_rhs_cpy_shape[1] *
                           l2_rhs_cpy_shape[2] * l2_rhs_cpy_shape[3];
  int32_t l2_rhs_set_num = l2_rhs_set_shape[0] * l2_rhs_set_shape[1]
                         * l2_rhs_set_shape[2] * l2_rhs_set_shape[3];
  int32_t l2_out_num = l1_n_step * l1_ho_step * l1_wo_step * l3_co_dim;

  extern __shared__ __valigned__ char l2_total_buf[];
  T* l2_rhs_buf = reinterpret_cast<T*>(l2_total_buf);
  T* l2_rhs_set_buf = l2_rhs_buf + l2_rhs_cpy_num;
  tops::mdspan l2_rhs(tops::Shared, l2_rhs_buf, l2_rhs_shape);
  tops::mdspan l2_rhs_cpy(tops::Shared, l2_rhs_buf, l2_rhs_cpy_shape);
  tops::mdspan l2_rhs_set(tops::Shared, l2_rhs_set_buf, l2_rhs_set_shape);

  T* l2_out_buf = l2_rhs_set_buf + l2_rhs_set_num + threadIdx.x * l2_out_num;

  int32_t l2_out_shape_3d[3] = {l1_n_step, l1_ho_step, l1_wo_step * l3_co_dim};
  int32_t l2_out_shape_4d[4] = {l1_n_step, l1_ho_step, l1_wo_step, l3_co_dim};
  tops::mdspan l2_out_3d(tops::Shared, l2_out_buf, l2_out_shape_3d);
  tops::mdspan l2_out_4d(tops::Shared, l2_out_buf, l2_out_shape_4d);

  // configure l1 buffer
  int32_t l1_lhs_shape[3] = {l1_n_step,  l1_hi_step, l1_wixci_step};
  int32_t l1_rhs_shape[4] = {l1_r_step,  l1_s_step,  l1_ci_step, l1_co_step};
  int32_t l1_out_shape[4] = {l1_n_step,  l1_ho_step, l1_wo_step, l1_co_step};
  int32_t l1_bias_shape[1] = {l3_co_dim_align};
  // configure buffer size
  int32_t l1_lhs_size = l1_lhs_shape[0] * l1_lhs_shape[1] * l1_lhs_shape[2];
  int32_t l1_rhs_size =
      l1_rhs_shape[0] * l1_rhs_shape[1] * l1_rhs_shape[2] * l1_rhs_shape[3];
  int32_t l1_out_size =
      l1_out_shape[0] * l1_out_shape[1] * l1_out_shape[2] * l1_out_shape[3];
  __local__ __valigned__ char l1_total_buf[VDMEM_SIZE];
  int32_t l1_align_size = 4 * L1_ALIGN_SIZE / bpe;
  T* l1_lhs_buf0 = reinterpret_cast<T*>(l1_total_buf);
  T* l1_lhs_buf1 = l1_lhs_buf0 + ALIGN_UP(l1_lhs_size, l1_align_size);
  T* l1_rhs_buf0 = l1_lhs_buf1 + ALIGN_UP(l1_lhs_size, l1_align_size);
  T* l1_rhs_buf1 = l1_rhs_buf0 + ALIGN_UP(l1_rhs_size, l1_align_size);
  T* l1_out_buf0 = l1_rhs_buf1 + ALIGN_UP(l1_rhs_size, l1_align_size);
  T* l1_out_buf1 = l1_out_buf0 + ALIGN_UP(l1_out_size, l1_align_size);
  T* l1_bias_buf = l1_out_buf1 + ALIGN_UP(l1_out_size, l1_align_size);
  tops::mdspan l1_lhs0(tops::Private, l1_lhs_buf0, l1_lhs_shape);
  tops::mdspan l1_lhs1(tops::Private, l1_lhs_buf1, l1_lhs_shape);
  tops::mdspan l1_rhs0(tops::Private, l1_rhs_buf0, l1_rhs_shape);
  tops::mdspan l1_rhs1(tops::Private, l1_rhs_buf1, l1_rhs_shape);
  tops::mdspan l1_out0(tops::Private, l1_out_buf0, l1_out_shape);
  tops::mdspan l1_out1(tops::Private, l1_out_buf1, l1_out_shape);
  tops::mdspan l1_bias(tops::Private, l1_bias_buf, l1_bias_shape);

  // configure cdte and sdte
  __private_dte__ tops_dte_ctx_t cdte_ctx_rhs;
  __private_dte__ tops_dte_ctx_t cdte_ctx_out;
  tops_dte_ctx_t sdte_ctx_lhs0;
  tops_dte_ctx_t sdte_ctx_lhs1;
  tops_dte_ctx_t sdte_ctx_rhs0;
  tops_dte_ctx_t sdte_ctx_rhs1;
  tops_dte_ctx_t sdte_ctx_out;
  tops_dte_ctx_t sdte_ctx_bias;
  tops::dte_scope scope_lhs0(sdte_ctx_lhs0);
  tops::dte_scope scope_lhs1(sdte_ctx_lhs1);
  tops::dte_scope scope_rhs0(sdte_ctx_rhs0);
  tops::dte_scope scope_rhs1(sdte_ctx_rhs1);
  tops::dte_scope scope_bias(sdte_ctx_bias);

  sdte_ctx_out.init();
  cdte_ctx_out.init();

  if (merge_woco) {
    sdte_ctx_out.connect(cdte_ctx_out);

    int32_t l2_out_offset[4] = {0, 0, 0, 0};
    sdte_ctx_out.config_deslice(l2_out_4d, l1_out0, l2_out_offset);
    cdte_ctx_out.config_deslice(l3_out_3d, l2_out_3d, {0, 0, 0});
  } else {
    sdte_ctx_out.config_deslice(l3_out_4d, l1_out0, {0, 0, 0, 0});
  }

  tops::event event_lhs0;
  tops::event event_lhs1;
  tops::event event_rhs0;
  tops::event event_rhs1;
  tops::event event_out;
  tops::event private_event_out;
  tops::event event_bias;

  // pre load bias
  event_bias = tops::memcpy_async(sdte_ctx_bias, l1_bias, l3_bias);
  // pre load rhs in the beginning
  if (threadIdx.x == 0) {
    cdte_ctx_rhs.init();
    event_rhs0 = tops::memcpy_async(cdte_ctx_rhs, l2_rhs_cpy, l3_rhs);
    if (l3_co_pad > 0) {
      tops::memset(sdte_ctx_rhs0, l2_rhs_set, T(0));
    }
    tops::wait(event_rhs0);
    cdte_ctx_rhs.destroy();
  }
  tops::wait(event_bias);

  __syncthreads();


  if (l1_n_dim > 0 && l1_co_dim > 0 && l1_ho_dim > 0 && l1_wo_dim > 0) {
    // main loop
    bool first_load_flag = true;
    bool rhs_dma_wait_flag = true;
    bool rhs_dma_work_flag = false;
    bool out_dma_flag = false;
    int32_t lhs_pp_cnt = 0;
    int32_t rhs_pp_cnt = 0;
    int32_t out_pp_cnt = 0;
    int32_t cur_n_offset = -1;
    int32_t cur_ho_offset = -1;
    int32_t cur_wo_offset = -1;
    int32_t cur_co_offset = -1;
    int32_t pre_n_offset = -1;
    int32_t pre_ho_offset = -1;
    int32_t pre_wo_offset = -1;
    int32_t pre_co_offset = -1;
    long lhs_addr, rhs_addr, out_addr, bias_addr;

    // pre dims info for func_call
    int32_t pre_l1_n_step = l1_n_step;

    for (int32_t n = 0; n < l1_n_dim; n += l1_n_step) {
      cur_n_offset = l3_n_offset + n;
      for (int32_t ho = 0; ho < l1_ho_dim; ho += l1_ho_step) {
        cur_ho_offset = l3_ho_offset + ho;
        int32_t l1_hi_begin = cur_ho_offset * stride_h;
        int32_t l1_hi_end = l1_hi_begin + l1_hi_step - 1;
        bool lhs_hi_flag = l1_hi_end >= padding_top &&
                           l1_hi_begin < l3_hi_dim + padding_top;
        int32_t l3_hi_offset = l1_hi_begin - padding_top;
        for (int32_t wo = 0; wo < l1_wo_dim; wo += l1_wo_step) {
          cur_wo_offset = l3_wo_offset + wo;
          int32_t l1_wi_begin = cur_wo_offset * stride_w;
          int32_t l1_wi_end = l1_wi_begin + l1_wi_step - 1;
          bool lhs_wi_flag = l1_wi_end >= padding_left &&
                             l1_wi_begin < l3_wi_dim + padding_left;
          int32_t l3_wixci_offset = (l1_wi_begin - padding_left) * l1_ci_step;
          // prepare lhs dte
          int32_t l3_lhs_offset[3] = {
            cur_n_offset, l3_hi_offset, l3_wixci_offset
          };
          auto cur_l1_lhs = (lhs_pp_cnt % 2) ? &l1_lhs1 : &l1_lhs0;
          auto cur_sdte_ctx_lhs =
                (lhs_pp_cnt % 2) ? &sdte_ctx_lhs1 : &sdte_ctx_lhs0;
          auto cur_event_lhs = (lhs_pp_cnt % 2) ? &event_lhs1 : &event_lhs0;

          // load lhs from l3 to l1
          if (lhs_hi_flag && lhs_wi_flag) {
            *cur_event_lhs = tops::slice_async(*cur_sdte_ctx_lhs, *cur_l1_lhs,
                                               l3_lhs, l3_lhs_offset);
          } else {
            *cur_event_lhs =
                tops::memset_async(*cur_sdte_ctx_lhs, *cur_l1_lhs, T(0));
          }

          for (int32_t co = 0; co < l1_co_dim; co += l1_co_step) {
            cur_co_offset = l3_co_offset + co;

            // prepare rhs
            if (pre_co_offset != cur_co_offset) {
              int32_t l2_rhs_offset[4] = {cur_co_offset, 0, 0, 0};
              auto cur_l1_rhs = (rhs_pp_cnt % 2) ? &l1_rhs1 : &l1_rhs0;
              auto cur_sdte_ctx_rhs =
                    (rhs_pp_cnt % 2) ? &sdte_ctx_rhs1 : &sdte_ctx_rhs0;
              auto cur_event_rhs =
                    (rhs_pp_cnt % 2) ? &event_rhs1 : &event_rhs0;
              *cur_event_rhs = tops::slice_transpose_async(
                  *cur_sdte_ctx_rhs, *cur_l1_rhs, l2_rhs, l2_rhs_offset,
                  l2_rhs_layout);
              rhs_dma_work_flag = true;
            } else {
              rhs_dma_work_flag = false;
            }

            if (!first_load_flag) {
              // wait dma lhs
              if (pre_co_offset == l3_co_offset /*pre_co == 0*/) {
                auto pre_event_lhs =
                      (lhs_pp_cnt % 2) ? &event_lhs0 : &event_lhs1;
                tops::wait(*pre_event_lhs);
              }
              // wait dma rhs
              if (rhs_dma_wait_flag) {
                auto pre_event_rhs =
                      (rhs_pp_cnt % 2) ? &event_rhs0 : &event_rhs1;
                tops::wait(*pre_event_rhs);
              }

              // call kernel function
              lhs_addr = (long)((lhs_pp_cnt % 2) ? l1_lhs_buf0 : l1_lhs_buf1);
              rhs_addr = (long)((rhs_pp_cnt % 2) ? l1_rhs_buf0 : l1_rhs_buf1);
              out_addr = (long)((out_pp_cnt % 2) ? l1_out_buf1 : l1_out_buf0);
              bias_addr = (long)(l1_bias_buf + pre_co_offset);

              IMAGE_CONV2D_BIAS_KERNEL_CALL(
                  lhs_addr, rhs_addr, out_addr, bias_addr, pre_l1_n_step,
                  l1_hi_step, l1_wixci_step, l1_ho_step, l1_wo_step, l1_co_step,
                  1, kernel_id);
              ACTIVATION_KERNEL_CALL(out_addr, l1_out_size, activation_mode,
                                     coef, 1);

              // dma out
              if (out_dma_flag) {
                if (merge_woco) {
                  tops::wait(private_event_out);
                } else {
                  tops::wait(event_out);
                }
              }

              // auto cur_l1_out = (out_pp_cnt % 2) ? &l1_out1 : &l1_out0;
              T* cur_l1_out_buf = (out_pp_cnt % 2) ? l1_out_buf1 : l1_out_buf0;

              cur_l1_out_buf[0] += 0.5;

              if (merge_woco) {
                int32_t l3_out_offset_3d[3] = {pre_n_offset, pre_ho_offset,
                                               pre_wo_offset * l3_co_dim};

                // tops::deslice_async(sdte_ctx_out, l2_out_4d,
                //                    *cur_l1_out, l2_out_offset);
                // event_out = tops::deslice_async(cdte_ctx_out, l3_out_3d,
                //                                 l2_out_3d, l3_out_offset_3d);

                sdte_ctx_out.set_src_addr(cur_l1_out_buf);
                event_out = sdte_ctx_out.trigger();

                cdte_ctx_out.set_dst_offset(0, l3_out_offset_3d[0]);
                cdte_ctx_out.set_dst_offset(1, l3_out_offset_3d[1]);
                cdte_ctx_out.set_dst_offset(2, l3_out_offset_3d[2]);
                private_event_out = cdte_ctx_out.trigger();
              } else {
                int32_t l3_out_offset_4d[4] = {pre_n_offset, pre_ho_offset,
                                               pre_wo_offset, pre_co_offset};

                // event_out = tops::deslice_async(
                //                       sdte_ctx_out, l3_out_4d,
                //                      *cur_l1_out, l3_out_offset_4d);

                sdte_ctx_out.set_src_addr(cur_l1_out_buf);
                sdte_ctx_out.set_dst_offset(0, l3_out_offset_4d[0]);
                sdte_ctx_out.set_dst_offset(1, l3_out_offset_4d[1]);
                sdte_ctx_out.set_dst_offset(2, l3_out_offset_4d[2]);
                sdte_ctx_out.set_dst_offset(3, l3_out_offset_4d[3]);
                event_out = sdte_ctx_out.trigger();
              }

              out_pp_cnt++;
              out_dma_flag = true;
            }

            // record loop info
            pre_n_offset = cur_n_offset;
            pre_ho_offset = cur_ho_offset;
            pre_wo_offset = cur_wo_offset;
            pre_co_offset = cur_co_offset;
            pre_l1_n_step = std::min(l1_n_dim - cur_n_offset, l1_n_step);
            if (rhs_dma_work_flag) {
              rhs_pp_cnt++;
              rhs_dma_wait_flag = true;
            } else {
              rhs_dma_wait_flag = false;
            }
            if (co == 0) {
              lhs_pp_cnt++;
            }
            first_load_flag = false;
          }
        }
      }
    }

    // last call
    if (pre_co_offset == l3_co_offset) {
      auto pre_event_lhs = (lhs_pp_cnt % 2) ? &event_lhs0 : &event_lhs1;
      tops::wait(*pre_event_lhs);
    }
    if (rhs_dma_wait_flag) {
      auto pre_event_rhs = (rhs_pp_cnt % 2) ? &event_rhs0 : &event_rhs1;
      tops::wait(*pre_event_rhs);
    }

    // call kernel function
    lhs_addr = (long)((lhs_pp_cnt % 2) ? l1_lhs_buf0 : l1_lhs_buf1);
    rhs_addr = (long)((rhs_pp_cnt % 2) ? l1_rhs_buf0 : l1_rhs_buf1);
    out_addr = (long)((out_pp_cnt % 2) ? l1_out_buf1 : l1_out_buf0);
    bias_addr = (long)(l1_bias_buf + pre_co_offset);

    IMAGE_CONV2D_BIAS_KERNEL_CALL(
        lhs_addr, rhs_addr, out_addr, bias_addr, pre_l1_n_step, l1_hi_step,
        l1_wixci_step, l1_ho_step, l1_wo_step, l1_co_step, 1, kernel_id);
    ACTIVATION_KERNEL_CALL(out_addr, l1_out_size, activation_mode, coef, 1);

    if (out_dma_flag) {
      if (merge_woco) {
        tops::wait(private_event_out);
      } else {
        tops::wait(event_out);
      }
    }

    // auto l1_out = (out_pp_cnt % 2) ? &l1_out1 : &l1_out0;
    T* cur_l1_out_buf = (out_pp_cnt % 2) ? l1_out_buf1 : l1_out_buf0;

    if (merge_woco) {
      int32_t l3_out_offset_3d[3] = {pre_n_offset, pre_ho_offset,
                                      pre_wo_offset * l3_co_dim};

      sdte_ctx_out.set_src_addr(cur_l1_out_buf);
      event_out = sdte_ctx_out.trigger();

      cdte_ctx_out.set_dst_offset(0, l3_out_offset_3d[0]);
      cdte_ctx_out.set_dst_offset(1, l3_out_offset_3d[1]);
      cdte_ctx_out.set_dst_offset(2, l3_out_offset_3d[2]);
      cdte_ctx_out.trigger_and_wait();
    } else {
      int32_t l3_out_offset_4d[4] = {pre_n_offset, pre_ho_offset,
                                     pre_wo_offset, pre_co_offset};

      sdte_ctx_out.set_src_addr(cur_l1_out_buf);
      sdte_ctx_out.set_dst_offset(0, l3_out_offset_4d[0]);
      sdte_ctx_out.set_dst_offset(1, l3_out_offset_4d[1]);
      sdte_ctx_out.set_dst_offset(2, l3_out_offset_4d[2]);
      sdte_ctx_out.set_dst_offset(3, l3_out_offset_4d[3]);
      sdte_ctx_out.trigger_and_wait();
    }
  }

  sdte_ctx_out.destroy();
  cdte_ctx_out.destroy();
}

template __global__ void
image_conv2d_bias_kernel<float>(float* out, float* lhs,
                                float* rhs, float* bias,
                                CONV2D_OP_PARAS conv_args);

template __global__ void
image_conv2d_bias_kernel<tops::half>(tops::half* out, tops::half* lhs,
                                     tops::half* rhs, tops::half* bias,
                                     CONV2D_OP_PARAS conv_args);

// dummy, invalid now
template __global__ void
image_conv2d_bias_kernel<tops::bfloat>(tops::bfloat* out, tops::bfloat* lhs,
                                       tops::bfloat* rhs, tops::bfloat* bias,
                                       CONV2D_OP_PARAS conv_args);
