#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// Hard-coded maximum. Increase if needed.
#define MAX_COL_BLOCKS 1000
#define STRIDE 4
#define N_OFFSETS 72 // if you use more than 72 offsets you will have to adjust this value
#define N_STRIPS (N_OFFSETS - 1)
#define PROP_SIZE (N_OFFSETS + 3)  // start, end, len, 72 offsets
#define DATASET_OFFSET 0

#define DIVUP(m,n) (((m)+(n)-1) / (n))
int64_t const threadsPerBlock = sizeof(unsigned long long) * 8;

// The functions below originates from Fast R-CNN
// See https://github.com/rbgirshick/py-faster-rcnn
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License
// Written by Shaoqing Ren

template <typename scalar_t>
// __device__ inline scalar_t devIoU(scalar_t const * const a, scalar_t const * const b) {
__device__ inline bool devIoU(scalar_t const * const a, scalar_t const * const b, const float threshold) {
  const int start_a = (int) (a[0] * N_STRIPS - DATASET_OFFSET + 0.5); // 0.5 rounding trick
  const int start_b = (int) (b[0] * N_STRIPS - DATASET_OFFSET + 0.5);
  const int start = max(start_a, start_b);
  const int end_a = start_a + a[2] - 1 + 0.5 - ((a[4] - 1) < 0); //  - (x<0) trick to adjust for negative numbers (in case length is 0)
  const int end_b = start_b + b[2] - 1 + 0.5 - ((b[4] - 1) < 0);
  const int end = min(min(end_a, end_b), N_OFFSETS - 1);
  // if (end < start) return 1e9;
  if (end < start) return false;
  scalar_t dist = 0;
  for(unsigned char i = 3 + start; i <= 3 + end; ++i) {
    if (a[i] < b[i]) {
      dist += b[i] - a[i];
    } else {
      dist += a[i] - b[i];
    }
  }
  // return (dist / (end - start + 1)) < threshold;
  return dist < (threshold * (end - start + 1));
  // return dist / (end - start + 1);
}

template <typename scalar_t>
__global__ void nms_kernel(const int64_t n_boxes, const scalar_t nms_overlap_thresh,
                           const scalar_t *dev_boxes, const int64_t *idx, int64_t *dev_mask) {
  const int64_t row_start = blockIdx.y;
  const int64_t col_start = blockIdx.x;

  if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ scalar_t block_boxes[threadsPerBlock * PROP_SIZE];
  if (threadIdx.x < col_size) {
    for (int i = 0; i <  PROP_SIZE; ++i) {
      block_boxes[threadIdx.x * PROP_SIZE + i] = dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * PROP_SIZE + i];
    }
  //   block_boxes[threadIdx.x * 4 + 0] =
  //       dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * 4 + 0];
  //   block_boxes[threadIdx.x * 4 + 1] =
  //       dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * 4 + 1];
  //   block_boxes[threadIdx.x * 4 + 2] =
  //       dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * 4 + 2];
  //   block_boxes[threadIdx.x * 4 + 3] =
  //       dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const scalar_t *cur_box = dev_boxes + idx[cur_box_idx] * PROP_SIZE;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * PROP_SIZE, nms_overlap_thresh)) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}


__global__ void nms_collect(const int64_t boxes_num, const int64_t col_blocks, int64_t top_k, const int64_t *idx, const int64_t *mask, int64_t *keep, int64_t *parent_object_index, int64_t *num_to_keep) {
  int64_t remv[MAX_COL_BLOCKS];
  int64_t num_to_keep_ = 0;

  for (int i = 0; i < col_blocks; i++) {
      remv[i] = 0;
  }

  for (int i = 0; i < boxes_num; ++i) {
      parent_object_index[i] = 0;
  }

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;


    if (!(remv[nblock] & (1ULL << inblock))) {
      int64_t idxi = idx[i];
      keep[num_to_keep_] = idxi;
      const int64_t *p = &mask[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
      for (int j = i; j < boxes_num; j++) {
        int nblockj = j / threadsPerBlock;
        int inblockj = j % threadsPerBlock;
        if (p[nblockj] & (1ULL << inblockj))
            parent_object_index[idx[j]] = num_to_keep_+1;
      }
      parent_object_index[idx[i]] = num_to_keep_+1;

      num_to_keep_++;

      if (num_to_keep_==top_k)
          break;
    }
  }

  // Initialize the rest of the keep array to avoid uninitialized values.
  for (int i = num_to_keep_; i < boxes_num; ++i)
      keep[i] = 0;

  *num_to_keep = min(top_k,num_to_keep_);
}

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::vector<at::Tensor> nms_cuda_forward(
        at::Tensor boxes,
        at::Tensor idx,
        float nms_overlap_thresh,
        unsigned long top_k) {

  const auto boxes_num = boxes.size(0);
  TORCH_CHECK(boxes.size(1) == PROP_SIZE, "Wrong number of offsets. Please adjust `PROP_SIZE`");

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  AT_ASSERTM (col_blocks < MAX_COL_BLOCKS, "The number of column blocks must be less than MAX_COL_BLOCKS. Increase the MAX_COL_BLOCKS constant if needed.");

  auto longOptions = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong);
  auto mask = at::empty({boxes_num * col_blocks}, longOptions);

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);

  CHECK_CONTIGUOUS(boxes);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(mask);

  AT_DISPATCH_FLOATING_TYPES(boxes.type(), "nms_cuda_forward", ([&] {
    nms_kernel<<<blocks, threads>>>(boxes_num,
                                    (scalar_t)nms_overlap_thresh,
                                    boxes.data<scalar_t>(),
                                    idx.data<int64_t>(),
                                    mask.data<int64_t>());
  }));

  auto keep = at::empty({boxes_num}, longOptions);
  auto parent_object_index = at::empty({boxes_num}, longOptions);
  auto num_to_keep = at::empty({}, longOptions);

  nms_collect<<<1, 1>>>(boxes_num, col_blocks, top_k,
                        idx.data<int64_t>(),
                        mask.data<int64_t>(),
                        keep.data<int64_t>(),
                        parent_object_index.data<int64_t>(),
                        num_to_keep.data<int64_t>());


  return {keep,num_to_keep,parent_object_index};
}
