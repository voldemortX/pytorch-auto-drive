#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>

std::vector<at::Tensor> nms_cuda_forward(
        at::Tensor boxes,
        at::Tensor idx,
        float nms_overlap_thresh,
        unsigned long top_k);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> nms_forward(
        at::Tensor boxes,
        at::Tensor scores,
        float thresh,
        unsigned long top_k) {


    auto idx = std::get<1>(scores.sort(0,true));

    CHECK_INPUT(boxes);
    CHECK_INPUT(idx);

    return nms_cuda_forward(boxes, idx, thresh, top_k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &nms_forward, "NMS forward");
}
