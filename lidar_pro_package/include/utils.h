#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor trilinear_fw_cu(
    const torch::Tensor& feats,
    const torch::Tensor& points
);


torch::Tensor trilinear_bw_cu(
    const torch::Tensor& dL_dfeat_interp,
    const torch::Tensor& feats,
    const torch::Tensor& points
);


torch::Tensor ptoi_cu(
    const torch::Tensor& points,
    const torch::Tensor& camera_in,
    const float depth,
    const int radius1,
    const int ww,
    const int hh
);

torch::Tensor ptoi_cu2(
    const torch::Tensor& points,
    const torch::Tensor& camera_in,
    const float depth,
    const int radius1,
    const int ww,
    const int hh
);

torch::Tensor points2rgb_cu(
    const torch::Tensor& points,
    const torch::Tensor& rgb_img,
    const torch::Tensor& camera_in,
    const float depth,
    const int radius1,
    const int ww,
    const int hh
);

