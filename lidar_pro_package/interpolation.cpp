#include "utils.h"


torch::Tensor trilinear_interpolation_fw(
    const torch::Tensor& feats,
    const torch::Tensor& points
){
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    return trilinear_fw_cu(feats, points);
}


torch::Tensor trilinear_interpolation_bw(
    const torch::Tensor& dL_dfeat_interp,
    const torch::Tensor& feats,
    const torch::Tensor& points
){
    CHECK_INPUT(dL_dfeat_interp);
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    return trilinear_bw_cu(dL_dfeat_interp, feats, points);
}




torch::Tensor points2img(
    const torch::Tensor& points,
    const torch::Tensor& camera_in,
    const float depth,
    const int radius1,
    const int ww,
    const int hh
){
    CHECK_INPUT(points);
    CHECK_INPUT(camera_in);
    return ptoi_cu(points, camera_in, depth, radius1, ww, hh);
}

torch::Tensor points2img2(
    const torch::Tensor& points,
    const torch::Tensor& camera_in,
    const float depth,
    const int radius1,
    const int ww,
    const int hh
){
    CHECK_INPUT(points);
    CHECK_INPUT(camera_in);
    return ptoi_cu2(points, camera_in, depth, radius1, ww, hh);
}

torch::Tensor points2rgb(
    const torch::Tensor& points,
    const torch::Tensor& rgb_img,
    const torch::Tensor& camera_in,
    const float depth,
    const int radius1,
    const int ww,
    const int hh
){
    CHECK_INPUT(points);
    CHECK_INPUT(camera_in);
    return points2rgb_cu(points, rgb_img, camera_in, depth, radius1, ww, hh);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    namespace py = pybind11;

    m.def("trilinear_interpolation_fw", &trilinear_interpolation_fw);
    m.def("trilinear_interpolation_bw", &trilinear_interpolation_bw);

    m.def(
        "points2img",
        &points2img,
        py::arg("points"),
        py::arg("camera_in"),
        py::arg("depth") = 80.0f,
        py::arg("radius1") = 1,
        py::arg("ww") = 1241,
        py::arg("hh") = 376
    );
    m.def(
        "points2img2",
        &points2img2,
        py::arg("points"),
        py::arg("camera_in"),
        py::arg("depth") = 80.0f,
        py::arg("radius1") = 1,
        py::arg("ww") = 1241,
        py::arg("hh") = 376
    );
    m.def(
        "points2rgb",
        &points2rgb,
        py::arg("points"),
        py::arg("rgb_img"),
        py::arg("camera_in"),
        py::arg("depth") = 80.0f,
        py::arg("radius1") = 1,
        py::arg("ww") = 1241,
        py::arg("hh") = 376
    );
}
