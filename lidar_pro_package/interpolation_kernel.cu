#include <torch/extension.h>


template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> feats,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> feat_interp
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n>=feats.size(0) || f>=feats.size(2)) return;

    // point -1~1
    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;
    
    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;
    feat_interp[n][f] = (1-u)*(a*feats[n][0][f] +
                               b*feats[n][1][f] +
                               c*feats[n][2][f] +
                               d*feats[n][3][f]) + 
                            u*(a*feats[n][4][f] +
                               b*feats[n][5][f] +
                               c*feats[n][6][f] +
                               d*feats[n][7][f]);
}


torch::Tensor trilinear_fw_cu(
    const torch::Tensor& feats,
    const torch::Tensor& points
){
    const int N = feats.size(0), F = feats.size(2);
    
    torch::Tensor feat_interp = torch::empty({N, F}, feats.options());

    const dim3 threads(16, 16);
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.scalar_type(), "trilinear_fw_cu", 
    ([&] {
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            points.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            feat_interp.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    return feat_interp;
}






template <typename scalar_t>
__global__ void trilinear_bw_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> dL_dfeat_interp,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> feats,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> dL_dfeats
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n>=feats.size(0) || f>=feats.size(2)) return;

    // point -1~1
    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;
    
    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;

    dL_dfeats[n][0][f] = (1-u)*a*dL_dfeat_interp[n][f];
    dL_dfeats[n][1][f] = (1-u)*b*dL_dfeat_interp[n][f];
    dL_dfeats[n][2][f] = (1-u)*c*dL_dfeat_interp[n][f];
    dL_dfeats[n][3][f] = (1-u)*d*dL_dfeat_interp[n][f];
    dL_dfeats[n][4][f] = u*a*dL_dfeat_interp[n][f];
    dL_dfeats[n][5][f] = u*b*dL_dfeat_interp[n][f];
    dL_dfeats[n][6][f] = u*c*dL_dfeat_interp[n][f];
    dL_dfeats[n][7][f] = u*d*dL_dfeat_interp[n][f];
}


torch::Tensor trilinear_bw_cu(
    const torch::Tensor& dL_dfeat_interp,
    const torch::Tensor& feats,
    const torch::Tensor& points
){
    const int N = feats.size(0), F = feats.size(2);
    
    torch::Tensor dL_dfeats = torch::empty({N, 8, F}, feats.options());

    const dim3 threads(16, 16);
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.scalar_type(), "trilinear_bw_cu", 
    ([&] {
        trilinear_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dfeat_interp.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            feats.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            points.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            dL_dfeats.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()
        );
    }));

    return dL_dfeats;
}



template <typename scalar_t>
__global__ void ptoi_cu_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> camera_in,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> depth_img,
    const int pointNums,
    const float depth,
    const int radius1,
    const int ww,
    const int hh
){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x; 
    if (id>pointNums-1)
        return;
    // if (n>=feats.size(0) || f>=feats.size(2)) return;

    // point -1~1
    const scalar_t x = points[id][0];
    const scalar_t y = points[id][1];
    const scalar_t z = points[id][2];

    const scalar_t x_ = x * camera_in[0][0] + z * camera_in[0][2];
    const scalar_t y_ = y * camera_in[1][1]  +  z * camera_in[1][2];
    const scalar_t z_ = z + 0.0001;

    int u = int(x_/z_);
    int v = int(y_/z_);
    int radius = 3;
    int width = ww;
    int hight = hh;

    for (int aa = u-radius; aa<=u+radius; aa++)
    {
      for(int bb = v-radius;bb<=v+radius; bb++)
      {
        if (aa>=0 && aa<width && bb>=0 && bb<hight && z<=depth && z>0.0)
        {
            if (aa>=u-radius1 && aa<=u+radius1 && bb>=v-radius1 && bb<=v+radius1
            && x<=3.0 && x>=-3.0 && y>=-3.0 && y<=3.0)
            {
               
                depth_img[bb][aa]=z/80.0;
            }
            else
            {
                depth_img[bb][aa]=depth_img[bb][aa];
            }
        }

      }
    }
}


torch::Tensor ptoi_cu(
    const torch::Tensor& points,
    const torch::Tensor& camera_in,
    const float depth,
    const int radius1,
    const int ww,
    const int hh
){
    const int pointNums = points.size(0);
    const int width = ww, height = hh;

    torch::Tensor depth_img = torch::zeros({height, width}, points.options());
    
    const dim3 threads(16);
    const dim3 blocks(int(pointNums/16)+1);

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "ptoi_cu", 
    ([&] {
        ptoi_cu_kernel<scalar_t><<<blocks, threads>>>(
            points.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            camera_in.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            depth_img.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            pointNums,
            depth,
            radius1,
            ww,
            hh
        );
    }));

    return depth_img;
}





template <typename scalar_t>
__global__ void ptoi_cu_kernel2(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> camera_in,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> depth_img,
    const int pointNums,
    const float depth,
    const int radius1,
    const int ww,
    const int hh
){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x; 
    if (id>pointNums-1)
        return;
    // if (n>=feats.size(0) || f>=feats.size(2)) return;

    // point -1~1
    const scalar_t x = points[id][0];
    const scalar_t y = points[id][1];
    const scalar_t z = points[id][2];

    const scalar_t x_ = x * camera_in[0][0] + z * camera_in[0][2];
    const scalar_t y_ = y * camera_in[1][1]  +  z * camera_in[1][2];
    const scalar_t z_ = z + 0.0001;

    int u = int(x_/z_);
    int v = int(y_/z_);
    int radius = 3;
    int width = ww;
    int hight = hh;

    for (int aa = u-radius; aa<=u+radius; aa++)
    {
      for(int bb = v-radius;bb<=v+radius; bb++)
      {
        if (aa>=0 && aa<width && bb>=0 && bb<hight && z<=depth && z>0.0)
        {
            if (aa>=u-radius1 && aa<=u+radius1 && bb>=v-radius1 && bb<=v+radius1)
            {
                depth_img[bb][aa]=z/80.0;
            }
            else
            {
                depth_img[bb][aa]=depth_img[bb][aa];
            }
        }

      }
    }
}


torch::Tensor ptoi_cu2(
    const torch::Tensor& points,
    const torch::Tensor& camera_in,
    const float depth,
    const int radius1,
    const int ww,
    const int hh
){
    const int pointNums = points.size(0);
    const int width = ww, height = hh;

    torch::Tensor depth_img = torch::zeros({height, width}, points.options());
    
    const dim3 threads(16);
    const dim3 blocks(int(pointNums/16)+1);

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "ptoi_cu2", 
    ([&] {
        ptoi_cu_kernel2<scalar_t><<<blocks, threads>>>(
            points.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            camera_in.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            depth_img.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            pointNums,
            depth,
            radius1,
            ww,
            hh
        );
    }));

    return depth_img;
}









__device__ float customFmod(float a, float b) {
    return a - b * floorf(a / b);
}

template <typename scalar_t>
__global__ void p2rgb_cu_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> camera_in,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> new_rgb_img,
    const int pointNums,
    const float depth,
    const int radius1,
    const int ww,
    const int hh
){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x; 
    if (id>pointNums-1)
        return;
    // if (n>=feats.size(0) || f>=feats.size(2)) return;

    // point -1~1
    const scalar_t x = points[id][0];
    const scalar_t y = points[id][1];
    const scalar_t z = points[id][2];

    const scalar_t x_ = x * camera_in[0][0] + z * camera_in[0][2];
    const scalar_t y_ = y * camera_in[1][1]  +  z * camera_in[1][2];
    const scalar_t z_ = z + 0.0001;

    int u = int(x_/z_);
    int v = int(y_/z_);
    int radius = 3;
    int width = ww;
    int hight = hh;

    for (int aa = u-radius; aa<=u+radius; aa++)
    {
      for(int bb = v-radius;bb<=v+radius; bb++)
      {
        if (aa>=0 && aa<width && bb>=0 && bb<hight && z<=128.0&& z>0.0)
        {
            if (aa>=u-radius1 && aa<=u+radius1 && bb>=v-radius1 && bb<=v+radius1)
            {
                float h = (z /128.0) * 360.0;
                float s = 1.0; // 饱和度
                float v = 1.0; // 明度
                float c = v * s;    

                float mod_result = customFmod(h / 60.0f, 2.0f);
                float x = c * (1 - fabsf(mod_result - 1));

                // float x = c * (1 - std::fabs(fmod(h / 60.0, 2) - 1));
                float m = v - c;
                float r = 0, g = 0, b = 0;
                if (h >= 0 && h < 60) {
                    r = c, g = x, b = 0;
                } else if (h >= 60 && h < 120) {
                    r = x, g = c, b = 0;
                } else if (h >= 120 && h < 180) {
                    r = 0, g = c, b = x;
                } else if (h >= 180 && h < 240) {
                    r = 0, g = x, b = c;
                } else if (h >= 240 && h < 300) {
                    r = x, g = 0, b = c;
                } else if (h >= 300 && h < 360) {
                    r = c, g = 0, b = x;
                }

                new_rgb_img[bb][aa][0]=r*255;
                new_rgb_img[bb][aa][1]=g*255;
                new_rgb_img[bb][aa][2]=b*255;
            }
            else
            {
                int bbb = 1;
            }
        }

      }
    }
}





torch::Tensor points2rgb_cu(
    const torch::Tensor& points,
    const torch::Tensor& rgb_img,
    const torch::Tensor& camera_in,
    const float depth,
    const int radius1,
    const int ww,
    const int hh
){
    const int pointNums = points.size(0);
    const int width = ww, height = hh;

    torch::Tensor new_rgb_img = torch::zeros({height, width, 3}, points.options());
    new_rgb_img.copy_(rgb_img);
    
    const dim3 threads(16);
    const dim3 blocks(int(pointNums/16)+1);

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "points2rgb_cu", 
    ([&] {
        p2rgb_cu_kernel<scalar_t><<<blocks, threads>>>(
            points.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            camera_in.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            new_rgb_img.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            pointNums,
            depth,
            radius1,
            ww,
            hh
        );
    }));

    return new_rgb_img;
}
