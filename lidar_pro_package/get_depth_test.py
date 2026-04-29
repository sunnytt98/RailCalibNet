import torch
import lidartodepth
import open3d as o3d
import cv2
import lidar_pro

import numpy as np
K = np.array(([[718.856,    0,     607.1928], 
                [0.,     718.856,  185.2157], 
                [0., 0., 1.]]), dtype=np.float32)
camera_matrix = torch.from_numpy(K).float().cuda()

point_cloud_pcd = o3d.io.read_point_cloud("/data/dataset/kitti/data_odometry_velodyne/dataset/sequences/00/velodyne/000647_calib.pcd")
point_cloud = np.array(point_cloud_pcd.points)
point_cloud = torch.from_numpy(point_cloud).cuda().float() #N 3
lidar_input = []

depth_img = lidar_pro.lidar_pro_points2img(point_cloud, camera_matrix)


def get_2D_lidar_projection(pcl, cam_intrinsic):
        pcl = pcl[(pcl[:,2] >= 0) & (pcl[:,2] <= 100)]
        # pcl = pcl[pcl[:,2]>=0 and pcl[:,2]<=100]
        pcl_xyz = cam_intrinsic @ pcl.T
        pcl_xyz = pcl_xyz.T
        pcl_z = pcl_xyz[:, 2]
        pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
        pcl_uv = pcl_xyz[:, :2]

        return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[(pc_rotated[:,2] >= 0) & (pc_rotated[:,2] <= 100)]
    pcl_xyz = cam_calib @ pc_rotated.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.permute(2, 0, 1)

    return depth_img, pcl_uv

# depth_img, pcl_uv= lidar_project_depth(point_cloud, K, (376, 1241))


# for aa in range(1000):
#     for i in range(16):
#         depth_img = torch.zeros((376, 1241), dtype=torch.float).cuda().contiguous()
#         lidartodepth.convert(point_cloud, depth_img, camera_matrix)
#         lidar_input.append(depth_img)
#         print(aa, i)
# depth_img =depth_img[:376, :1241]
depth_img = depth_img*255
depth_img = depth_img.cpu().numpy()
depth_img = np.array(depth_img, dtype=np.uint8)
cv2.imwrite("/data/code/calib/st_calib_kitti/check/kitti_p3.jpg", depth_img)
