import csv
import os
import mathutils
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import open3d as o3d
from utils import invert_pose, rotate_back
from torch.utils.data.dataloader import default_collate


class DatasetLidarCameraKittiOdometry(Dataset):

    def __init__(self, max_t=0.5, max_r=5.0, split='val', 
                 base_path="/data/p2_calib/data/cd_frame",
                 val_RT_file_path = '/data/p2_calib/p2_calib/ckpt'):
        super(DatasetLidarCameraKittiOdometry, self).__init__()
        self.max_r = max_r
        self.max_t = max_t
        self.split = split
        self.train_list_img = []
        self.train_list_lidar = []
        self.train_list_K = []

        self.all_files = []
        self.train_list = []
        self.test_list = []
        self.base_path = base_path
        self.sequence_list = ['20', '33','26','28','32','30','17','35','39','8','31']
        for it in os.listdir(self.base_path):
            if it.endswith(".jpg") and it.split("_")[0] in self.sequence_list:
                self.test_list.append(it)
            elif it.endswith(".jpg"):
                self.train_list.append(it)
        self.K = np.array(([[7346.0909801549, 0., 1904.2700214182],
                            [0., 7346.9262890386, 1061.1803855291],
                            [0., 0., 1.]]), dtype=np.double)
        if split == 'val' or split == 'test':
            self.all_files = self.test_list
        else:
            self.all_files = self.train_list

        self.val_RT = []

        if split == 'val' or split == 'test':
            print(len(self.all_files))
            print("-------------------------------------------------------")
            val_RT_file = os.path.join(val_RT_file_path, f'{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                         float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"
 

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)


    def __getitem__(self, idx):
        item = self.all_files[idx]
        img_path = os.path.join(self.base_path, item)
        pcd_path = os.path.join(self.base_path, item.replace(".jpg", "_calibed.pcd"))
        KK = self.K
        point_cloud_pcd = o3d.io.read_point_cloud(pcd_path)
        point_cloud = np.array(point_cloud_pcd.points)
        pc_org = torch.from_numpy(point_cloud.astype(np.float32))
        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")
        
        pc_in = pc_org
        h_mirror = False
        img = Image.open(img_path)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        R = mathutils.Euler((rotx, roty, rotz))
        T = mathutils.Vector((transl_x, transl_y, transl_z))
        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)
        

        R_ = mathutils.Quaternion(R).to_matrix()
        R_.resize_4x4()
        T_ = mathutils.Matrix.Translation(T)
        RT_ = T_ @ R_
        pc_rotated = pc_in # torch.Size([4, 125588])
        pc_rotated = rotate_back(pc_in, RT_)
        img = self.custom_transform(img, 0.0, h_mirror)
        if self.split == 'test':
            sample = {'rgb': img, 'point_gt': pc_in, 
                      'tr_error': T, 'rot_error': R, 'pc_rotated':pc_rotated, 
                      'KK':KK,'initial_RT': initial_RT, 'name':item}
        else:
            sample = {'rgb': img, 'point_gt': pc_in, 
                      'tr_error': T, 'rot_error': R, 
                      'pc_rotated':pc_rotated, 'KK':KK, 'name':item}

        return sample


def merge_inputs(queries):
    point_clouds = []
    pc_rotateds = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'point_cloud'  and key!= 'pc_rotated' }
    for input in queries:
        point_clouds.append(input['point_cloud'])
        pc_rotateds.append(input['pc_rotated'])
    returns['point_cloud'] = point_clouds
    returns['pc_rotated'] = pc_rotateds
    return returns

if __name__ == "__main__":
    dataset_class = DatasetLidarCameraKittiOdometry
    dataset_val = dataset_class()
    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                               shuffle=False,
                                               batch_size=16,
                                               num_workers=16,
                                               collate_fn=merge_inputs,
                                               drop_last=False,
                                               pin_memory=False)
    for batch_idx, sample in enumerate(ValImgLoader):
        sample['tr_error'] = sample['tr_error'].cuda()
        sample['rot_error'] = sample['rot_error'].cuda()
        sample['rgb'] = sample['rgb'].cuda()
        lidar_input = []
        for idx in range(len(sample['pc_rotated'])):
            pc_rotated_ = sample['pc_rotated'][idx].permute(1, 0).float()[:,:3].cuda()
            camera_matrix = sample['KK'][idx].cuda().float()
        lidar_input = torch.stack(lidar_input).unsqueeze(1)
        print(batch_idx)
