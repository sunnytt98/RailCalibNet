import random
import cv2
from tqdm import tqdm
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from models.railcalibnet import calib_net
from DatasetLidar import DatasetLidarCameraKittiOdometry

from quaternion_distances import quaternion_distance
from utils import mat2xyzrpy, merge_inputs, quat2mat, quaternion_from_matrix, tvector2mat

import lidar_pro
import argparse



plt.rcParams['axes.unicode_minus'] = False
font_EN = {'weight': 'normal', 'size': 16}
font_CN = { 'weight': 'normal', 'size': 16}
plt_size = 10.5

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_config():
    parser = argparse.ArgumentParser(description='Calibration Training Configuration')
    parser.add_argument('--checkpoints', type=str, default='/data/p2_calib/p2_calib/ckpt/checkpoint.tar')
    parser.add_argument('--max_t', type=float, default=0.5, help='Max translation error')
    parser.add_argument('--max_r', type=float, default=5.0, help='Max rotation error')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_worker', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--random_seed', type=int, default=1203, help='Path to pre-trained weights')
    parser.add_argument('--input_shape', default=[540, 960], help='Path to pre-trained weights')
    parser.add_argument('--log_frequency', type=int, default=100, help='Path to pre-trained weights')
    return parser
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    opts = get_config().parse_args()
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    dataset_class = DatasetLidarCameraKittiOdometry
    dataset_test = dataset_class(max_r=opts.max_r, max_t=opts.max_t, split='test')
    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                shuffle=False,
                                                batch_size=opts.batch_size,
                                                num_workers=opts.num_worker,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)
    
    model = calib_net()
    checkpoint = torch.load(opts.checkpoints, weights_only=False)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict, strict=True)
    model = model.cuda()
    model.eval()
    errors_r = []
    errors_t = []
    errors_t2 = []
    errors_rpy = []
    while torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()
            lidar_input = []
            rgb_input = []
            for idx in range(len(sample['point_gt'])):
                rgb_img = sample['rgb'][idx].cuda()
                c, hh, ww = rgb_img.shape
                R = quat2mat(sample['rot_error'][idx])
                T = tvector2mat(sample['tr_error'][idx])
                R = torch.round(R * 1e4) / 1e4 
                T = torch.round(T * 1e4) / 1e4  
                RT_inv = torch.mm(T, R)
                RT = RT_inv.clone().inverse()
                pc_rotated_ = sample['pc_rotated'][idx].permute(1, 0)[:,:3].cuda().contiguous()
                camera_matrix = sample['KK'][idx].cuda().float()
                depth_img = lidar_pro.lidar_pro(pc_rotated_, camera_matrix, 90.0, 2, ww, hh)
                depth_img = F.interpolate(depth_img.unsqueeze(0).unsqueeze(0), size=opts.input_shape, mode="bilinear")
                rgb_img =  F.interpolate(rgb_img.unsqueeze(0), size=opts.input_shape, mode="bilinear")
                lidar_input.append(depth_img)
                rgb_input.append(rgb_img)
            
            lidar_input = torch.stack(lidar_input, dim=0).squeeze(1)
            rgb_input = torch.stack(rgb_input, dim=0).squeeze(1)
            T_predicted, R_predicted = model(rgb_input, lidar_input)
            R_predicted = quat2mat(R_predicted[0])
            T_predicted = tvector2mat(T_predicted[0])
            RT_predicted = torch.mm(T_predicted, R_predicted)
            RTs = torch.mm(RT, RT_predicted)
            T_composed = RTs[:3, 3].detach()
            R_composed = quaternion_from_matrix(RTs.detach())
            errors_t.append(T_composed.norm().item())
            errors_r.append(quaternion_distance(R_composed.unsqueeze(0), torch.tensor([1., 0., 0., 0.], 
                                                device=R_composed.device).unsqueeze(0), R_composed.device))
            rpy_error = mat2xyzrpy(RTs)[3:]
            rpy_error *= (180.0 / 3.141592)
            errors_rpy.append(rpy_error)
            errors_t2.append(T_composed)
    errors_r = torch.tensor(errors_r).abs().cpu() * (180.0 / 3.141592)
    errors_t = torch.tensor(errors_t).abs().cpu()* 100
    errors_rpy = torch.stack(errors_rpy, dim=0).abs().cpu()
    errors_t2 = torch.stack(errors_t2, dim=0).abs().cpu()* 100
    errors_t = errors_t2[:,0:3].view(-1, 1)
    errors_r = errors_rpy[:,0:3].view(-1, 1)
    print(f"Translation mean/median/std:{errors_t.mean():.4f}/{errors_t.median():.4f}/{errors_t.std():.4f} cm")
    print(f"Translation X mean/median/std:{errors_t2[:,0].mean():.4f}/{errors_t2[:,0].median():.4f}/{errors_t2[:,0].std():.4f} cm")
    print(f"Translation Y mean/median/std:{errors_t2[:,1].mean():.4f}/{errors_t2[:,1].median():.4f}/{errors_t2[:,1].std():.4f} cm")
    print(f"Translation Z mean/median/std:{errors_t2[:,2].mean():.4f}/{errors_t2[:,2].median():.4f}/{errors_t2[:,2].std():.4f} cm")

    print(f"Rotation mean/amedian/std:{errors_r.mean():.4f}/{errors_r.median():.4f}/{errors_r.std():.4f} deg")
    print(f"Rotation Roll mean/median/std:{errors_rpy[:,0].mean():.4f}/{errors_rpy[:,0].median():.4f}/{errors_rpy[:,0].std():.4f} deg")
    print(f"Rotation Pitch mean/median/std:{errors_rpy[:,1].mean():.4f}/{errors_rpy[:,1].median():.4f}/{errors_rpy[:,1].std():.4f} deg")
    print(f"Rotation Yaw mean/median/std:{errors_rpy[:,2].mean():.4f}/{errors_rpy[:,2].median():.4f}/{errors_rpy[:,2].std():.4f} deg")
    # rotation rpy



    


if __name__=='__main__':
    main() 

