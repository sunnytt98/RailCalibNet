import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from DatasetLidar import DatasetLidarCameraKittiOdometry
from losses import CombinedLoss
from models.railcalibnet import calib_net
from quaternion_distances import quaternion_distance
from tensorboardX import SummaryWriter
from utils import merge_inputs
import lidar_pro
import argparse
from tqdm import tqdm

def get_config():
    parser = argparse.ArgumentParser(description='Calibration Training Configuration')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Path to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--base_learning_rate', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--max_t', type=float, default=0.5, help='Max translation error')
    parser.add_argument('--max_r', type=float, default=5.0, help='Max rotation error')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_worker', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type')
    parser.add_argument('--weights', type=str, default=None, help='Path to pre-trained weights')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training')
    parser.add_argument('--random_seed', type=int, default=1203, help='Path to pre-trained weights')
    parser.add_argument('--input_shape', default=[540, 960], help='Path to pre-trained weights')
    parser.add_argument('--log_frequency', type=int, default=100, help='Path to pre-trained weights')
    return parser

def main():
    opts = get_config().parse_args()
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    dataset_class = DatasetLidarCameraKittiOdometry
    dataset_train = dataset_class(max_r=opts.max_r, max_t=opts.max_t, split='train')
    dataset_val = dataset_class(max_r=opts.max_r, max_t=opts.max_t, split='val')
    model_savepath = os.path.join(opts.checkpoints,  'models')
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)
    log_savepath = os.path.join(opts.checkpoints, 'log')
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    train_writer = SummaryWriter(os.path.join(log_savepath, 'train'))
    val_writer = SummaryWriter(os.path.join(log_savepath, 'val'))

    train_dataset_size = len(dataset_train)
    val_dataset_size = len(dataset_val)
    print('Number of the train dataset: {}'.format(train_dataset_size))
    print('Number of the val dataset: {}'.format(val_dataset_size))

    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=opts.batch_size,
                                                 num_workers=opts.num_worker,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)

    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                               shuffle=False,
                                               batch_size=opts.batch_size,
                                               num_workers=opts.num_worker,
                                               collate_fn=merge_inputs,
                                               drop_last=False,
                                               pin_memory=True)

    loss_fn = CombinedLoss()
    model = calib_net()
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=opts.base_learning_rate, weight_decay=5e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.5)
    train_iter = 0
    val_iter = 0
    BEST_VAL_LOSS = 10000
    old_save_filename = None
    start_epoch = 1

    if opts.resume is not None:
        if not os.path.isfile(opts.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {opts.resume}")
        print(f"Loading checkpoint from: {opts.resume}")
        checkpoint = torch.load(opts.resume, map_location='cpu', weights_only=False)
        saved_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(saved_state_dict, strict=True)

        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        BEST_VAL_LOSS = checkpoint.get('best_val_loss', BEST_VAL_LOSS)
        train_iter = checkpoint.get('train_iter', train_iter)
        val_iter = checkpoint.get('val_iter', val_iter)
        print(f"Resume training from epoch {start_epoch} / {opts.epochs}")
        print(f"Best validation loss loaded: {BEST_VAL_LOSS:.6f}")

    if start_epoch > opts.epochs:
        print(f"Checkpoint epoch ({start_epoch - 1}) already reached/exceeded --epochs ({opts.epochs}). Nothing to train.")
        return

    for epoch in range(start_epoch, opts.epochs + 1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = opts.base_learning_rate * math.exp((1 - epoch) * 4e-2)
        model.train()
        rot_loss = 0.0
        trans_loss = 0.0
        combine_loss = 0.0
        for batch_idx, sample in enumerate(tqdm(TrainImgLoader, desc=f"Training Epoch {epoch}/{opts.epochs}", unit="batch")):
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()
            # sample['rgb'] = sample['rgb'].cuda()
            lidar_input = []
            rgb_input = []
            pc_rotated_list = []
            point_gt_list = []
            for idx in range(len(sample['point_gt'])):
                rgb_img = sample['rgb'][idx]
                c, hh, ww = rgb_img.shape
                pc_rotated_ = sample['pc_rotated'][idx].permute(1, 0)[:,:3].cuda().contiguous()
                pc_rotated_list.append(sample['pc_rotated'][idx].cuda())
                point_gt_list.append(sample['point_gt'][idx].cuda())
                camera_matrix = sample['KK'][idx].cuda().float()
                depth_img = lidar_pro.lidar_pro(pc_rotated_, camera_matrix, 90.0, 2, ww, hh)
                depth_img = F.interpolate(depth_img.unsqueeze(0).unsqueeze(0), size=opts.input_shape, mode="bilinear")
                rgb_img =  F.interpolate(rgb_img.unsqueeze(0), size=opts.input_shape, mode="bilinear")
                rgb_img = rgb_img.cuda()
                lidar_input.append(depth_img)
                rgb_input.append(rgb_img)

            lidar_input = torch.stack(lidar_input, dim=0).squeeze(1)
            rgb_input = torch.stack(rgb_input, dim=0).squeeze(1)
            optimizer.zero_grad()
            transl_err, rot_err = model(rgb_input, lidar_input)
            losses = loss_fn(sample['tr_error'], sample['rot_error'], transl_err, rot_err)
            losses['total_loss'].backward()
            optimizer.step()

            rot_loss += losses['rot_loss'].item()
            trans_loss += losses['transl_loss'].item()
            combine_loss += losses['total_loss'].item()
            train_iter = train_iter+1
            if train_iter % opts.log_frequency == 0:
                train_writer.add_scalar("Loss_Total", combine_loss/opts.log_frequency, train_iter)
                train_writer.add_scalar("Loss_Translation", trans_loss/opts.log_frequency, train_iter)
                train_writer.add_scalar("Loss_Rotation", rot_loss/opts.log_frequency, train_iter)
                rot_loss = 0.0
                trans_loss = 0.0
                combine_loss = 0.0

        
        model.eval()
        val_rot_loss = 0.0
        val_trans_loss = 0.0
        val_combine_loss = 0.0
        total_val_loss = 0.0
        total_val_t = 0.
        total_val_r = 0.
        for batch_idx, sample in enumerate(tqdm(ValImgLoader, desc=f"Val Epoch {epoch}/{opts.epochs}", unit="batch")):
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()    
            lidar_input = []
            rgb_input = []
            pc_rotated_list = []
            point_gt_list = []
            for idx in range(len(sample['point_gt'])):
                rgb_img = sample['rgb'][idx].cuda()
                c, hh, ww = rgb_img.shape
                pc_rotated_ = sample['pc_rotated'][idx].permute(1, 0)[:,:3].cuda().contiguous()

                pc_rotated_list.append(sample['pc_rotated'][idx].cuda())
                point_gt_list.append(sample['point_gt'][idx].cuda())
                camera_matrix = sample['KK'][idx].cuda().float()
                # 80000.0, 0, hh, ww)
                depth_img = lidar_pro.lidar_pro(pc_rotated_, camera_matrix, 90.0, 2, ww, hh)
                depth_img = F.interpolate(depth_img.unsqueeze(0).unsqueeze(0), size=opts.input_shape, mode="bilinear")
                rgb_img =  F.interpolate(rgb_img.unsqueeze(0), size=opts.input_shape, mode="bilinear")
                lidar_input.append(depth_img)
                rgb_input.append(rgb_img)

            lidar_input = torch.stack(lidar_input, dim=0).squeeze(1)
            rgb_input = torch.stack(rgb_input, dim=0).squeeze(1)
            with torch.no_grad():
                transl_err, rot_err = model(rgb_input, lidar_input)
            losses = loss_fn(sample['tr_error'], sample['rot_error'], transl_err, rot_err)
            total_trasl_error = torch.tensor(0.0).cuda()
            total_rot_error = quaternion_distance(sample['rot_error'], rot_err, sample['rot_error'].device)
            total_rot_error = total_rot_error * 180. / math.pi

            for j in range(rgb_input.shape[0]):
                total_trasl_error += torch.norm(sample['tr_error'][j] - transl_err[j]) * 100.

            total_val_t += total_trasl_error.item()
            total_val_r += total_rot_error.sum().item()
            
            val_iter +=1
            val_rot_loss += losses['rot_loss'].item()
            val_trans_loss += losses['transl_loss'].item()
            val_combine_loss += losses['total_loss'].item()

            if (batch_idx+1) % opts.log_frequency == 0:
                val_writer.add_scalar("Loss_Total", val_combine_loss/opts.log_frequency, val_iter)
                val_writer.add_scalar("Loss_Translation", val_trans_loss/opts.log_frequency, val_iter)
                val_writer.add_scalar("Loss_Rotation", val_rot_loss/opts.log_frequency, val_iter)
                val_rot_loss = 0.0
                val_trans_loss = 0.0
                val_combine_loss = 0.0
            total_val_loss += losses['total_loss'].item() * len(sample['rgb'])

        traslation_error = total_val_t / len(dataset_val)
        roation_error = total_val_r / len(dataset_val)
        val_writer.add_scalar("roation_error", roation_error, epoch)
        val_writer.add_scalar("traslation_error", traslation_error, epoch)

        val_loss = total_val_loss / len(dataset_val)
        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            savefilename = f'{model_savepath}/checkpoint_r{opts.max_r:.2f}_t{opts.max_t:.2f}_e{epoch}_{val_loss:.3f}.tar'
            torch.save({
            'config': vars(opts),
            'epoch': epoch,
            'state_dict': model.state_dict(), # single gpu
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': BEST_VAL_LOSS,
            'train_iter': train_iter,
            'val_iter': val_iter}, 
            savefilename)
            if old_save_filename is not None:
                if os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
            old_save_filename = savefilename
if __name__=='__main__':
    main() 
