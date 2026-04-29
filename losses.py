from torch import nn as nn
from quaternion_distances import quaternion_distance


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.loss = {}
    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        pose_loss = loss_rot+ loss_transl
        self.loss['total_loss'] = pose_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        return self.loss

