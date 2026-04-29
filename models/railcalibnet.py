import torch
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class pre_head(nn.Module):
    def __init__(self):
        super(pre_head,self).__init__()
        self.fc0 = nn.Linear(261120,256*8) 
        self.fc1 = nn.Linear(256*8,256*4)  # 96*10
        self.fc2 = nn.Linear(256*8,256*4)  # 96*10
        self.fc_tr = nn.Linear(256*4, 3) 
        self.fc_rot = nn.Linear(256*4, 4) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        nn.init.xavier_normal_(self.fc0.weight,0.1)
        nn.init.xavier_normal_(self.fc1.weight,0.1)
        nn.init.xavier_normal_(self.fc2.weight,0.1)
        nn.init.xavier_normal_(self.fc_tr.weight,0.1)
        nn.init.xavier_normal_(self.fc_rot.weight,0.1)

    def forward(self,x:torch.Tensor):
        x = x.reshape(x.shape[0],-1) #torch.Size([16, 163840])
        x = self.fc0(x)
        x_tr = self.fc1(x)
        x_tr = self.fc_tr(x_tr)
        x_rot = self.fc2(x)
        x_rot = self.fc_rot(x_rot)
        x_rot = F.normalize(x_rot, dim=1)
        return x_tr,x_rot


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=2,
                 qkv_bias=False,
                 attn_drop_ratio=0.0,
                 proj_drop_ratio=0.0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # torch.Size([16, 2040, 512])
        x = self.fc1(x) # torch.Size([16, 2040, 512])
        #torch.Size([16, 2040, 512])
        # x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
           # embed_dim=1024 num_heads=16 mlp_ratio=4.0 qkv_bias=true qk_scale=None
            # drop_ratio=0.0 attn_drop_ratio=0.0 drop_path_ratio=0.0
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class sf_block(nn.Module):
    def __init__(self, input_dim=128, num_heads=4, cross = True):
        super(sf_block, self).__init__()
        self.cross = cross
        self.input_dim = 128
        self.num_heads = num_heads
        self.scale = (input_dim/self.num_heads) ** -0.5
        self.norm_diff = nn.LayerNorm(input_dim)
        self.q_depth = nn.Linear(input_dim, input_dim, bias=False)
        self.q_rgb = nn.Linear(input_dim, input_dim, bias=False)
        self.kv_depth = nn.Linear(input_dim, input_dim*2, bias=False)
        self.kv_rgb = nn.Linear(input_dim, input_dim*2, bias=False)
        self.proj_depth = nn.Linear(input_dim, input_dim)
        self.proj_rgb = nn.Linear(input_dim, input_dim)
        self.mlp_depth = Mlp(in_features=input_dim, hidden_features=input_dim*4, act_layer=nn.GELU, drop=0.0)
        self.mlp_rgb = Mlp(in_features=input_dim, hidden_features=input_dim*4, act_layer=nn.GELU, drop=0.0)
        self.norm_rgb = nn.LayerNorm(input_dim)
        self.norm_depth = nn.LayerNorm(input_dim)

        self.norm_depth1 = nn.LayerNorm(input_dim)
        self.norm_rgb1 = nn.LayerNorm(input_dim)
        self.norm_depth2 = nn.LayerNorm(input_dim)
        self.norm_rgb2 = nn.LayerNorm(input_dim)


    def normalize_embeddings(self, a, eps=1e-8):
        a_n = a.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        return a_norm


    def forward(self, rgb_token, lidar_token):
        diff_r_d =rgb_token - lidar_token
        diff_d_r = lidar_token - rgb_token
        B, N, C = diff_d_r.shape
        q_depth = self.q_depth(diff_d_r).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        q_rgb = self.q_rgb(diff_r_d).reshape(B, N,1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        # q_rgb = q_depth
        kv_depth = self.kv_depth(lidar_token).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_depth, v_depth = kv_depth[0], kv_depth[1]

        kv_rgb = self.kv_rgb(rgb_token).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_rgb, v_rgb = kv_rgb[0], kv_rgb[1]


        attn_rgb = (q_rgb @ k_rgb.transpose(-2, -1)) * self.scale
        attn_rgb = attn_rgb.softmax(dim=-1)
        x_rgb = (attn_rgb @ v_rgb).transpose(1, 2).reshape(B, N, C)

        attn_depth = (q_depth @ k_depth.transpose(-2, -1)) * self.scale
        attn_depth = attn_depth.softmax(dim=-1)
        x_depth = (attn_depth @ v_depth).transpose(1, 2).reshape(B, N, C)
        
        x_rgb = self.proj_rgb(x_rgb)
        x_depth = self.proj_depth(x_depth)

        rgb_token = rgb_token + x_rgb
        lidar_token = lidar_token + x_depth


        rgb_token = self.norm_rgb1(rgb_token)
        lidar_token = self.norm_depth1(lidar_token)

        x_rgb = self.mlp_rgb(rgb_token) #torch.Size([16, 128, 128])
        x_depth = self.mlp_depth(lidar_token) #torch.Size([16, 128, 128])

        rgb_token = rgb_token + x_rgb
        lidar_token = lidar_token + x_depth
      
        return rgb_token, lidar_token
    



class fuse_block(nn.Module):
    def __init__(self, input_channel=256, num_heads=4):
        super(fuse_block, self).__init__()
        self.num_heads = num_heads
        self.scale = (input_channel/self.num_heads) ** -0.5
        self.fuse_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU())
        self.norm_q = nn.LayerNorm(128)
        self.norm_k = nn.LayerNorm(128)
        self.norm_v = nn.LayerNorm(128)
        self.q_l = nn.Linear(128, 128, bias=False)
        self.k_l = nn.Linear(128, 128, bias=False)
        self.v_l = nn.Linear(128, 128, bias=False)

        self.rgb_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU())
        self.depth_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU())
        self.kv_rgb = nn.Linear(128, 128*2, bias=False)
        self.kv_depth = nn.Linear(128, 128*2, bias=False)
        self.proj_depth = nn.Linear(128, 128)
        self.proj_rgb = nn.Linear(128, 128)

        self.mlp_depth = Mlp(in_features=128, hidden_features=128*4, act_layer=nn.GELU, drop=0.0)
        self.mlp_rgb = Mlp(in_features=128, hidden_features=128*4, act_layer=nn.GELU, drop=0.0)

    def forward(self,  f_r, f_d, cat_token):
        q = self.norm_q(f_r)
        k = self.norm_k(f_d)
        v = cat_token
        B, N, C = q.shape
        q = self.q_l(q).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        k = self.k_l(k).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        v = self.v_l(v).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x_rgb = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x_rgb = self.proj_rgb(x_rgb)

        return x_rgb


class SpatialFeatureToTokenWeights(nn.Module):
    def __init__(self):
        super(SpatialFeatureToTokenWeights, self).__init__()
        self.liner1 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU(inplace=True)
        self.liner2 = nn.Linear(64, 1)
        self.sig1 = nn.Sigmoid()
    def forward(self, x):
        B, _,_ = x.shape
        # x = x.reshape(B, -1)
        x = self.liner1(x)
        x = self.relu1(x)
        x = self.liner2(x)
        x = self.sig1(x)  
        return x



class calib_net(nn.Module):
    def __init__(self):
        super(calib_net, self).__init__()
        self.conv1_lidar = nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2)
        self.maxpool_lidar = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        return_layers = {'layer3': 'layer3'}
        self.rgb_backbone = IntermediateLayerGetter(resnet18(pretrained=True),return_layers)
        self.depth_backbone = IntermediateLayerGetter(resnet18(pretrained=True),return_layers)
        self.rgb_conv = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU())
        
        self.depth_conv = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU())
        self.block_fuse4 = sf_block(input_dim=128, num_heads=2)
        self.block_fuse3 = sf_block(input_dim=128, num_heads=2)
        self.block_fuse1 = sf_block(input_dim=128, num_heads=2)
        self.block_fuse2 = sf_block(input_dim=128, num_heads=2)
        self.fuse_head = fuse_block(input_channel=128, num_heads=2)
        self.w4_r = SpatialFeatureToTokenWeights()
        self.w4_d = SpatialFeatureToTokenWeights()
        self.head = pre_head()

    def normalize_embeddings(self, a, eps=1e-8):
        a_n = a.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        return a_norm


    def forward(self, rgb, lidar):
        # step1
        lidar_feaure = self.conv1_lidar(lidar)
        lidar_feaure = self.maxpool_lidar(lidar_feaure)

        lidar_feaure = self.depth_backbone(lidar_feaure)
        rgb_feature = self.rgb_backbone(rgb)

        rgb_feature= rgb_feature['layer3'] 
        depth_feature = lidar_feaure['layer3']
        rgb_feature = self.rgb_conv(rgb_feature)
        depth_feature = self.depth_conv(depth_feature)  
        # step2
        B, _, H, W = rgb_feature.shape
        f_r = rgb_feature.flatten(2).transpose(1, 2)
        f_d = depth_feature.flatten(2).transpose(1, 2) # torch.Size([16, 2040, 128])
        f_r, f_d = self.block_fuse1(f_r, f_d)
        f_r, f_d = self.block_fuse2(f_r, f_d)
        f_r, f_d = self.block_fuse3(f_r, f_d)
        f_r, f_d = self.block_fuse4(f_r, f_d) # torch.Size([16, 2040, 128])
        cat_token = 0.5*(self.w4_r(f_r)*self.normalize_embeddings(f_r) \
                         + self.w4_d(f_d)*self.normalize_embeddings(f_d))
        transl, rot = self.head(cat_token)
        return transl, rot

if __name__=='__main__': 

    model = calib_net()
    rgb_input = torch.rand(16, 3, 540, 960)
    lidar_input = torch.rand(16, 1, 540, 960)
    pre = model(rgb_input, lidar_input)

        


