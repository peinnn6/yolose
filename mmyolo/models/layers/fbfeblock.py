# Yuyu Zhang
# yuyuzhang@tongji.edu.cnimport torch

import torch
from mmcv.cnn import build_norm_layer
from torch import nn
import numpy as np
from torch import nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from mmyolo.registry import MODELS
from mmcv.cnn import ConvModule
from ..utils import autopad
from ..layers import SELANBlock

class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert kernel_size == 3
        assert padding == 1
        
        padding_11 = padding - kernel_size // 2
        
        self.nonlinearity = nn.ReLU()
        
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        
        else:
            self.rbr_identity = nn.BatchNorm2d(
                    num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
    
    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()
        
        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class Conv(nn.Module):
    '''Normal Conv with SiLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        self.add_module('c', nn.Conv2d(
                a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    
    def forward(self, x):
        return self.relu(x + 3) / 6

def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
                self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
    
    def forward(self, x):  
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1) 
        
        xx = torch.matmul(attn, vv)
        
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class top_Block(nn.Module):
    
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer,
                              norm_cfg=norm_cfg)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)
    
    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class TopBasicLayer(nn.Module):  
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=nn.ReLU6):
        super().__init__()
        self.block_num = block_num
        
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                    embedding_dim, key_dim=key_dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                    drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg, act_layer=act_layer))
    
    def forward(self, x):
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x

def onnx_AdaptiveMaxPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    maxpool = nn.MaxPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = maxpool(x)
    return x


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride, pool_mode='onnx'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
    
    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        
        output_size = np.array([H, W])
        
        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d
        
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        
        out = [self.pool(inp, output_size) for inp in inputs]
        
        return torch.cat(out, dim=1)


def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool

def get_max_pool():
    if torch.onnx.is_in_onnx_export():
        max_pool = onnx_AdaptiveMaxPool2d
    else:
        max_pool = nn.functional.adaptive_max_pool2d
    return max_pool


class InjectionMultiSum_Auto_pool(nn.Module): # Inject
    def __init__(
            self,
            inp: int,  
            oup: int, 
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
            global_inp=None,
    ) -> None:
        super().__init__()
        self.norm_cfg = norm_cfg
        
        if not global_inp:
            global_inp = inp
        
        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None) 
        self.global_embedding = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None) 
        self.global_act = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None) 
        self.act = h_sigmoid()
    
    def forward(self, x_l, x_g): 
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape 
        use_pool = H < g_H 
        
        local_feat = self.local_embedding(x_l) 
        
        global_act = self.global_act(x_g) 
        global_feat = self.global_embedding(x_g) 
        
        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])
            
            sig_act = avg_pool(global_act, output_size) 
            global_feat = avg_pool(global_feat, output_size) 
        
        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False) 
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False) 
        
        out = local_feat * sig_act + global_feat 
        return out 


class SimConv(nn.Module):
    '''Normal Conv with ReLU VAN_activation''' 
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
class AdvPoolFusion(nn.Module):   
    def forward(self, x1, x2):
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d
        
        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)
        
        return torch.cat([x1, x2], 1)
    
class MaxPoolFusion(nn.Module):
    def forward(self, x1, x2):
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveMaxPool2d
        else:
            self.pool = nn.functional.adaptive_max_pool2d

        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)

        return torch.cat([x1, x2], 1)


class SimFusion_3in(nn.Module):  
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        self.cv1 = SimConv(in_channel_list[0], out_channels, 1, 1) 
        self.cv_fuse = SimConv(out_channels * 4, out_channels, 1, 1) 
        self.downsample = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)
        
        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])
        
        x0 = self.downsample(x[0], output_size)
        x1 = self.cv1(x[1])
        x2 = F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False)
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))


class SimFusion_4in(nn.Module):  
   
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d  
    
    def forward(self, x):
        x_l, x_m, x_s = x  
        B, C, H, W = x_m.shape  
        output_size = np.array([H, W])
        
        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d
        
        x_l = self.avg_pool(x_l, output_size)  
        x_s = F.interpolate(x_s, size=(H, W), mode='bilinear', align_corners=False) 
        
        out = torch.cat([x_l, x_m, x_s], 1)  
        return out


@MODELS.register_module()
class FFEBlock(nn.Module):    
    def __init__(
            self,
            ffeb_use=False,  
            channels_list=None,
            num_repeats=None,
            block=RepVGGBlock,  
            extra_cfg=None):
        super().__init__()


    
        assert channels_list is not None
        assert num_repeats is not None
        self.ffeb_use = ffeb_use
        self.low_FAM = SimFusion_4in() 
        self.low_IFM = nn.Sequential(
                Conv(extra_cfg.fusion_in, extra_cfg.embed_dim_p, kernel_size=1, stride=1, padding=0), 
                *[block(extra_cfg.embed_dim_p, extra_cfg.embed_dim_p,) for _ in range(extra_cfg.fuse_block_num)],
                Conv(extra_cfg.embed_dim_p, sum(extra_cfg.trans_channels[0:2]), kernel_size=1, stride=1, padding=0), 
        )
    
        self.reduce_layer_c5 = SimConv(
                in_channels=channels_list[5],  
                out_channels=channels_list[4],  
                kernel_size=1,
                stride=1
        )
        
        self.LAF_p4 = SimFusion_3in(
                in_channel_list=[channels_list[4], channels_list[3]],  
                out_channels=channels_list[3],  
        )
       
        self.Inject_p4 = InjectionMultiSum_Auto_pool(channels_list[3], channels_list[3], norm_cfg=extra_cfg.norm_cfg,activations=nn.ReLU6) 

        self.Rep_p4 = RepBlock(
                in_channels=channels_list[3],  
                out_channels=channels_list[4],  
                n=num_repeats[0],
                block=block
        )
        
        self.reduce_layer_p4 = SimConv(
                in_channels=channels_list[4],  
                out_channels=channels_list[3],  
                kernel_size=1,
                stride=1
        )
        
        self.LAF_p3 = AdvPoolFusion()
        self.Inject_p3 = InjectionMultiSum_Auto_pool(channels_list[4], channels_list[3], norm_cfg=extra_cfg.norm_cfg,
                                                     activations=nn.ReLU6) 
        self.Rep_p3 = RepBlock(
                in_channels=channels_list[3],  
                out_channels=channels_list[3],  
                n=num_repeats[1],
                block=block
        )

        self.trans_channels = extra_cfg.trans_channels
    
    def forward(self, input):
        if self.ffeb_use:
            (c3, c4, c5) = input 
            low_align_feat = self.low_FAM(input) 
            low_fuse_feat = self.low_IFM(low_align_feat) 
            low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1)

            ## inject low-level global info to p4
            c5_half = self.reduce_layer_c5(c5)  
            p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])  
            p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
            p4 = self.Rep_p4(p4)
            
            ## inject low-level global info to p3
            p4_half = self.reduce_layer_p4(p4) 
            p3_adjacent_info = self.LAF_p3(p4_half,c3) 
            p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])  
            p3 = self.Rep_p3(p3) 

            
            outputs = [p3, p4, c5] 
            return outputs


@MODELS.register_module()
class BFEBlock(nn.Module):    
    def __init__(
            self,
            bfeb_use=False, 
            channels_list=None,
            num_repeats=None,
            block=RepVGGBlock, 
            extra_cfg=None):
        super().__init__()


        
        assert channels_list is not None
        assert num_repeats is not None
        self.bfeb_use = bfeb_use
        

        self.high_FAM = PyramidPoolAgg(stride=extra_cfg.c2t_stride, pool_mode=extra_cfg.pool_mode)
        dpr = [x.item() for x in torch.linspace(0, extra_cfg.drop_path_rate, extra_cfg.depths)]
        self.high_IFM = TopBasicLayer(
                block_num=extra_cfg.depths,
                embedding_dim=extra_cfg.embed_dim_n,
                key_dim=extra_cfg.key_dim,
                num_heads=extra_cfg.num_heads,
                mlp_ratio=extra_cfg.mlp_ratios,
                attn_ratio=extra_cfg.attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
                norm_cfg=extra_cfg.norm_cfg
        )
        self.conv_1x1_n = nn.Conv2d(extra_cfg.embed_dim_n, sum(extra_cfg.trans_channels[2:4]), 1, 1, 0)
        
        self.reduce_layer_p4 = SimConv(
                in_channels=channels_list[4],  
                out_channels=channels_list[3],  
                kernel_size=1,
                stride=1
        )
        self.LAF_n4 = AdvPoolFusion()
        self.Inject_n4 = InjectionMultiSum_Auto_pool(channels_list[4], channels_list[4],
                                                     norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6) 
        self.Rep_n4 = RepBlock(
                in_channels=channels_list[3] + channels_list[3],  
                out_channels=channels_list[4], 
                n=num_repeats[2],
                block=block
        )
        
        self.reduce_layer_c5 = SimConv(
                in_channels=channels_list[5], 
                out_channels=channels_list[4],  
                kernel_size=1,
                stride=1
        )
        self.LAF_n5 = MaxPoolFusion()
        self.Inject_n5 = InjectionMultiSum_Auto_pool(channels_list[5], channels_list[5], 
                                                     norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6) 
        self.Rep_n5 = RepBlock(
                in_channels=channels_list[4] + channels_list[4],
                out_channels=channels_list[5], 
                n=num_repeats[3],
                block=block
        )
        self.trans_channels = extra_cfg.trans_channels
    
    def forward(self, input):
        

        if self.bfeb_use:   
            (p3, p4, p5) = input 
            high_align_feat = self.high_FAM([p3, p4, p5])
            high_fuse_feat = self.high_IFM(high_align_feat)
            high_fuse_feat = self.conv_1x1_n(high_fuse_feat) 
            high_global_info = high_fuse_feat.split(self.trans_channels[2:4], dim=1) 
            
            ## inject low-level global info to n4
            p4_half = self.reduce_layer_p4(p4) 
            n4_adjacent_info = self.LAF_n4(p3, p4_half) 
            n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0]) 
            n4 = self.Rep_n4(n4)
            
            ## inject low-level global info to n5
            p5_half = self.reduce_layer_c5(p5) 
            n5_adjacent_info = self.LAF_n5(n4, p5_half) 
            n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1]) 
            n5 = self.Rep_n5(n5) 
            
            outputs = [p3, n4, n5]
            return outputs
        
        return input

