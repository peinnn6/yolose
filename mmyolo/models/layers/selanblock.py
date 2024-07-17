# Yuyu Zhang
# yuyuzhang@tongji.edu.cn
from typing import Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmyolo.registry import MODELS
from mmdet.utils import OptConfigType
from mmcv.cnn import ConvModule
from .yolo_bricks import RepVGGBlock

from ..utils import autopad


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class SPPELAN(nn.Module):
    # spp-elan
    """
    Args:
        in_channel (int): The input channels of this Module.
        out_channel (int): The output channels of this Module.
        kernel_sizes (list(int, tuple[int])): Sequential of kernel sizes in SELAN.
            
        in_expand_ratio (float): Channel expand ratio for inputs of SELAN. Defaults to 3.
        mid_expand_ratio (float): Channel expand ratio for each branch in SELAN. Defaults to 2.
        layers_num (int): Number of layer in SELAN. Defaults to 3.
        in_down_ratio (float): Channel down ratio for downsample conv layer in SELAN. Defaults to 1.
        
        attention_cfg (:obj:`ConfigDict` or dict, optional): Config dict for attention in SELAN. Defaults to None.
        
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to dict(type='SiLU', inplace=True).
    """
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 mid_channel: int,
                 kernel_size: Union[int, Sequence[int]],
                 conv_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None) -> None: 
        super().__init__()
        self.c = mid_channel
        self.cv1 = ConvModule(in_channel,
                              mid_channel, 
                              1, 
                              conv_cfg=conv_cfg,
                              act_cfg=act_cfg,
                              norm_cfg=norm_cfg)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = ConvModule(4*mid_channel,
                              out_channel, 
                              1, 
                              conv_cfg=conv_cfg,
                              act_cfg=act_cfg,
                              norm_cfg=norm_cfg)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))
        
class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self,
                 in_channel: int,
                 out_channel: int, 
                 shortcut=True, 
                 padding=1,
                 groups=1, 
                 kernels=(3, 3), 
                 expansion=0.5,
                 ):  
        super().__init__()
        hidden_channel = int(out_channel * expansion)  
        self.cv1 = RepVGGBlock(in_channel, hidden_channel, kernels[0], 1) 
        self.cv2 = ConvModule(hidden_channel, out_channel, kernels[1], 1, 1, groups=groups)
        self.add = shortcut and in_channel == out_channel


    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
       

class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, 
                 in_channel: int, 
                 out_channel: int,
                 number=1, 
                 shortcut=True, 
                 groups=1, 
                 expansion=0.5):  
        super().__init__()
        hidden_channel = int(out_channel * expansion)  
        self.cv1 = ConvModule(in_channel, hidden_channel, 1, 1)
        self.cv2 = ConvModule(in_channel, hidden_channel, 1, 1)
        self.cv3 = ConvModule(2 * hidden_channel, out_channel, 1)  
        self.m = nn.Sequential(*(RepNBottleneck(hidden_channel, hidden_channel, shortcut, groups, expansion=1.0) for _ in range(number)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))



class RepNCSPELAN4(nn.Module):
    # csp-elan
    """
    Args:
        in_channel (int): The input channels of this Module.
        out_channel (int): The output channels of this Module.
        kernel_sizes (list(int, tuple[int])): Sequential of kernel sizes in SELAN.
            
        in_expand_ratio (float): Channel expand ratio for inputs of SELAN. Defaults to 3.
        mid_expand_ratio (float): Channel expand ratio for each branch in SELAN. Defaults to 2.
        layers_num (int): Number of layer in SELAN. Defaults to 3.
        in_down_ratio (float): Channel down ratio for downsample conv layer in SELAN. Defaults to 1.
        
        attention_cfg (:obj:`ConfigDict` or dict, optional): Config dict for attention in SELAN. Defaults to None.
        
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to dict(type='SiLU', inplace=True).
    """
    def __init__(self, 
                 in_channel: int, 
                 out_channel: int, 
                 mid_channel: int,  
                 mid_out_channel: int, 
                 kernel_size: Union[int, Sequence[int]], 
                 expansion=1,
                 shortcut=True,
                 conv_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None) -> None:  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = mid_channel//2
        self.cv1 = ConvModule(in_channel,
                              mid_channel, 
                              1, 
                              conv_cfg=conv_cfg,
                              act_cfg=act_cfg,
                              norm_cfg=norm_cfg)
        self.cv2 = nn.Sequential(RepNCSP(mid_channel//2,
                                         mid_out_channel, 
                                         expansion), 
                                 ConvModule(mid_out_channel, 
                                            mid_out_channel, 
                                            3, 
                                            padding=1, 
                                            conv_cfg=conv_cfg,
                                            act_cfg=act_cfg,
                                            norm_cfg=norm_cfg))
        self.cv3 = nn.Sequential(RepNCSP(mid_out_channel, 
                                         mid_out_channel, 
                                         expansion), 
                                 ConvModule(mid_out_channel,
                                            mid_out_channel, 
                                            3, 
                                            padding=1, 
                                            conv_cfg=conv_cfg,
                                            act_cfg=act_cfg,
                                            norm_cfg=norm_cfg))
        self.cv4 = ConvModule(mid_channel+(2*mid_out_channel),
                              out_channel, 
                              1,
                              conv_cfg=conv_cfg,
                              act_cfg=act_cfg,
                              norm_cfg=norm_cfg)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))



class SELANBlock(nn.Module):
    """SELANBlock

    Args:
        in_channel (int): The input channels of this Module.
        out_channel (int): The output channels of this Module.
        kernel_sizes (list(int, tuple[int])): Sequential of kernel sizes in SELAN.
            
        in_expand_ratio (float): Channel expand ratio for inputs of SELAN. Defaults to 3.
        mid_expand_ratio (float): Channel expand ratio for each branch in SELAN. Defaults to 2.
        layers_num (int): Number of layer in SELAN. Defaults to 3.
        in_down_ratio (float): Channel down ratio for downsample conv layer in SELAN. Defaults to 1.
        
        attention_cfg (:obj:`ConfigDict` or dict, optional): Config dict for attention in SELAN. Defaults to None.
        
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to dict(type='SiLU', inplace=True).
    """
    def __init__(self, 
                 in_channel: int,
                 out_channel: int,
                 kernel_sizes: Sequence[Union[int, Sequence[int]]] = [1, (3, 3), (3, 3)],
                 
                 in_expand_ratio: float = 3.,
                 mid_expand_ratio: float = 2.,
                 layers_num: int = 3,
                 in_down_ratio: float = 1.,
                 
                 attention_cfg: OptConfigType = None,
                 conv_cfg: OptConfigType = None, 
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 ) -> None:
        super().__init__()
                
        self.in_channel = int(in_channel*in_expand_ratio)//in_down_ratio 
        self.mid_channel = self.in_channel//len(kernel_sizes) 
        self.mid_expand_ratio = mid_expand_ratio
        groups = int(self.mid_channel*self.mid_expand_ratio) 
        self.layers_num = layers_num 
        self.in_attention = None
            
        self.attention = None
        if attention_cfg is not None:
            attention_cfg["dim"] = out_channel
            self.attention = MODELS.build(attention_cfg)

        
        self.in_conv = ConvModule(in_channel,
                                  self.in_channel,
                                  1,
                                  conv_cfg=conv_cfg,
                                  act_cfg=act_cfg,
                                  norm_cfg=norm_cfg)
        
        self.mid_convs = []
        for kernel_size in kernel_sizes: 
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())                   
                continue
            mid_convs = [RepNCSPELAN4(self.mid_channel, 
                                        groups//2, 
                                        out_channel, 
                                        out_channel, 
                                        kernel_size=kernel_size, 
                                        conv_cfg=conv_cfg,
                                        act_cfg=act_cfg,
                                        norm_cfg=norm_cfg) for _ in range(int(self.layers_num))]
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = ConvModule(self.in_channel,
                                   out_channel,
                                   1,
                                   conv_cfg=conv_cfg,
                                   act_cfg=act_cfg,
                                   norm_cfg=norm_cfg)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        out = self.in_conv(x)
        channels = []
        for i,mid_conv in enumerate(self.mid_convs):
            channel = out[:,i*self.mid_channel:(i+1)*self.mid_channel,...]
            if i >= 1:
                channel = channel + channels[i-1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        if self.attention is not None:
            out = self.attention(out)  
        return out
