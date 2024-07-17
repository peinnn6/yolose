# Yuyu Zhang
# yuyuzhang@tongji.edu.cn
from typing import Sequence, Union, List


import torch
import torch.nn as nn

from mmyolo.registry import MODELS
from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN
from mmyolo.models.utils import make_divisible
from mmdet.utils import ConfigType
from mmcv.cnn import ConvModule

from .fbfeb_yolo_neck import FBFEBYOLONeck
from ..layers import SELANBlock


@MODELS.register_module()
class YOLOv8YOLOSEPAFPN(FBFEBYOLONeck):
    """Path Aggregation Network in YOLOv8 with MS-Block.

    Args:
        in_expand_ratio (float): Channel expand ratio for inputs of SELAN. Defaults to 3.
        in_down_ratio (float): Channel down ratio for downsample conv layer in SELAN. Defaults to 1.
        mid_expand_ratio (float): Channel expand ratio for each branch in SELAN. Defaults to 2.
        layers_num (int): Number of layer in SELAN. Defaults to 3.
        kernel_sizes (list(int, tuple[int])): Sequential or number of kernel sizes in SELAN. Defaults to [1,3,3].
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[int, List[int]],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 upsample_feats_cat_first: bool = True,
                 freeze_all: bool = False,
                 in_expand_ratio: float = 1,
                 in_down_ratio: float = 1,
                 mid_expand_ratio: float = 2,
                 layers_num: int = 3,
                 kernel_sizes: Sequence[Union[int, Sequence[int]]] = [1,3,3],
                 ffeb_cfg: ConfigType = None, 
                 bfeb_cfg: ConfigType = None,
                 norm_cfg: ConfigType = None,
                 act_cfg: ConfigType = None,
                 **kwargs):
        self.in_expand_ratio = in_expand_ratio
        self.in_down_ratio = in_down_ratio
        self.mid_expand_ratio = mid_expand_ratio
        self.kernel_sizes = kernel_sizes
        self.ffeb_cfg = ffeb_cfg
        self.bfeb_cfg = bfeb_cfg
        self.layers_num = layers_num
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            upsample_feats_cat_first=upsample_feats_cat_first,
            freeze_all=freeze_all,
            ffeb_cfg=ffeb_cfg,  
            bfeb_cfg=bfeb_cfg, 
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=None,
            **kwargs
            )
    
    def init_weights(self):
        if self.init_cfg is None:
            """Initialize the parameters."""
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.reset_parameters()
        else:
            super().init_weights()

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()
    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(scale_factor=2, mode='nearest')

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return SELANBlock(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            in_expand_ratio=self.in_expand_ratio,
            in_down_ratio=self.in_down_ratio,
            mid_expand_ratio=self.mid_expand_ratio,
            kernel_sizes=self.kernel_sizes,
            layers_num=int(self.layers_num * self.deepen_factor),
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
    
    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return ConvModule(
            make_divisible(self.in_channels[idx], self.widen_factor),
            make_divisible(self.in_channels[idx], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return SELANBlock(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            in_expand_ratio=self.in_expand_ratio,
            in_down_ratio=self.in_down_ratio,
            mid_expand_ratio=self.mid_expand_ratio,
            kernel_sizes=self.kernel_sizes,
            layers_num=int(self.layers_num * self.deepen_factor),
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()