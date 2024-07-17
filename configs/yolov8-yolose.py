# Yuyu Zhang
# yuyuzhang@tongji.edu.cn
# Reference to
# https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py
_base_ = 'mmyolo::yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'

# ========================Frequently modified parameters======================
# -----data related-----
data_root = 'data/coco/'  # Root path of data
# Path of train annotation file
train_ann_file = 'annotations/instances_train2017.json'
train_data_prefix = 'train2017/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/instances_val2017.json'
val_data_prefix = 'val2017/'  # Prefix of val image path

num_classes = 80  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 32
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# ========================Possible modified parameters========================
# -----data related-----
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# -----model related-----

# Number of layer in SELAN
layers_num = 3
# The scaling factor that controls the depth of the network structure
deepen_factor = 1 / 3
# The scaling factor that controls the width of the network structure
widen_factor = 0.15

# Channel expand ratio for inputs of SELAN
in_expand_ratio = 3
# Channel expand ratio for each branch in SELAN
mid_expand_ratio = 2
# Channel down ratio for downsample conv layer in SELAN
in_down_ratio = 2

# The output channel of the last stage
last_stage_out_channels = 1280

# Kernel sizes of SELAN in PAFPN
kernel_sizes = [1, (3, 3), (3, 3)]

# Normalization config
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
# Activation config
act_cfg = dict(type='SiLU', inplace=True)

# ffeb_cfg
ffeb_cfg = dict(
                ffeb_use=True, 
                num_repeats=[4, 4, 4, 4], 
                channels_list=[6, 12, 24, 48, 96, 192], 
                extra_cfg=dict(
                    norm_cfg=dict(type='SyncBN', requires_grad=True),
                    depths=2,
                    fusion_in=336,   
                    fusion_act=dict(type='ReLU6'),
                    fuse_block_num=3,
                    embed_dim_p=32, 
                    embed_dim_n=336, 
                    key_dim=8,       
                    num_heads=4,     
                    mlp_ratios=1,    
                    attn_ratios=2,   
                    c2t_stride=2,    
                    drop_path_rate=0.1, 
                    trans_channels=[48, 96, 96, 192],
                    pool_mode='torch'
                )
)
# bfeb_cfg
bfeb_cfg = dict(
                bfeb_use=True,  
                num_repeats=[4, 4, 4, 4],
                channels_list=[6, 12, 24, 48, 96, 192], 
                extra_cfg=dict(
                    norm_cfg=dict(type='SyncBN', requires_grad=True),
                    depths=2,
                    fusion_in=336,  
                    fusion_act=dict(type='ReLU6'),
                    fuse_block_num=3,
                    embed_dim_p=32, 
                    embed_dim_n=336, 
                    key_dim=8,      
                    num_heads=4,     
                    mlp_ratios=1,    
                    attn_ratios=2,   
                    c2t_stride=2,    
                    drop_path_rate=0.1, 
                    trans_channels=[48, 96, 96, 192],
                    pool_mode='torch'
                )
)
# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(_delete_=True,
                  type='YOLOv8MS',
                  arch='C3-K3579',
                  last_stage_out_channels=last_stage_out_channels,
                  deepen_factor=deepen_factor,
                  widen_factor=widen_factor,
                  norm_cfg=norm_cfg,
                  in_expand_ratio=in_expand_ratio,
                  mid_expand_ratio=mid_expand_ratio,
                  layers_num=layers_num,
                  act_cfg=act_cfg,
    ),
    neck=dict(_delete_=True,
              type='YOLOv8YOLOSEPAFPN',
              deepen_factor=deepen_factor,
              widen_factor=widen_factor,
              in_channels=[320, 640, last_stage_out_channels],
              out_channels=[320, 640, last_stage_out_channels],
              in_expand_ratio=in_expand_ratio,
              in_down_ratio=in_down_ratio,
              mid_expand_ratio=mid_expand_ratio,
              kernel_sizes=kernel_sizes,
              layers_num=layers_num,
              norm_cfg=norm_cfg,
              act_cfg=act_cfg,
              ffeb_cfg=ffeb_cfg, 
              bfeb_cfg=bfeb_cfg,
    ),
    bbox_head=dict(
        head_module=dict(widen_factor=widen_factor,
                         num_classes=num_classes,
                         in_channels=[320, 640, last_stage_out_channels])),
    train_cfg=dict(assigner=dict(num_classes=num_classes)),
)

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        num_workers=train_num_workers,
                        persistent_workers=persistent_workers,
                        pin_memory=True,
                        sampler=dict(_delete_=True,
                                     type='DefaultSampler',
                                     shuffle=True),
                        collate_fn=dict(_delete_=True, type='yolov5_collate'),
                        dataset=dict(data_root=data_root,
                                     ann_file=train_ann_file,
                                     data_prefix=dict(img=train_data_prefix)))

val_dataloader = dict(batch_size=val_batch_size_per_gpu,
                      num_workers=val_num_workers,
                      persistent_workers=persistent_workers,
                      pin_memory=True,
                      drop_last=False,
                      sampler=dict(_delete_=True,
                                   type='DefaultSampler',
                                   shuffle=False),
                      dataset=dict(data_root=data_root,
                                   test_mode=True,
                                   data_prefix=dict(img=val_data_prefix),
                                   ann_file=val_ann_file))

test_dataloader = val_dataloader
