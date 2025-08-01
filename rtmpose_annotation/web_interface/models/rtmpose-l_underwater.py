import numpy as np
import matplotlib.pyplot as plt
from mmpose.evaluation import BaseMetric
from mmpose.registry import METRICS


# _base_ = ['../_base_/default_runtime.py']

from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *

dataset_info = dict(
    dataset_name='underwater_pose',
    keypoint_info={
        0: dict(name='finger_tip', id=0, color=[255, 0, 0], type='upper', swap=''),
        1: dict(name='wrist', id=1, color=[255, 85, 0], type='upper', swap=''),
        2: dict(name='elbow', id=2, color=[255, 170, 0], type='upper', swap=''),
        3: dict(name='head', id=3, color=[255, 255, 0], type='upper', swap=''),
        4: dict(name='shoulder', id=4, color=[170, 255, 0], type='upper', swap=''),
        5: dict(name='pelvis', id=5, color=[85, 255, 0], type='lower', swap=''),
        6: dict(name='hip', id=6, color=[0, 255, 0], type='lower', swap=''),
        7: dict(name='knee', id=7, color=[0, 255, 85], type='lower', swap=''),
        8: dict(name='ankle', id=8, color=[0, 255, 170], type='lower', swap=''),
        9: dict(name='toe', id=9, color=[0, 255, 255], type='lower', swap=''),
    },
    skeleton_info = {
        0: dict(link=('finger_tip', 'wrist'), id=0, color=[255, 0, 0]),
        1: dict(link=('wrist', 'elbow'), id=1, color=[0, 255, 0]),
        2: dict(link=('elbow', 'head'), id=2, color=[0, 0, 255]),
        3: dict(link=('head', 'shoulder'), id=3, color=[255, 255, 0]),
        4: dict(link=('shoulder', 'pelvis'), id=4, color=[255, 0, 255]),
        5: dict(link=('pelvis', 'hip'), id=5, color=[0, 255, 255]),
        6: dict(link=('hip', 'knee'), id=6, color=[255, 85, 0]),
        7: dict(link=('knee', 'ankle'), id=7, color=[85, 255, 0]),
        8: dict(link=('ankle', 'toe'), id=8, color=[0, 170, 255])
    },
    skeleton=[
        [0, 1],  # finger_tip to wrist
        [1, 2],  # wrist to elbow
        [2, 3],  # elbow to head
        [3, 4],  # head to shoulder
        [4, 5],  # shoulder to pelvis
        [5, 6],  # pelvis to hip
        [6, 7],  # hip to knee
        [7, 8],  # knee to ankle
        [8, 9]   # ankle to toe
    ],
    pose_kpt_color=[[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
                    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
                    [0, 255, 170], [0, 255, 255]],
    pose_link_color=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                     [255, 0, 255], [0, 255, 255], [255, 85, 0], [85, 255, 0],
                     [0, 170, 255]],    
    num_keypoints=10,
    joint_weights=[1.0] * 10,
    sigmas=[0.20, 0.25, 0.30, 0.25, 0.30, 0.35, 0.35, 0.30, 0.25, 0.20],
    flip_pairs=[],
)

# runtime
max_epochs = 60 # 270
stage2_num_epochs = 30
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='checkpoints/rtmpose-l.pth'
            #checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            #'rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'  # noqa: E501
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=1024,
        out_channels=10,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=False, ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'custom_underwater_dataset'

backend_args = dict(backend='local')
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         f'{data_root}': 's3://openmmlab/datasets/detection/coco/',
#         f'{data_root}': 's3://openmmlab/datasets/detection/coco/'
#     }))

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    # dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    # dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline,
        metainfo=dataset_info
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        # bbox_file='data/coco/person_detection_results/'
        # 'COCO_val2017_detections_AP_H_56_person.json',
        pipeline=val_pipeline,
        metainfo = dataset_info
    ))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dataset_info
    )
)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/AP', rule='greater', max_keep_ckpts=1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/val.json',
    metric='keypoints',
    format_only=False,
    # Add these new parameters:
    additional_metrics=['per_joint_accuracy'],
    per_joint_accuracy_thresholds=[0.2, 0.5],  # PCK@0.2 and PCK@0.5
    score_mode='keypoint',
    keypoint_score_thr=0.2,
    nms_mode='none',
    use_area=False,
    gt_num_keypoints=10,  # Must match your number of joints
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/test.json',
    metric='keypoints',
    format_only=False,
    additional_metrics=['per_joint_accuracy'],
    per_joint_accuracy_thresholds=[0.2, 0.5],
    score_mode='keypoint',
    keypoint_score_thr=0.2,
    nms_mode='none',
    use_area=False,
    gt_num_keypoints=10,
)

