_base_ = [
    '../_base_/models/efficientnet_v2/efficientnetv2_s.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

data_root="caries"

# model setting
model = dict(head=dict(num_classes=3))

# dataset settings
dataset_type = 'ImageNet21k'
data_preprocessor = dict(
    num_classes=3,
    # RGB format normalization parameters
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline,
        data_root=data_root,))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline,
        data_root=data_root,))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline,
        data_root=data_root,))

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=4e-3),
    clip_grad=dict(max_norm=5.0),
)
