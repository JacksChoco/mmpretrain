_base_ = [
    'efficientnetv2-s_8xb32_in1k-384px.py',
]

data_root="caries"

# model setting
model = dict(backbone=dict(arch='xl'), )

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=384, crop_padding=0),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=512, crop_padding=0),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
        data_root=data_root + "/train"
    ))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline,
                                   data_root=data_root + '/valid'
                                   ))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline,
                                    data_root=data_root + '/test',
                                    ))
