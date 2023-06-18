_base_ = [
    'efficientnetv2-s_8xb32_in1k-384px.py',
]

data_root="caries"

# model setting
model = dict(backbone=dict(arch='xl'), )

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=512),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
        data_root=data_root,
    ))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline,
                                   data_root=data_root,

                                   ))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline,
                                    data_root=data_root,

                                    ))
