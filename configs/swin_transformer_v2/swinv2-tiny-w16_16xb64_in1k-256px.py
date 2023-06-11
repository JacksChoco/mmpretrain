_base_ = [
    '../_base_/models/swin_transformer_v2/tiny_256.py',
    '../_base_/datasets/imagenet_bs64_swin_256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

model = dict(backbone=dict(window_size=[16, 16, 16, 8]))

data_root="caries"

# dataset settings
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root + "/train",
        ann_file='',
        data_prefix='',
        with_label=True,   # or False for unsupervised tasks        
    )
)

val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root + '/valid',
        ann_file='',
        data_prefix='',
        with_label=True,   # or False for unsupervised tasks        
    )
)

test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='CustomDataset',
        ann_file='',
        data_prefix='',
        data_root=data_root + '/test',
        with_label=True,   # or False for unsupervised tasks        
    )
)

val_evaluator = dict(type='Accuracy', topk=(1, 2))

test_evaluator = val_evaluator