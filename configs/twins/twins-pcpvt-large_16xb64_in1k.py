_base_ = ['twins-pcpvt-base_8xb128_in1k.py']

# model settings
model = dict(backbone=dict(arch='large'), head=dict(in_channels=512))

data_root="DiseaseID-1"

# dataset settings
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',
        data_prefix='/train',
        with_label=True,   # or False for unsupervised tasks        
    )
)

val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_prefix=data_root + '/valid',
        with_label=True,   # or False for unsupervised tasks        
    )
)

test_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_prefix=data_root + '/test',
        with_label=True,   # or False for unsupervised tasks        
    )
)


