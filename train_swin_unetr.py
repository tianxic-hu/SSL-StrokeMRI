# -*- coding: utf-8 -*-
import glob
import os
import pickle
import time

import nibabel as nib
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.data import CacheDataset, Dataset, DataLoader, write_nifti, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
from monai.metrics import compute_hausdorff_distance, compute_dice, DiceMetric, get_confusion_matrix, HausdorffDistanceMetric, compute_average_surface_distance
from monai.networks.layers import Norm
from monai.networks.nets import UNet, SegResNet, DynUNet, SwinUNETR
from monai.metrics import DiceMetric
import monai.transforms as tf
from monai.utils import get_torch_version_tuple, set_determinism
import pandas as pd
import torch
import numpy as np
from monai.utils import first, set_determinism
################################################################################
### DATA SETUP
################################################################################
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# get path to $SLURM_TMPDIR, change to output path if not training on slurm
slurm_tmpdir = os.getenv('SLURM_TMPDIR')

# change path to output directory here
# out_dir = os.path.join(base_dir, 'out')
out_dir = os.path.join(slurm_tmpdir, 'out')
print(f'Files will be saved to: {out_dir}')

# change path to data directory here
data_dir = os.path.join(slurm_tmpdir, 'dataset-ISLES22-multimodal-unzipped')

set_determinism(seed=0)
print('Loading files...')

# training files (remove validation subjects after)
train_dwi = sorted(glob.glob(os.path.join(data_dir, 'rawdata', 'sub-strokecase*', 'ses-0001', 'sub-strokecase*_dwi.nii.gz')))
train_labels = sorted(glob.glob(os.path.join(data_dir, 'derivatives', 'sub-strokecase*', 'ses-0001', 'sub-strokecase*_msk.nii.gz')))

train_files = [
    {'image': train_dwi, 'label': label_name}
    for train_dwi, label_name in zip(train_dwi, train_labels)
]

# validation files
# Note validation_subjs_50.txt is untarred from validation_subjs_50.tar.gz prior to running this script
with open(os.path.join(slurm_tmpdir, "validation_subjs_50.txt")) as file:
        val_subj_list = [line.rstrip() for line in file]

val_files = [train_file for train_file in train_files if train_file['label'].split('/')[-3] in val_subj_list]
train_files = [train_file for train_file in train_files if train_file['label'].split('/')[-3] not in val_subj_list]

# !!! Added for small compilation test. Comment out for actual training.
# train_files = train_files[:8]
# val_files = val_files[:2]

print(f'Total {len(train_files)} subjects for training.')
print(f'Total {len(val_files)} subjects for validation.')

################################################################################
### DEFINE TRANSFORMS
################################################################################

# train transforms
train_transforms = tf.Compose(
    [
        tf.LoadImaged(keys=["image", "label"]),
        tf.EnsureChannelFirstd(keys=["image", "label"]),
        tf.CropForegroundd(keys=["image", "label"], source_key="image"),
        tf.Orientationd(keys=["image", "label"], axcodes="RAS"),
        tf.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        tf.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
        ),
    ]
)
val_transforms = tf.Compose(
    [
        tf.LoadImaged(keys=["image", "label"]),
        tf.EnsureChannelFirstd(keys=["image", "label"]),
        tf.CropForegroundd(keys=["image", "label"], source_key="image"),
        tf.Orientationd(keys=["image", "label"], axcodes="RAS"),
        tf.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ]
)
################################################################################
### DATASET AND DATALOADERS
################################################################################

# train dataset
# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
# train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

# val dataset
# val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

################################################################################
### CHECK TRANSFORMS
################################################################################
check_ds = Dataset(data=val_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image, label = (check_data["image"][0][0], check_data["label"][0][0])
print(f"image shape: {image.shape}, label shape: {label.shape}")
# plot the slice [:, :, 80]
plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 80], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 80])
plt.show()
plt.savefig(os.path.join(out_dir, 'image_check.png'), bbox_inches='tight')


################################################################################
### MODEL AND LOSS
################################################################################

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=2,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True,
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# model.load_state_dict(torch.load(os.path.join(slurm_tmpdir, "best_dice_model_01304.pth")))
# print(f"Loaded pretrained model.")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal model parameters: {total_params}")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable model parameters: {total_trainable_params}\n")

################################################################################
### TRAINING LOOP
################################################################################

max_epochs = 500
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
val_epoch = []
post_pred = tf.Compose([tf.AsDiscrete(argmax=True, to_onehot=2)])
post_label = tf.Compose([tf.AsDiscrete(to_onehot=2)])

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(f"input: {inputs.shape}, output: {outputs.shape}, label: {labels.shape}")
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        val_epoch.append(epoch + 1)
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(out_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

print('Saving training outputs to csv in output dir...')
df = pd.DataFrame(data={'epoch num': val_epoch, "metric": metric_values})
df.to_csv(os.path.join(out_dir, 'val_eval.csv'))
print('Done. Finished saving files.')

################################################################################
### PLOT TRAINING CURVES
################################################################################
# plot loss and validation metric
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].set_title('Epoch Average Loss')
ax[0].set_xlabel('epoch')
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
ax[0].plot(x, y, label='Training loss')
x = [val_interval * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
ax[0].plot(x, y, label='Validation loss')
ax[0].legend()
ax[1].set_title('Val Mean Dice')
ax[1].set_xlabel('epoch')
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
ax[1].plot(x, y)
plt.savefig(os.path.join(out_dir, 'training_curves.png'), bbox_inches='tight')
