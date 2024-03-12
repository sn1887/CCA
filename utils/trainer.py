import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics import JaccardIndex

from data_loader import MedicalImageDataset
from losses_metrics import *
from evaluate import evaluate

dir_img = Path('/kaggle/input/cis-data/data/images')
dir_mask = Path('/kaggle/input/cis-data/data/masks')
dir_checkpoint = Path('/kaggle/working/checkpoints/')


def train_model(
        model,
        args,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):

    train_set = MedicalImageDataset('train',
                  augment=True,
                  noise=True, noise_typ = 'speckle')
    val_set   = MedicalImageDataset('val',
                  augment=False,
                  noise=False, noise_typ = 'speckle')
    n_val = int(len(val_set))
    n_train = int(len(train_set))
    run, npt_logger = args.run, args.npt_logger
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    num_train_batches = len(train_loader)


    optimizer = optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay,betas=(momentum, 0.999), foreach=True)
    IoU = IoUIndex()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if args.classes > 1 else DiceBCELoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, total_dice, total_tversky, total_iou = 0, 0, 0, 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img',position=0, leave = True,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10}', colour='green') as pbar:
            for i, batch in enumerate(train_loader):
                images, true_masks = batch[0], batch[1]

                assert images.shape[1] == args.n_channels, \
                    f'Network has been defined with {args.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    _, masks_pred = model(images)
                    if args.classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.squeeze(1).float())
                        #loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.squeeze(1).float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, args.classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )


                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                torch.cuda.empty_cache()
                
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                #-------------------------------------------------------------------------------------------------
                #-------------------------------------------------------------------------------------------------
                # Calculate and accumulate Metrics
                if args.classes == 1:
                    predicted_masks = (F.sigmoid(masks_pred) > 0.5).float()
                else:
                    predicted_masks = F.softmax(masks_pred, dim=1).argmax(dim=1)

                batch_dice = dice_coeff(predicted_masks, true_masks)
                total_dice += batch_dice
                # Calculate and accumulate Tversky coefficient
                batch_tversky = tversky_coef(predicted_masks, true_masks)
                total_tversky += batch_tversky
                # Calculate and accumulate IoU coefficient
                batch_iou = IoU(predicted_masks, true_masks)
                total_iou += batch_iou
                #-------------------------------------------------------------------------------------------------

                pbar.set_postfix(**{'loss (batch)': loss.item(), f'Epoch {epoch} Dice': f'{batch_dice:.4f}', 'Tversky': f'{batch_tversky:.4f}', 'IoU': f'{batch_iou:.4f}'})

        # Log Metrics to Neptune
        run[npt_logger.base_namespace]["epoch/Training loss"].append(loss.item())
        run[npt_logger.base_namespace]["epoch/Training dice"].append(total_dice/num_train_batches)
        run[npt_logger.base_namespace]["epoch/Training IoU"].append(total_iou/num_train_batches)
        run[npt_logger.base_namespace]["epoch/Training Tversky"].append(total_tversky/num_train_batches)
                
        print(f'\nEpoch {epoch} -Dice: {total_dice/num_train_batches:.4f} -Tversky: {total_tversky/num_train_batches:.4f} -IoU: {total_iou/num_train_batches:.4f}')
        # Evaluation round

        _dice, _tversky, _IoU, _val_loss = evaluate('validation round', args, model, val_loader, device, amp, criterion = criterion,dir_checkpoint =dir_checkpoint)
        scheduler.step(_dice)
        run[npt_logger.base_namespace]["epoch/Validation Dice Score"].append(_dice)
        run[npt_logger.base_namespace]["epoch/Validation Tversky Score"].append(_tversky)
        run[npt_logger.base_namespace]["epoch/Validation IoU Score"].append(_IoU)
        run[npt_logger.base_namespace]["epoch/Validation Loss"].append(_val_loss)