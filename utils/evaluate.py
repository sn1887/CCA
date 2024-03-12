# %% [code]
# %% [code]
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import logging


from data_loader import MedicalImageDataset
from losses_metrics import *

@torch.inference_mode()
def evaluate(mode, args, net, dataloader, device, amp, criterion, save_checkpoint=False, dir_checkpoint=None):
    assert mode in ['validation round', 'testing round']
    IoU = IoUIndex()
    net.eval()
    num_val_batches = len(dataloader)
    dice_score, total_iou, total_tversky, total_loss = 0, 0, 0, 0
    best_loss = float('inf')  # Initialize best_loss to positive infinity

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        with tqdm(total=num_val_batches, desc= mode, unit='img', position=0, leave=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10}', colour='blue') as pbar:
            for batch in dataloader:
                image, mask_true = batch[0], batch[1]

                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                # predict the mask
                _, mask_pred = net(image)

                if args.classes == 1:
                    assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    # compute the Dice score
                    batch_dice = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                    dice_score += batch_dice
                    # Calculate and accumulate Tversky coefficient
                    batch_tversky = tversky_coef(mask_pred, mask_true)
                    total_tversky += batch_tversky
                    # Calculate and accumulate IoU coefficient
                    batch_iou = IoU(mask_pred, mask_true)
                    total_iou += batch_iou
                    # Compute loss
                    loss = F.binary_cross_entropy_with_logits(mask_pred, mask_true.float())
                else:
                    assert mask_true.min() >= 0 and mask_true.max() < args.classes, 'True mask indices should be in [0, n_classes['
                    # convert to one-hot format
                    mask_true = F.one_hot(mask_true, args.classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), args.classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                    # Compute loss
                    loss = criterion(mask_pred.squeeze(1), mask_true.squeeze(1).float())

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(
                    **{
                        'Dice': f'{batch_dice:.4f}',
                        'Tversky': f'{batch_tversky:.4f}',
                        'IoU': f'{batch_iou:.4f}',
                        'Loss': f'{loss.item():.4f}',
                    }
                )

            # Compute average loss for the entire validation set
            avg_loss = total_loss / num_val_batches

            # Update best_loss and save checkpoint if a new best loss is found
            if avg_loss < best_loss and True:
                #print('loss imporved from(',best_loss,') to ', avg_loss )
                best_loss = avg_loss
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = net.state_dict()
                state_dict['mask_values'] = [0, 1]
                torch.save(state_dict, str(dir_checkpoint / 'best_checkpoint.pth'))
                logging.info(f'Best checkpoint saved! Loss: {avg_loss:.4f}')
                # Log the best checkpoint and associated metrics to Neptune
                args.run['metrics/best_loss'].log(best_loss)
                args.run['artifacts/best_checkpoint'].upload('/kaggle/working/checkpoints/best_checkpoint.pth')

    print(f'Dice: {(dice_score / max(num_val_batches, 1)):.4f} -Tversky: {(total_tversky / max(num_val_batches, 1)):.4f} -IoU: {(total_iou / max(num_val_batches, 1)):.4f} -Loss: {(avg_loss):.4f}')
    net.train()
    return ( dice_score / max(num_val_batches, 1), total_tversky / max(num_val_batches, 1),  total_iou / max(num_val_batches, 1), avg_loss    )