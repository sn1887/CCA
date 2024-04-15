# %% [code]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

#-----------------------------------------------------------------------------------------------------------------------------------

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

#-----------------------------------------------------------------------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
#-----------------------------------------------------------------------------------------------------------------------------------


class IoUIndex(nn.Module):
    def __init__(self, smooth=1):
        super(IoUIndex, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Intersection is equivalent to True Positive count
        # Union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + self.smooth) / (union + self.smooth)

        return IoU

    
#-----------------------------------------------------------------------------------------------------------------------------------
    
def tversky_coef(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-5):
    """
    Compute the Tversky coefficient between predicted and true binary masks.

    Parameters:
    - y_true (Tensor): Ground truth binary mask tensor (shape: [batch_size, ...]).
    - y_pred (Tensor): Predicted binary mask tensor (shape: [batch_size, ...]).
    - alpha (float): Weight parameter for false positives.
    - beta (float): Weight parameter for false negatives.
    - smooth (float): Smoothing factor to avoid division by zero.

    Returns:
    - tversky (Tensor): Tversky coefficient tensor.
    """
    assert y_true.size() == y_pred.size(), "Input shapes must match."

    # Flatten the tensors
    y_true_flat = y_true.view(y_true.size(0), -1)
    y_pred_flat = y_pred.view(y_pred.size(0), -1)

    # True positives, false positives, and false negatives
    tp = torch.sum(y_true_flat * y_pred_flat, dim=1)
    fp = torch.sum((1 - y_true_flat) * y_pred_flat, dim=1)
    fn = torch.sum(y_true_flat * (1 - y_pred_flat), dim=1)

    # Tversky coefficient
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

    return tversky.mean()
    
    
    
    
