import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss function, used for binary segmentation tasks. It calculates the overlap between 
    the predicted and ground truth masks using the Dice coefficient, with an optional smoothing factor.

    Args:
        weight (Tensor, optional): The weight for the loss. Default is None.
        size_average (bool, optional): If True, the loss is averaged across the batch. Default is True.

    Methods:
        forward(inputs, targets, smooth=1): Computes the Dice loss.
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass for calculating Dice loss.

        Args:
            inputs (Tensor): The predicted mask values (probabilities).
            targets (Tensor): The ground truth mask values (binary).
            smooth (float, optional): A small constant to avoid division by zero. Default is 1.

        Returns:
            Tensor: The Dice loss.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class DiceBCELoss(nn.Module):
    """
    Combination of Dice Loss and Binary Cross-Entropy Loss (BCE) for binary segmentation tasks.
    The Dice loss is used for measuring overlap, while BCE is used for pixel-wise classification.

    Args:
        weight (Tensor, optional): The weight for the loss. Default is None.
        size_average (bool, optional): If True, the loss is averaged across the batch. Default is True.

    Methods:
        forward(inputs, targets, smooth=1): Computes the Dice + BCE loss.
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass for calculating the Dice + BCE loss.

        Args:
            inputs (Tensor): The predicted mask values (probabilities).
            targets (Tensor): The ground truth mask values (binary).
            smooth (float, optional): A small constant to avoid division by zero. Default is 1.

        Returns:
            Tensor: The combined Dice + BCE loss.
        """
        # Apply a threshold to the inputs for Dice loss
        inputs_final = (inputs > 0.5).float()
        inputs_final_flat = inputs_final.view(-1)
        targets_flat = targets.view(-1)
        
        # Compute Dice loss
        intersection = (inputs_final_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_final_flat.sum() + targets_flat.sum() + smooth)
        
        # Compute BCE loss
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # Combine both Dice and BCE losses
        Dice_BCE = 0.5 * BCE + 0.5 * dice_loss
        return Dice_BCE

class WeightedBCEDiceLoss(nn.Module):
    """
    Combination of Weighted Binary Cross-Entropy Loss and Dice Loss for imbalanced binary segmentation tasks.
    This loss function allows different weights for positive and negative classes to deal with class imbalance.

    Args:
        weight_pos (float, optional): Weight for the positive class. Default is 1.0.
        weight_neg (float, optional): Weight for the negative class. Default is 1.0.

    Methods:
        forward(inputs, targets, smooth=1): Computes the weighted BCE + Dice loss.
    """
    def __init__(self, weight_pos=1.0, weight_neg=1.0):
        super(WeightedBCEDiceLoss, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass for calculating the weighted BCE + Dice loss.

        Args:
            inputs (Tensor): The predicted mask values (probabilities).
            targets (Tensor): The ground truth mask values (binary).
            smooth (float, optional): A small constant to avoid division by zero. Default is 1.

        Returns:
            Tensor: The combined weighted BCE + Dice loss.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute Weighted BCE loss
        bce_loss = -(
            self.weight_pos * targets * torch.log(inputs + 1e-7) +
            self.weight_neg * (1 - targets) * torch.log(1 - inputs + 1e-7)
        ).mean()
        
        # Compute Dice loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # Return the combined loss
        return bce_loss + dice_loss

class FocalDiceLoss(nn.Module):
    """
    Combination of Focal Loss and Dice Loss for binary segmentation tasks, designed to focus on hard-to-classify examples.
    Focal loss helps address class imbalance by down-weighting easy negatives.

    Args:
        alpha (float, optional): Balancing factor for the focal loss. Default is 0.8.
        gamma (float, optional): Focusing parameter for the focal loss. Default is 2.

    Methods:
        forward(inputs, targets, smooth=1): Computes the focal + Dice loss.
    """
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass for calculating the focal + Dice loss.

        Args:
            inputs (Tensor): The predicted mask values (probabilities).
            targets (Tensor): The ground truth mask values (binary).
            smooth (float, optional): A small constant to avoid division by zero. Default is 1.

        Returns:
            Tensor: The combined focal + Dice loss.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute BCE loss for focal loss
        bce_loss = -(
            targets * torch.log(inputs + 1e-7) +
            (1 - targets) * torch.log(1 - inputs + 1e-7)
        )
        focal_loss = self.alpha * (1 - inputs) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()
        
        # Compute Dice loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # Return the combined focal and Dice loss
        return focal_loss + dice_loss

class MaskedBCEDiceLoss(nn.Module):
    """
    Binary Cross-Entropy and Dice Loss with masking support, allowing loss calculation only on regions of interest (ROI).
    This can be useful in cases where certain areas in the input are not relevant (e.g., background or invalid regions).

    Args:
        None

    Methods:
        forward(logits, targets, mask, smooth=1): Computes the masked BCE + Dice loss.
    """
    def forward(self, logits, targets, mask, smooth=1):
        """
        Forward pass for calculating the masked BCE + Dice loss.

        Args:
            logits (Tensor): The predicted logits (before sigmoid).
            targets (Tensor): The ground truth binary mask values.
            mask (Tensor): The mask tensor, indicating which regions to consider.
            smooth (float, optional): A small constant to avoid division by zero. Default is 1.

        Returns:
            Tensor: The combined BCE + Dice loss over the masked regions.
        """
        # Apply mask to logits and targets
        masked_logits = torch.sigmoid(logits) * mask
        masked_targets = targets * mask
        
        # Compute BCE loss for masked regions
        bce_loss = -(
            masked_targets * torch.log(masked_logits + 1e-7) +
            (1 - masked_targets) * torch.log(1 - masked_logits + 1e-7)
        ).mean()
        
        # Compute Dice loss for masked regions
        intersection = (masked_logits * masked_targets).sum(dim=(2, 3))
        dice_loss = 1 - (2. * intersection + smooth) / (
            masked_logits.sum(dim=(2, 3)) + masked_targets.sum(dim=(2, 3)) + smooth)
        dice_loss = dice_loss.mean()
        
        # Return the combined masked BCE and Dice loss
        return bce_loss + dice_loss
