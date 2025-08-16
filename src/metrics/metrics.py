import torch
import torch.nn.functional as F

def dice_score(pred, target, smooth=1e-6):
    """
    Computes the Dice Similarity Coefficient (DSC) between predicted and target binary masks.
    The Dice score measures the overlap between the predicted and target masks, with values 
    ranging from 0 (no overlap) to 1 (perfect overlap).

    Args:
        pred (Tensor): The predicted binary mask (probabilities or logits).
        target (Tensor): The ground truth binary mask.
        smooth (float, optional): A small constant to avoid division by zero. Default is 1e-6.

    Returns:
        float: The Dice score between the predicted and target masks.
    """
    # Flatten the input tensors to 1D for computation
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    # Compute the intersection between predicted and target
    intersection = (pred * target).sum()
    
    # Compute the Dice score
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def nsd_score(pred, target, tolerance=1.0):
    """
    Computes the Normalized Surface Dice (NSD) score between the predicted and target surfaces.
    The NSD score evaluates the overlap between the surfaces of the predicted and target masks.
    It is sensitive to boundary differences, making it useful for evaluating segmentation accuracy 
    along the object boundaries.

    Args:
        pred (Tensor): The predicted binary mask (probabilities or logits).
        target (Tensor): The ground truth binary mask.
        tolerance (float, optional): The maximum allowed distance for considering two surface points as matching. Default is 1.0.

    Returns:
        float: The NSD score, representing the average overlap between the predicted and target surfaces.
    """
    # Compute the surface of the predicted and target masks by subtracting the inner region
    pred_surface = pred - F.max_pool3d(pred, kernel_size=3, stride=1, padding=1)
    target_surface = target - F.max_pool3d(target, kernel_size=3, stride=1, padding=1)
    
    # Get the coordinates of the surface points (non-zero values)
    pred_coords = torch.nonzero(pred_surface)
    target_coords = torch.nonzero(target_surface)

    # If no surface points are detected in either prediction or target, return 0 score
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 0.0

    # Compute pairwise distances between the predicted and target surface points
    distances = torch.cdist(pred_coords.float(), target_coords.float())
    
    # Find the minimum distance for each predicted surface point and check if it's within tolerance
    pred_matches = (distances.min(dim=1).values <= tolerance).float().mean().item()
    
    # Find the minimum distance for each target surface point and check if it's within tolerance
    target_matches = (distances.min(dim=0).values <= tolerance).float().mean().item()

    # Return the average of matching surfaces
    return (pred_matches + target_matches) / 2.0
