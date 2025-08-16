import os
import matplotlib.pyplot as plt
import argparse

def save_predictions(epoch, video, target, output, activations, save_dir='./predictions'):
    """
    Saves the predicted results, ground truth, and activations for visualization during training.
    The results are saved as PNG images for each slice of the video at a given epoch.

    Args:
        epoch (int): The current training epoch. This is used to label the saved file.
        video (Tensor): The input video tensor with shape (batch_size, channels, depth, height, width).
        target (Tensor): The ground truth binary mask tensor with shape (batch_size, depth, height, width).
        output (Tensor): The predicted output tensor with shape (batch_size, depth, height, width).
        activations (list): A list of activation maps (tensors) from different layers in the model.
        save_dir (str, optional): Directory where the prediction images will be saved. Default is './predictions'.
        
    Saves:
        PNG images of the input, target, and output as well as the activations for each slice.
    """
    # Create the directory to save images if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays for visualization
    video_np = video[0, 0].cpu().detach().numpy()
    target_np = target[0].cpu().detach().numpy()
    
    # Convert output probabilities to binary (threshold at 0.5)
    output = (output > 0.5).float()
    output_np = output[0].cpu().detach().numpy()
    
    # Get the number of slices in the video (along the depth axis)
    num_slices = video_np.shape[-1]

    # Filter activations to keep only 3D tensors
    activations_3d = [activation for activation in activations if activation[0, 0].dim() == 3]

    # Determine the number of columns for the subplot (based on the number of activations)
    num_cols = max(3, len(activations_3d))
    num_rows = 2

    # Iterate over slices to save individual slice visualizations
    for slice_idx in range(num_slices):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

        # Plot input image, ground truth, and prediction for the current slice
        axes[0, 0].imshow(video_np[:, :, slice_idx], cmap='gray')
        axes[0, 0].set_title(f'Input Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(target_np[:, :, slice_idx], cmap='gray')
        axes[0, 1].set_title(f'Ground Truth')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(output_np[:, :, slice_idx], cmap='gray')
        axes[0, 2].set_title(f'Prediction')
        axes[0, 2].axis('off')

        # Plot activations for each 3D layer
        for layer_idx, activation in enumerate(activations_3d):
            activation_np = activation[0, 0].cpu().detach().numpy()
            row_idx = 1
            col_idx = layer_idx
            axes[row_idx, col_idx].imshow(activation_np[:, :, slice_idx], cmap='jet', interpolation='nearest')
            axes[row_idx, col_idx].set_title(f'3D Layer {layer_idx}')
            axes[row_idx, col_idx].axis('off')

        # Turn off axes for empty subplots
        for i in range(num_rows):
            for j in range(num_cols):
                if not axes[i, j].has_data():
                    axes[i, j].axis('off')
        
        # Save the combined figure for the current slice
        combined_save_path = os.path.join(save_dir, f'epoch_{epoch}_slice_{slice_idx}_combined.png')
        plt.tight_layout()
        plt.savefig(combined_save_path, bbox_inches='tight')
        plt.close(fig)

def generate_save_path(task_name, patch_size, batch_size, lr, epoch=None):
    """
    Generates a file path for saving model predictions or weights based on the task and hyperparameters.

    Args:
        task_name (str): The name of the task (e.g., segmentation, classification).
        patch_size (tuple): The size of the patches used in training.
        batch_size (int): The batch size used in training.
        lr (float): The learning rate used in training.
        epoch (int, optional): The current epoch number. If provided, includes it in the file path.

    Returns:
        str: A file path for saving the model or predictions.
    """
    # Create a base name using the task name, patch size, batch size, and learning rate
    base_name = f"{task_name}_{patch_size}_{batch_size}_{lr}"
    
    # If epoch is provided, include it in the filename for predictions
    if epoch is not None:
        return f"./predictions/{base_name}_epoch_{epoch}.png"
    
    # If no epoch, generate a path for saving the model
    return f"./new_model_{base_name}.pth"

class TupleAction(argparse.Action):
    """
    Custom argparse action to parse a comma-separated tuple input from the command line.

    Converts a comma-separated string into a tuple of integers.

    Args:
        parser (argparse.ArgumentParser): The argument parser.
        namespace (argparse.Namespace): The object to store the parsed values.
        values (str): The comma-separated string to be converted into a tuple.
        option_string (str, optional): The option string that triggered this action.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        # Split the input string by commas and convert each element to an integer, then store in the namespace
        setattr(namespace, self.dest, tuple(map(int, values.split(','))))
