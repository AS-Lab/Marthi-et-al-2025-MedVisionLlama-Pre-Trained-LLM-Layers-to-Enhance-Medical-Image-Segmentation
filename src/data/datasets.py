import os
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, Subset, ConcatDataset
from scipy.ndimage import zoom, rotate

def get_img_label_paths(data_file):
    """
    Reads a CSV file containing image and label paths and returns a list of tuples.
    
    Args:
        data_file (str): Path to the CSV file where each line contains the image and label paths separated by a comma.
        
    Returns:
        list: A list of tuples, where each tuple contains the image and label paths.
    """
    img_label_plist = []
    with open(data_file, 'r') as f:
        for l in f:
            img_label_plist.append(l.strip().split(','))
    return img_label_plist

def get_img(img_path):
    """
    Loads an image from the specified file path using SimpleITK and converts it to a numpy array.
    
    Args:
        img_path (str): Path to the image file.
        
    Returns:
        np.ndarray: A numpy array representing the image.
    """
    img_itk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_itk)
    return img

def resize_3d(img, resize_shape, order=0):
    """
    Resizes a 3D image to the specified shape using zoom interpolation.
    
    Args:
        img (np.ndarray): The input 3D image to be resized.
        resize_shape (tuple): The target shape to resize the image to.
        order (int, optional): The order of the spline interpolation. Default is 0 (nearest neighbor).
        
    Returns:
        np.ndarray: The resized 3D image.
    """
    zoom0 = resize_shape[0] / img.shape[0]
    zoom1 = resize_shape[1] / img.shape[1]
    zoom2 = resize_shape[2] / img.shape[2]
    img = zoom(img, (zoom0, zoom1, zoom2), order=order)
    return img

def z_score_norm(img):
    """
    Normalizes the image using Z-score normalization (zero mean, unit variance).
    
    Args:
        img (np.ndarray): The input image to be normalized.
        
    Returns:
        np.ndarray: The normalized image.
    """
    u = np.mean(img)
    s = np.std(img)
    img -= u
    if s == 0:
        return img
    return img / s

def min_max_norm(img, epsilon=1e-5):
    """
    Normalizes the image to the range [0, 1] using min-max normalization.
    
    Args:
        img (np.ndarray): The input image to be normalized.
        epsilon (float, optional): A small constant added to avoid division by zero. Default is 1e-5.
        
    Returns:
        np.ndarray: The normalized image.
    """
    minv = np.min(img)
    maxv = np.max(img)
    return (img - minv + epsilon) / (maxv - minv + epsilon)

class MyDataset(Dataset):
    """
    Custom Dataset class for loading medical images and their corresponding labels.
    
    Args:
        data_dir (str): Directory where the image and label files are stored.
        data_file (str): Path to the CSV file containing image and label file paths.
        image_size (tuple): The target size to which each image will be resized.
        transforms (callable, optional): A function/transform to apply to the input image.
        target_transforms (callable, optional): A function/transform to apply to the target label.
    """
    def __init__(self, data_dir, data_file, image_size, transforms=None, target_transforms=None):
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.img_label_plist = get_img_label_paths(data_file)
        self.input_shape = image_size
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.img_label_plist)

    def __getitem__(self, index):
        """
        Retrieves and processes a sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve.
            
        Returns:
            tuple: A tuple (input_image, target_image) where input_image and target_image are 
                   numpy arrays (after preprocessing and transformations).
        """
        x_path, y_path = self.img_label_plist[index]
        img_x = get_img(os.path.join(self.data_dir, x_path)).astype(np.float32)
        img_y = get_img(os.path.join(self.data_dir, y_path)).astype(np.float32)
        
        # Normalize and resize the images
        img_x = z_score_norm(img_x)
        img_x = min_max_norm(img_x)
        if img_x.ndim == 4:
            img_x = img_x[0, :, :, :]
            
        img_x = resize_3d(img_x, self.input_shape, 1)
        img_x = np.expand_dims(img_x, 0)

        img_y = resize_3d(img_y, self.input_shape, 1)
        img_y = (img_y > 0).astype(int)  # Binarize the label
        
        # Apply any additional transformations if specified
        if self.transforms is not None:
            img_x = self.transforms(img_x)
        
        if self.target_transforms is not None:
            img_y = self.target_transforms(img_y)
        
        return img_x, img_y

class RotatedDataset(Dataset):
    """
    Dataset class that applies random rotations to the images and labels.
    
    Args:
        base_dataset (Dataset): The base dataset to apply rotations on.
        rotation_range (tuple): The range of angles (in degrees) for random rotations.
    """
    def __init__(self, base_dataset, rotation_range=(-30, 30)):
        super(RotatedDataset, self).__init__()
        self.base_dataset = base_dataset
        self.rotation_range = rotation_range

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.base_dataset)

    def __getitem__(self, index):
        """
        Retrieves a rotated sample from the base dataset.
        
        Args:
            index (int): Index of the sample to retrieve.
            
        Returns:
            tuple: A tuple (rotated_input_image, rotated_target_image).
        """
        img_x, img_y = self.base_dataset[index]
        angle = np.random.uniform(*self.rotation_range)
        img_x_rot = rotate(img_x, angle, axes=(1, 2), reshape=False).copy()
        img_y_rot = rotate(img_y, angle, axes=(1, 2), reshape=False).copy()
        return img_x_rot, img_y_rot

class FlippedDataset(Dataset):
    """
    Dataset class that applies random flipping (horizontal or vertical) to the images and labels.
    
    Args:
        original_dataset (Dataset): The base dataset to apply flipping on.
        flip_axis (str): Axis to flip along ('horizontal' or 'vertical').
    """
    def __init__(self, original_dataset, flip_axis='horizontal'):
        self.original_dataset = original_dataset
        if flip_axis not in ['horizontal', 'vertical']:
            raise ValueError("flip_axis must be either 'horizontal' or 'vertical'.")
        self.flip_axis = flip_axis

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.original_dataset)

    def __getitem__(self, index):
        """
        Retrieves a flipped sample from the original dataset.
        
        Args:
            index (int): Index of the sample to retrieve.
            
        Returns:
            tuple: A tuple (flipped_input_image, flipped_target_image).
        """
        img_x, img_y = self.original_dataset[index]
        if self.flip_axis == 'horizontal':
            img_x = np.flip(img_x, axis=2).copy()
            img_y = np.flip(img_y, axis=2).copy()
        elif self.flip_axis == 'vertical':
            img_x = np.flip(img_x, axis=1).copy()
            img_y = np.flip(img_y, axis=1).copy()
        return img_x, img_y

class IntensityAdjustedDataset(Dataset):
    """
    Dataset class that applies random intensity adjustments (brightness and contrast) to the images.
    
    Args:
        original_dataset (Dataset): The base dataset to apply intensity adjustments on.
        brightness_range (tuple): The range of brightness factors for adjustment.
        contrast_range (tuple): The range of contrast factors for adjustment.
    """
    def __init__(self, original_dataset, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
        self.original_dataset = original_dataset
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.original_dataset)

    def __getitem__(self, index):
        """
        Retrieves an intensity-adjusted sample from the original dataset.
        
        Args:
            index (int): Index of the sample to retrieve.
            
        Returns:
            tuple: A tuple (adjusted_input_image, target_image).
        """
        img_x, img_y = self.original_dataset[index]
        img_x = self._adjust_brightness(img_x)
        img_x = self._adjust_contrast(img_x)
        return img_x, img_y

    def _adjust_brightness(self, img):
        """
        Adjusts the brightness of the image by multiplying it by a random factor.
        
        Args:
            img (np.ndarray): The input image to adjust.
            
        Returns:
            np.ndarray: The brightness-adjusted image.
        """
        brightness_factor = np.random.uniform(*self.brightness_range)
        return img * brightness_factor

    def _adjust_contrast(self, img):
        """
        Adjusts the contrast of the image by applying a random contrast factor.
        
        Args:
            img (np.ndarray): The input image to adjust.
            
        Returns:
            np.ndarray: The contrast-adjusted image.
        """
        contrast_factor = np.random.uniform(*self.contrast_range)
        mean_intensity = np.mean(img)
        return (img - mean_intensity) * contrast_factor + mean_intensity