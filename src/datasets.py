"""
Dataset loader for COCO 2017 with optional subsetting
Supports loading a subset of images for fast experimentation
(Matches original Fast Neural Style Transfer implementation)
"""

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch


# Mean and std for ImageNet normalization (used in original)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class CocoDataset(Dataset):
    """
    Simple COCO dataset loader for style transfer training
    
    Args:
        root_dir: Root directory containing images
        image_size: Size to resize images to (default: 256)
        subset_size: Optional limit on number of images to use (for faster training)
        transform: Optional custom transform (uses default if None)
    """
    
    def __init__(self, root_dir, image_size=256, subset_size=None, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform if transform is not None else self.default_transform()
        
        # Get all image files
        self.image_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
        
        # Limit to subset if specified
        if subset_size is not None:
            self.image_files = self.image_files[:subset_size]
        
        print(f"Loaded {len(self.image_files)} images from {root_dir}")
    
    def default_transform(self):
        """
        Default transformation for training images
        (Matches original: Resize with 1.15 factor, RandomCrop, ToTensor, Normalize)
        """
        return transforms.Compose([
            transforms.Resize(int(self.image_size * 1.15)),
            transforms.RandomCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            return transforms.ToTensor()(Image.new('RGB', (self.image_size, self.image_size)))


def get_style_image_transform(image_size=None):
    """
    Transform for style images
    (Matches original implementation)
    
    Args:
        image_size: Optional size to resize to (if None, no resizing)
    """
    transform_list = []
    if image_size is not None:
        transform_list.extend([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist())
    ])
    return transforms.Compose(transform_list)


def get_test_image_transform(image_size=None):
    """
    Transform for test/inference images
    (Matches original implementation)
    
    Args:
        image_size: Optional size to resize to (if None, keeps original size)
    """
    transform_list = []
    if image_size is not None:
        transform_list.append(transforms.Resize(image_size))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist())
    ])
    return transforms.Compose(transform_list)


def denormalize(tensors):
    """
    Denormalizes image tensors using ImageNet mean and std
    (Matches original implementation)
    
    Args:
        tensors: Image tensor(s)
    
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    
    if tensors.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    # Denormalize: x_original = x_normalized * std + mean
    for c in range(3):
        tensors[:, c].mul_(std[0, c]).add_(mean[0, c])
    
    return tensors


def deprocess(image_tensor):
    """
    Denormalizes and rescales image tensor to numpy array
    (Matches original deprocess function)
    
    Args:
        image_tensor: Image tensor (1, C, H, W) or (C, H, W)
    
    Returns:
        Numpy array image (H, W, C) in range [0, 255]
    """
    # Handle batch dimension
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.clone()
    else:
        image_tensor = image_tensor.unsqueeze(0).clone()
    
    # Denormalize
    image_tensor = denormalize(image_tensor)[0]
    
    # Scale to [0, 255]
    image_tensor *= 255
    
    # Clip and convert to numpy
    image_np = torch.clamp(image_tensor, 0, 255).cpu().numpy().astype(np.uint8)
    image_np = image_np.transpose(1, 2, 0)
    
    return image_np


def tensor_to_image(tensor):
    """
    Convert tensor to PIL Image (matches original implementation)
    
    Args:
        tensor: Image tensor of shape (C, H, W) or (B, C, H, W)
    
    Returns:
        PIL Image
    """
    # Handle batch dimension
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    # Use deprocess to properly denormalize and convert
    image_np = deprocess(tensor)
    return Image.fromarray(image_np)


def create_dataloader(root_dir, batch_size=4, image_size=256, subset_size=None, 
                      shuffle=True, num_workers=2):
    """
    Create a DataLoader for training
    
    Args:
        root_dir: Root directory containing images
        batch_size: Batch size for training
        image_size: Size to resize images to
        subset_size: Optional limit on number of images
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for loading
    
    Returns:
        DataLoader instance
    """
    dataset = CocoDataset(root_dir, image_size=image_size, subset_size=subset_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    return dataloader


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset module...")
    
    # Test with dummy directory
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Create some dummy images
    for i in range(5):
        img = Image.new('RGB', (256, 256), color=(i*50, i*50, i*50))
        img.save(os.path.join(temp_dir, f'test_{i}.jpg'))
    
    # Create dataset
    dataset = CocoDataset(temp_dir, image_size=256, subset_size=3)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading
    img = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
    
    # Test dataloader
    dataloader = create_dataloader(temp_dir, batch_size=2, subset_size=3, num_workers=0)
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        break
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print("Test complete!")
