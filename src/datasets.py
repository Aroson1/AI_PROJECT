"""
Dataset loader for COCO 2017 with optional subsetting
Supports loading a subset of images for fast experimentation
"""

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


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
        """Default transformation for training images"""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            # No normalization here - will be done in loss function for VGG
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
    
    Args:
        image_size: Optional size to resize to (if None, no resizing)
    """
    transform_list = []
    if image_size is not None:
        transform_list.extend([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ])
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def get_test_image_transform(image_size=None):
    """
    Transform for test/inference images
    
    Args:
        image_size: Optional size to resize to (if None, keeps original size)
    """
    transform_list = []
    if image_size is not None:
        transform_list.append(transforms.Resize(image_size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def denormalize(tensor):
    """
    Denormalize tensor (if it was normalized) and clip to valid range
    
    Args:
        tensor: Image tensor
    
    Returns:
        Denormalized tensor in range [0, 1]
    """
    return tensor.clamp(0, 1)


def tensor_to_image(tensor):
    """
    Convert tensor to PIL Image
    
    Args:
        tensor: Image tensor of shape (C, H, W) or (B, C, H, W)
    
    Returns:
        PIL Image
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize and convert to numpy
    tensor = denormalize(tensor)
    image = tensor.cpu().numpy()
    image = (image.transpose(1, 2, 0) * 255).astype('uint8')
    return Image.fromarray(image)


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
