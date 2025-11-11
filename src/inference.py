"""
Inference script for Fast Neural Style Transfer
Load a trained model and stylize input images
"""

import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt

from transformer import TransformerNet
from datasets import get_test_image_transform, tensor_to_image


def stylize_image(model, image_path, output_path=None, device='cpu', display=True):
    """
    Stylize a single image using the trained model
    
    Args:
        model: Trained TransformerNet model
        image_path: Path to input content image
        output_path: Optional path to save output (default: output.jpg)
        device: Device to run inference on
        display: Whether to display the result
    
    Returns:
        PIL Image of stylized output
    """
    # Load and transform input image
    content_image = Image.open(image_path).convert('RGB')
    original_size = content_image.size
    
    # Transform for model
    transform = get_test_image_transform(image_size=None)
    content_tensor = transform(content_image).unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output_tensor = model(content_tensor)
    
    # Convert to image
    output_image = tensor_to_image(output_tensor)
    
    # Resize back to original size if needed
    if output_image.size != original_size:
        output_image = output_image.resize(original_size, Image.LANCZOS)
    
    # Save output
    if output_path is None:
        output_path = 'output.jpg'
    output_image.save(output_path)
    print(f"Stylized image saved to: {output_path}")
    
    # Display side-by-side comparison
    if display:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(content_image)
            axes[0].set_title('Original Content', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(output_image)
            axes[1].set_title('Stylized Output', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")
    
    return output_image


def stylize_batch(model, input_dir, output_dir, device='cpu', file_extensions=('.jpg', '.jpeg', '.png')):
    """
    Stylize all images in a directory
    
    Args:
        model: Trained TransformerNet model
        input_dir: Directory containing input images
        output_dir: Directory to save stylized images
        device: Device to run inference on
        file_extensions: Tuple of valid file extensions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(file_extensions)]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to stylize")
    
    for idx, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"stylized_{filename}")
        
        print(f"[{idx}/{len(image_files)}] Processing {filename}...")
        try:
            stylize_image(model, input_path, output_path, device=device, display=False)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"\nAll images processed! Outputs saved to: {output_dir}")


def load_model(checkpoint_path, device='cpu'):
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model on
    
    Returns:
        Loaded TransformerNet model
    """
    print(f"Loading model from: {checkpoint_path}")
    model = TransformerNet().to(device)
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stylize images using trained Fast Neural Style Transfer model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='output.jpg',
                        help='Path to output image or directory (default: output.jpg)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all images in input directory')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference (default: auto-detect GPU)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display results (useful for batch processing)')
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()
    
    # Set device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device=device)
    
    # Run inference
    if args.batch:
        # Batch processing
        stylize_batch(model, args.input, args.output, device=device)
    else:
        # Single image
        stylize_image(model, args.input, args.output, device=device, display=not args.no_display)
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()
