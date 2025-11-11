"""
Training script for Fast Neural Style Transfer
Default settings optimized for Colab Free Tier
"""

import os
import time
import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
from PIL import Image

from transformer import TransformerNet
from vgg_loss import StyleTransferLoss
from datasets import create_dataloader, get_style_image_transform


def train(args):
    """
    Main training loop for Fast Neural Style Transfer
    
    Args:
        args: Training arguments containing all hyperparameters
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Create output directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize model
    print("Initializing transformer network...")
    transformer = TransformerNet().to(device)
    
    # Initialize loss function
    print("Initializing loss function with VGG19...")
    loss_fn = StyleTransferLoss(
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(transformer.parameters(), lr=args.lr)
    
    # Load style image
    print(f"Loading style image: {args.style_image}")
    style_transform = get_style_image_transform(args.image_size)
    style_image = Image.open(args.style_image).convert('RGB')
    style_tensor = style_transform(style_image).unsqueeze(0).to(device)
    
    # Pre-compute style Gram matrices for efficiency
    print("Computing style features...")
    style_grams = loss_fn.extract_style_gram(style_tensor)
    
    # Create dataloader
    print(f"Loading dataset from: {args.dataset_path}")
    dataloader = create_dataloader(
        args.dataset_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        subset_size=args.subset_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Image size: {args.image_size}")
    print(f"Content weight: {args.content_weight}, Style weight: {args.style_weight}, TV weight: {args.tv_weight}\n")
    
    loss_history = {
        'total': [],
        'content': [],
        'style': [],
        'tv': []
    }
    
    for epoch in range(args.epochs):
        transformer.train()
        epoch_losses = {'total': 0, 'content': 0, 'style': 0, 'tv': 0}
        num_batches = 0
        
        epoch_start_time = time.time()
        
        # Progress bar for batches
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, content_batch in enumerate(pbar):
            content_batch = content_batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            generated_batch = transformer(content_batch)
            
            # Compute loss
            losses = loss_fn(generated_batch, content_batch, style_features_gram=style_grams)
            
            # Backward pass
            losses['total'].backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += losses['total'].item()
            epoch_losses['content'] += losses['content']
            epoch_losses['style'] += losses['style']
            epoch_losses['tv'] += losses['tv']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.2f}",
                'content': f"{losses['content']:.2f}",
                'style': f"{losses['style']:.2f}"
            })
        
        # Calculate average losses for epoch
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        # Store losses
        for key in loss_history:
            loss_history[key].append(avg_losses[key])
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s")
        print(f"  Average Total Loss: {avg_losses['total']:.4f}")
        print(f"  Average Content Loss: {avg_losses['content']:.4f}")
        print(f"  Average Style Loss: {avg_losses['style']:.4f}")
        print(f"  Average TV Loss: {avg_losses['tv']:.6f}\n")
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(transformer.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}\n")
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save(transformer.state_dict(), final_model_path)
    print(f"Training complete! Final model saved to: {final_model_path}")
    
    # Plot and save loss curves
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(loss_history['total'], 'b-', linewidth=2)
        plt.title('Total Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(loss_history['content'], 'g-', label='Content', linewidth=2)
        plt.plot(loss_history['style'], 'r-', label='Style', linewidth=2)
        plt.title('Content & Style Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(loss_history['tv'], 'm-', linewidth=2)
        plt.title('Total Variation Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_plot_path = os.path.join(args.checkpoint_dir, 'loss_plot.png')
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        print(f"Loss plot saved to: {loss_plot_path}")
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping loss plot.")
    
    return loss_history


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Fast Neural Style Transfer')
    
    # Required arguments
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to training images (e.g., COCO dataset)')
    parser.add_argument('--style-image', type=str, required=True,
                        help='Path to style image')
    
    # Training hyperparameters (optimized for Colab Free Tier)
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Size of training images (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    
    # Loss weights
    parser.add_argument('--content-weight', type=float, default=1.0,
                        help='Weight for content loss (default: 1.0)')
    parser.add_argument('--style-weight', type=float, default=10.0,
                        help='Weight for style loss (default: 10.0)')
    parser.add_argument('--tv-weight', type=float, default=1e-6,
                        help='Weight for total variation loss (default: 1e-6)')
    
    # Dataset options
    parser.add_argument('--subset-size', type=int, default=2000,
                        help='Number of images to use (default: 2000)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers (default: 2)')
    
    # Output options
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                        help='Directory to save checkpoints (default: models/checkpoints)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loss_history = train(args)
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Final Total Loss: {loss_history['total'][-1]:.4f}")
    print(f"Final Content Loss: {loss_history['content'][-1]:.4f}")
    print(f"Final Style Loss: {loss_history['style'][-1]:.4f}")
    print(f"Final TV Loss: {loss_history['tv'][-1]:.6f}")
    print("="*60)
