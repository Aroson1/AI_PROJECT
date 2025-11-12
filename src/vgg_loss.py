"""
VGG16-based Loss Functions for Neural Style Transfer
Includes content loss, style loss, and total variation loss
(Matches the original Fast Neural Style Transfer implementation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import namedtuple


class VGG16(nn.Module):
    """
    VGG16 feature extractor for style transfer
    Extracts features from relu1_2, relu2_2, relu3_3, relu4_3 layers
    (Matches original implementation)
    """
    
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        
        # Split VGG16 into different feature extraction stages
        # relu1_2, relu2_2, relu3_3, relu4_3
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(4):  # relu1_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):  # relu2_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):  # relu3_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):  # relu4_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """Extract features from multiple VGG layers"""
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


def gram_matrix(y):
    """
    Compute Gram matrix for style representation
    (Matches original implementation)
    
    Args:
        y: Feature maps of shape (batch, channels, height, width)
    
    Returns:
        Gram matrix of shape (batch, channels, channels)
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class StyleTransferLoss(nn.Module):
    """
    Combined loss function for neural style transfer
    Includes content loss, style loss, and total variation loss
    (Matches original implementation)
    """
    
    def __init__(self, content_weight=1e5, style_weight=1e10, tv_weight=0):
        super(StyleTransferLoss, self).__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        self.vgg = VGG16(requires_grad=False)
        self.mse_loss = nn.MSELoss()
    
    def content_loss(self, generated_features, content_features):
        """Compute content loss using relu2_2 features (matches original)"""
        return self.mse_loss(generated_features.relu2_2, content_features.relu2_2)
    
    def style_loss(self, generated_features, style_grams):
        """
        Compute style loss using Gram matrices from all 4 layers
        (Matches original implementation)
        """
        style_loss = 0
        # Use all 4 layers for style: relu1_2, relu2_2, relu3_3, relu4_3
        for gen_feat, style_gram in zip(generated_features, style_grams):
            gen_gram = gram_matrix(gen_feat)
            style_loss += self.mse_loss(gen_gram, style_gram[:gen_feat.size(0)])
        return style_loss
    
    def total_variation_loss(self, generated):
        """
        Compute total variation loss for smoothness
        Encourages spatial smoothness in the generated image
        """
        tv_loss = torch.mean(torch.abs(generated[:, :, :, :-1] - generated[:, :, :, 1:])) + \
                  torch.mean(torch.abs(generated[:, :, :-1, :] - generated[:, :, 1:, :]))
        return tv_loss
    
    def forward(self, generated, content, style_features_gram=None):
        """
        Compute total loss
        (Matches original implementation)
        
        Args:
            generated: Generated image batch (already normalized)
            content: Content image batch (already normalized)
            style_features_gram: Pre-computed style Gram matrices (list of 4 tensors)
        
        Returns:
            Dictionary with total loss and individual loss components
        """
        # Extract features directly (images are already normalized in transforms)
        generated_features = self.vgg(generated)
        content_features = self.vgg(content)
        
        # Compute content loss
        c_loss = self.content_loss(generated_features, content_features)
        
        # Compute style loss using pre-computed Gram matrices
        s_loss = self.style_loss(generated_features, style_features_gram)
        
        # Compute total variation loss
        tv_loss = self.total_variation_loss(generated) if self.tv_weight > 0 else 0
        
        # Combine losses (use original weights)
        total_loss = (self.content_weight * c_loss + 
                     self.style_weight * s_loss)
        
        if self.tv_weight > 0:
            total_loss += self.tv_weight * tv_loss
            tv_val = tv_loss.item() if isinstance(tv_loss, torch.Tensor) else tv_loss
        else:
            tv_val = 0
        
        return {
            'total': total_loss,
            'content': c_loss.item(),
            'style': s_loss.item(),
            'tv': tv_val
        }
    
    def extract_style_gram(self, style_image):
        """
        Pre-compute and return Gram matrices for a style image
        Useful for efficient batch training
        (Matches original implementation)
        
        Args:
            style_image: Style image batch (already normalized)
        """
        with torch.no_grad():
            style_features = self.vgg(style_image)
            # Compute Gram matrices for all 4 layers
            style_grams = [gram_matrix(feat) for feat in style_features]
        return style_grams


if __name__ == "__main__":
    # Test the loss module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_fn = StyleTransferLoss(content_weight=1e5, style_weight=1e10, tv_weight=0)
    loss_fn.to(device)
    
    # Dummy inputs
    generated = torch.randn(2, 3, 256, 256).to(device)
    content = torch.randn(2, 3, 256, 256).to(device)
    style = torch.randn(1, 3, 256, 256).to(device)
    
    # Pre-compute style grams
    style_grams = loss_fn.extract_style_gram(style)
    
    # Compute loss
    losses = loss_fn(generated, content, style_features_gram=style_grams)
    print("Loss components:")
    for key, value in losses.items():
        print(f"  {key}: {value if isinstance(value, float) else value.item():.4f}")
