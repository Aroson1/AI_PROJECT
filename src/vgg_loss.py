"""
VGG19-based Loss Functions for Neural Style Transfer
Includes content loss, style loss, and total variation loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGG19(nn.Module):
    """
    VGG19 feature extractor for style transfer
    Extracts features from specific layers for computing content and style losses
    """
    
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        
        # Split VGG19 into different feature extraction stages
        # relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(4):  # relu1_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):  # relu2_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):  # relu3_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):  # relu4_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):  # relu5_3
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """Extract features from multiple VGG layers"""
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return h1, h2, h3, h4, h5


def gram_matrix(y):
    """
    Compute Gram matrix for style representation
    
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
    """
    
    def __init__(self, content_weight=1.0, style_weight=10.0, tv_weight=1e-6):
        super(StyleTransferLoss, self).__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        self.vgg = VGG19(requires_grad=False)
        self.mse_loss = nn.MSELoss()
        
        # Normalization for VGG (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize_batch(self, batch):
        """Normalize batch for VGG input"""
        return (batch - self.mean) / self.std
    
    def content_loss(self, generated_features, content_features):
        """Compute content loss using relu3_3 features"""
        return self.mse_loss(generated_features[2], content_features[2])
    
    def style_loss(self, generated_features, style_features):
        """Compute style loss using Gram matrices from multiple layers"""
        style_loss = 0
        # Use relu1_2, relu2_2, relu3_3, relu4_3 for style
        for gen_feat, style_feat in zip(generated_features[:4], style_features[:4]):
            gen_gram = gram_matrix(gen_feat)
            style_gram = gram_matrix(style_feat)
            style_loss += self.mse_loss(gen_gram, style_gram)
        return style_loss
    
    def total_variation_loss(self, generated):
        """
        Compute total variation loss for smoothness
        Encourages spatial smoothness in the generated image
        """
        tv_loss = torch.mean(torch.abs(generated[:, :, :, :-1] - generated[:, :, :, 1:])) + \
                  torch.mean(torch.abs(generated[:, :, :-1, :] - generated[:, :, 1:, :]))
        return tv_loss
    
    def forward(self, generated, content, style_features_gram=None, style=None):
        """
        Compute total loss
        
        Args:
            generated: Generated image batch
            content: Content image batch
            style_features_gram: Pre-computed style Gram matrices (optional)
            style: Style image (optional, used if style_features_gram not provided)
        
        Returns:
            Dictionary with total loss and individual loss components
        """
        # Normalize images for VGG
        generated_normalized = self.normalize_batch(generated)
        content_normalized = self.normalize_batch(content)
        
        # Extract features
        generated_features = self.vgg(generated_normalized)
        content_features = self.vgg(content_normalized)
        
        # Compute content loss
        c_loss = self.content_loss(generated_features, content_features)
        
        # Compute style loss
        if style_features_gram is None and style is not None:
            style_normalized = self.normalize_batch(style)
            style_features = self.vgg(style_normalized)
            s_loss = self.style_loss(generated_features, style_features)
        else:
            # Use pre-computed Gram matrices (more efficient for batch training)
            s_loss = 0
            for gen_feat, style_gram in zip(generated_features[:4], style_features_gram):
                gen_gram = gram_matrix(gen_feat)
                s_loss += self.mse_loss(gen_gram, style_gram.expand_as(gen_gram))
        
        # Compute total variation loss
        tv_loss = self.total_variation_loss(generated)
        
        # Combine losses
        total_loss = (self.content_weight * c_loss + 
                     self.style_weight * s_loss + 
                     self.tv_weight * tv_loss)
        
        return {
            'total': total_loss,
            'content': c_loss.item(),
            'style': s_loss.item(),
            'tv': tv_loss.item()
        }
    
    def extract_style_gram(self, style_image):
        """
        Pre-compute and return Gram matrices for a style image
        Useful for efficient batch training
        """
        with torch.no_grad():
            style_normalized = self.normalize_batch(style_image)
            style_features = self.vgg(style_normalized)
            style_grams = [gram_matrix(feat) for feat in style_features[:4]]
        return style_grams


if __name__ == "__main__":
    # Test the loss module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_fn = StyleTransferLoss(content_weight=1.0, style_weight=10.0, tv_weight=1e-6)
    loss_fn.to(device)
    
    # Dummy inputs
    generated = torch.randn(2, 3, 256, 256).to(device)
    content = torch.randn(2, 3, 256, 256).to(device)
    style = torch.randn(1, 3, 256, 256).to(device)
    
    # Compute loss
    losses = loss_fn(generated, content, style=style)
    print("Loss components:")
    for key, value in losses.items():
        print(f"  {key}: {value if isinstance(value, float) else value.item():.4f}")
