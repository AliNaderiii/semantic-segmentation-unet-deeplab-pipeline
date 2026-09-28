"""
Professional Segmentation Models - U-Net, DeepLabV3+, SegFormer, FPN
Production-grade architectures via segmentation-models-pytorch
Real-world: Pascal VOC, Cityscapes, industrial inspection
"""

from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

ModelName = Literal['unet', 'deeplabv3plus', 'deeplabv3+', 'segformer', 'fpn']
EncoderName = Literal['resnet18', 'resnet34', 'resnet50', 'resnet101', 'mit_b2', 'mit_b4']

def get_unet(
    encoder: str = 'resnet34',
    num_classes: int = 2,
    pretrained: bool = True,
    in_channels: int = 3
) -> nn.Module:
    """
    U-Net with ResNet encoder - Best for thin structures, boundaries
    
    Args:
        encoder: Backbone (resnet18/34/50/101)
        num_classes: Number of segmentation classes
        pretrained: Use ImageNet pretrained encoder
        in_channels: Input channels (3 for RGB)
    
    Returns:
        SMP U-Net model
    
    Why U-Net:
        - Skip connections preserve spatial details for precise boundaries
        - Pretrained encoder accelerates convergence on small datasets
        - Lightweight (14M for resnet18) for CPU/edge inference
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights='imagenet' if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
        activation=None
    )
    return model

def get_deeplabv3plus(
    encoder: str = 'resnet50',
    num_classes: int = 2,
    pretrained: bool = True,
    in_channels: int = 3
) -> nn.Module:
    """
    DeepLabV3+ with ASPP - Best for multi-scale context
    
    Args:
        encoder: Backbone (resnet50/101)
        num_classes: Segmentation classes
        pretrained: ImageNet pretrained
        in_channels: Input channels
    
    Features:
        - Atrous Spatial Pyramid Pooling (ASPP) rates [1,6,12,18]
        - Output stride 16, multi-scale context
        - Decoder fuses low-level features (256ch) for boundary refinement
    """
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights='imagenet' if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
        activation=None
    )
    return model

def get_fpn(
    encoder: str = 'resnet34',
    num_classes: int = 2,
    pretrained: bool = True,
    in_channels: int = 3
) -> nn.Module:
    """
    Feature Pyramid Network - Multi-scale feature fusion
    Good for objects at different scales
    """
    model = smp.FPN(
        encoder_name=encoder,
        encoder_weights='imagenet' if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
        activation=None
    )
    return model

def get_segformer(
    encoder: str = 'mit_b2',
    num_classes: int = 2,
    pretrained: bool = True,
    in_channels: int = 3
) -> nn.Module:
    """
    SegFormer - Transformer-based, hierarchical, no positional encoding
    SOTA for complex scenes, efficient
    
    Fallback to DeepLabV3+ if SegFormer not available in SMP version
    """
    try:
        model = smp.Segformer(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        return model
    except Exception as e:
        print(f"SegFormer not available ({e}), falling back to DeepLabV3+")
        return get_deeplabv3plus(encoder='resnet50', num_classes=num_classes, pretrained=pretrained)

def get_model(
    model_name: str = 'unet',
    num_classes: int = 2,
    encoder: str = 'resnet34',
    pretrained: bool = True
) -> nn.Module:
    """
    Factory for segmentation models
    
    Args:
        model_name: unet, deeplabv3plus, fpn, segformer
        num_classes: 2 for binary, 21 for VOC multi-class
        encoder: Backbone name
        pretrained: Use ImageNet weights
    
    Returns:
        Segmentation model
    """
    model_name = model_name.lower()
    
    if model_name == 'unet':
        return get_unet(encoder=encoder, num_classes=num_classes, pretrained=pretrained)
    elif model_name in ['deeplabv3plus', 'deeplabv3+', 'deeplab']:
        # Default to resnet50 for deeplab if resnet34 passed
        enc = 'resnet50' if encoder == 'resnet34' else encoder
        return get_deeplabv3plus(encoder=enc, num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'fpn':
        return get_fpn(encoder=encoder, num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'segformer':
        return get_segformer(encoder=encoder, num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model {model_name}. Choose from: unet, deeplabv3plus, fpn, segformer")

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation - handles class imbalance
    
    Dice = 2*|pred∩true| / (|pred|+|true|)
    Loss = 1 - Dice
    
    Smooth avoids division by zero
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw logits
            targets: (B, H, W) class indices
        Returns:
            Dice loss scalar
        """
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """
    Combined Dice + Cross-Entropy - Standard for imbalanced segmentation
    
    Dice handles overlap (good for small foreground),
    CE handles pixel-wise accuracy (stabilizes training)
    
    For binary crack/defect: dice_weight 0.6, ce_weight 0.4
    For balanced: 0.5/0.5
    """
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        ce_loss = self.ce(logits, targets)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss

def count_parameters(model: nn.Module) -> float:
    """Count parameters in millions"""
    return sum(p.numel() for p in model.parameters()) / 1e6

if __name__ == "__main__":
    for name in ['unet', 'deeplabv3plus', 'fpn']:
        enc = 'resnet18' if name == 'unet' else 'resnet50'
        model = get_model(name, num_classes=2, encoder=enc)
        params = count_parameters(model)
        x = torch.randn(1, 3, 256, 256)
        y = model(x)
        print(f"{name} ({enc}): {params:.1f}M params, Input {x.shape} -> Output {y.shape}")
