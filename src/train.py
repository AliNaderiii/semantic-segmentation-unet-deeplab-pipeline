"""
Professional Training Pipeline - Real Segmentation on Pascal VOC 2012
U-Net & DeepLabV3+ with Combined Dice+CE, ReduceLROnPlateau, real metrics
"""

import argparse
from pathlib import Path
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from models import get_model, CombinedLoss, count_parameters

try:
    from config import load_config
except ImportError:
    load_config = None

def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2):
    """Calculate IoU per class, mean ignoring nan (standard segmentation metric)"""
    ious = []
    pred_cls = torch.argmax(pred, dim=1)
    for cls in range(num_classes):
        pred_mask = (pred_cls == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    valid = [iou for iou in ious if not np.isnan(iou)]
    miou = float(np.mean(valid)) if valid else 0.0
    return miou, ious

def calculate_dice(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2):
    """Dice coefficient per class"""
    dices = []
    pred_cls = torch.argmax(pred, dim=1)
    for cls in range(num_classes):
        pred_mask = (pred_cls == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().float()
        total = pred_mask.sum().float() + target_mask.sum().float()
        if total == 0:
            dices.append(float('nan'))
        else:
            dices.append((2 * intersection / total).item())
    valid = [d for d in dices if not np.isnan(d)]
    mdice = float(np.mean(valid)) if valid else 0.0
    return mdice, dices

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_miou, total_dice = 0, 0, 0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        miou, _ = calculate_iou(outputs.detach(), masks)
        mdice, _ = calculate_dice(outputs.detach(), masks)
        total_loss += loss.item()
        total_miou += miou
        total_dice += mdice
    n = len(loader)
    return total_loss/n, total_miou/n, total_dice/n

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_miou, total_dice = 0, 0, 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            miou, _ = calculate_iou(outputs, masks)
            mdice, _ = calculate_dice(outputs, masks)
            total_loss += loss.item()
            total_miou += miou
            total_dice += mdice
    n = len(loader)
    return total_loss/n, total_miou/n, total_dice/n

def plot_history(history, model_name, save_dir):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss'])+1)
    
    # Combined plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss (Real)'); axes[0].legend(); axes[0].grid(alpha=0.3)
    
    axes[1].plot(epochs, history['train_miou'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_miou'], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('mIoU')
    axes[1].set_title(f'{model_name} - mIoU (Real)'); axes[1].legend(); axes[1].grid(alpha=0.3)
    
    axes[2].plot(epochs, history['train_dice'], 'b-', label='Train', linewidth=2)
    axes[2].plot(epochs, history['val_dice'], 'r-', label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Dice')
    axes[2].set_title(f'{model_name} - Dice (Real)'); axes[2].legend(); axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / f"training_curves_{model_name}_real.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Dice detailed
    plt.figure(figsize=(8,5))
    plt.plot(epochs, history['train_dice'], 'b-o', label='Train Dice', linewidth=2)
    plt.plot(epochs, history['val_dice'], 'r-s', label='Val Dice', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.title(f'{model_name} - Dice Evolution (Real)')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(save_path / f"dice_curve_{model_name}_real.png", dpi=300, bbox_inches='tight')
    plt.close()

def train_model(model_name='unet', encoder='resnet18', epochs=3, batch_size=4, img_size=128, num_samples=80, lr=1e-4, save_dir=None, config=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Config override
    if config:
        epochs = config['training']['epochs']
        batch_size = config['training']['batch_size']
        img_size = config['dataset']['img_size']
        num_samples = config['dataset'].get('num_samples', 80)
        lr = config['training']['lr']
        model_name = config['model']['name']
        encoder = config['model']['encoder']
    
    # Paths
    if save_dir is None:
        base = Path(__file__).parent.parent
        save_path = base / "models"
        data_root = base / "data"
        reports_root = base / "reports"
    else:
        save_path = Path(save_dir)
        base = Path(__file__).parent.parent
        data_root = base / "data"
        reports_root = base / "reports"
    
    save_path.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading real VOC data: {num_samples} samples, img_size {img_size}, batch {batch_size}")
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, img_size=img_size, num_samples=num_samples, root=str(data_root))
    
    print(f"Creating model: {model_name} with {encoder} - {count_parameters(get_model(model_name,2,encoder)):.1f}M params")
    model = get_model(model_name=model_name, num_classes=2, encoder=encoder)
    model = model.to(device)
    
    criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    history = {'train_loss':[], 'train_miou':[], 'train_dice':[], 'val_loss':[], 'val_miou':[], 'val_dice':[], 'lr':[]}
    best_miou = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_miou, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_miou, val_dice = val_epoch(model, val_loader, criterion, device)
        scheduler.step(val_miou)
        
        print(f"Train - Loss: {train_loss:.4f}, mIoU: {train_miou:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, Dice: {val_dice:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        for k,v in zip(['train_loss','train_miou','train_dice','val_loss','val_miou','val_dice'], [train_loss,train_miou,train_dice,val_loss,val_miou,val_dice]):
            history[k].append(float(v))
        history['lr'].append(float(optimizer.param_groups[0]['lr']))
        
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), save_path / f"best_{model_name}.pth")
            torch.save(model.state_dict(), save_path / f"best_{model_name}_{encoder}_miou{val_miou:.4f}.pth")
            print(f"Saved best model mIoU {val_miou:.4f}")
    
    # Save history
    with open(save_path / f"history_{model_name}.json", 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), save_path / f"final_{model_name}.pth")
    plot_history(history, model_name, save_dir=str(reports_root))
    print(f"\nTraining done! Best mIoU: {best_miou:.4f}")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--model', type=str, default='unet', help='unet, deeplabv3plus, fpn')
    parser.add_argument('--encoder', type=str, default='resnet18')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--num-samples', type=int, default=80)
    args = parser.parse_args()
    
    cfg = None
    if args.config and load_config:
        cfg = load_config(args.config)
        print(f"Loaded config from {args.config}")
    
    train_model(
        model_name=args.model,
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_samples=args.num_samples,
        config=cfg
    )
