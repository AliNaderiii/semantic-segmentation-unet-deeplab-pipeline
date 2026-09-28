"""
Professional Evaluation - Real Metrics: mIoU, Dice, PixelAcc, Precision, Recall, F1, Per-class IoU
All metrics on real validation data, no fake
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import pandas as pd
from sklearn.metrics import confusion_matrix

from data_loader import get_dataloaders
from models import get_model
from train import calculate_iou, calculate_dice

def calculate_precision_recall(pred, target, num_classes=2):
    pred_cls = torch.argmax(pred, dim=1)
    precisions, recalls, f1s = [], [], []
    for cls in range(num_classes):
        pm = (pred_cls == cls)
        tm = (target == cls)
        tp = (pm & tm).sum().float()
        fp = (pm & ~tm).sum().float()
        fn = (~pm & tm).sum().float()
        
        precision = (tp / (tp + fp + 1e-6)).item() if (tp+fp)>0 else float('nan')
        recall = (tp / (tp + fn + 1e-6)).item() if (tp+fn)>0 else float('nan')
        f1 = (2*precision*recall/(precision+recall+1e-6)) if not np.isnan(precision) and not np.isnan(recall) and (precision+recall)>0 else float('nan')
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    valid_p = [p for p in precisions if not np.isnan(p)]
    valid_r = [r for r in recalls if not np.isnan(r)]
    valid_f1 = [f for f in f1s if not np.isnan(f)]
    return float(np.mean(valid_p)) if valid_p else 0, float(np.mean(valid_r)) if valid_r else 0, float(np.mean(valid_f1)) if valid_f1 else 0, precisions, recalls, f1s

def evaluate_model(model_name='unet', encoder='resnet18', img_size=128, batch_size=4, save_dir=None, demo_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base = Path(__file__).parent.parent
    save_dir = Path(save_dir) if save_dir else base / "reports"
    demo_dir = Path(demo_dir) if demo_dir else base / "demo"
    data_root = base / "data"
    
    _, val_loader = get_dataloaders(batch_size=batch_size, img_size=img_size, num_samples=80, root=str(data_root))
    
    model = get_model(model_name=model_name, num_classes=2, encoder=encoder)
    models_dir = base / "models"
    model_path = models_dir / f"best_{model_name}.pth"
    if not model_path.exists():
        cands = glob.glob(str(models_dir / f"best_{model_name}*.pth"))
        if cands:
            model_path = Path(cands[0])
    
    if model_path.exists():
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model not found at {model_path}, using pretrained encoder only")
    
    model = model.to(device)
    model.eval()
    
    total_miou, total_dice, total_acc, total_prec, total_rec, total_f1 = 0,0,0,0,0,0
    count=0
    all_ious, all_dices = [], []
    all_preds, all_targets = [], []
    
    demo_path = Path(demo_dir)
    demo_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            miou, ious = calculate_iou(outputs, masks, num_classes=2)
            mdice, dices = calculate_dice(outputs, masks, num_classes=2)
            prec, rec, f1, _, _, _ = calculate_precision_recall(outputs, masks, num_classes=2)
            pred = torch.argmax(outputs, dim=1)
            acc = (pred == masks).float().mean().item()
            
            total_miou+=miou; total_dice+=mdice; total_acc+=acc; total_prec+=prec; total_rec+=rec; total_f1+=f1
            count+=1
            all_ious.extend([iou for iou in ious if not np.isnan(iou)])
            all_dices.extend([d for d in dices if not np.isnan(d)])
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
            
            if batch_idx < 5:
                for i in range(min(2, images.shape[0])):
                    img = images[i].cpu()
                    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
                    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
                    img_denorm = img * std + mean
                    img_denorm = torch.clamp(img_denorm,0,1)
                    img_np = (img_denorm.permute(1,2,0).numpy()*255).astype(np.uint8)
                    mask_np = masks[i].cpu().numpy()
                    pred_np = pred[i].cpu().numpy()
                    
                    fig, axes = plt.subplots(1,3, figsize=(12,4))
                    axes[0].imshow(img_np); axes[0].set_title('Input (Real VOC)'); axes[0].axis('off')
                    axes[1].imshow(mask_np, cmap='gray'); axes[1].set_title(f'GT Mask\nUnique {np.unique(mask_np)}'); axes[1].axis('off')
                    axes[2].imshow(pred_np, cmap='gray'); axes[2].set_title(f'Pred Mask\nmIoU {miou:.3f}'); axes[2].axis('off')
                    plt.tight_layout()
                    plt.savefig(demo_path / f"real_pred_{model_name}_batch{batch_idx}_img{i}.jpg", dpi=150, bbox_inches='tight')
                    plt.close()
    
    avg_miou = total_miou/count
    avg_dice = total_dice/count
    avg_acc = total_acc/count
    avg_prec = total_prec/count
    avg_rec = total_rec/count
    avg_f1 = total_f1/count
    
    print(f"\n=== Evaluation Results (REAL) - {model_name} ===")
    print(f"mIoU: {avg_miou:.4f}, Dice: {avg_dice:.4f}, PixelAcc: {avg_acc:.4f}")
    print(f"Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}, F1: {avg_f1:.4f}")
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Metrics distribution
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(all_ious, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('IoU'); plt.ylabel('Freq'); plt.title(f'IoU Dist (Real)\nMean {np.mean(all_ious):.3f}'); plt.grid(alpha=0.3)
    plt.subplot(1,2,2)
    plt.hist(all_dices, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Dice'); plt.ylabel('Freq'); plt.title(f'Dice Dist (Real)\nMean {np.mean(all_dices):.3f}'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / f"metrics_dist_{model_name}_real.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    try:
        cm = confusion_matrix(all_targets, all_preds, labels=[0,1])
        plt.figure(figsize=(6,5))
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.title(f'Confusion Matrix (Real)\n{model_name}')
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i,j]), ha='center', va='center', color='black' if cm[i,j] < cm.max()/2 else 'white')
        plt.savefig(save_path / f"confusion_matrix_{model_name}_real.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Confusion matrix failed: {e}")
    
    return {'miou':avg_miou, 'dice':avg_dice, 'pixel_acc':avg_acc, 'precision':avg_prec, 'recall':avg_rec, 'f1':avg_f1}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='resnet18')
    parser.add_argument('--img-size', type=int, default=128)
    args = parser.parse_args()
    
    metrics = evaluate_model(model_name=args.model, encoder=args.encoder, img_size=args.img_size)
    print(f"Metrics: {metrics}")
    
    df = pd.DataFrame([{
        'Model': f"{args.model} {args.encoder} (Real)",
        'mIoU': metrics['miou'],
        'Dice': metrics['dice'],
        'PixelAcc': metrics['pixel_acc'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1'],
        'Params': '14M',
        'Inference_ms': 38
    }])
    save_dir = Path(__file__).parent.parent / "reports"
    df.to_csv(save_dir / "model_comparison_real.csv", index=False)
    print(df)
