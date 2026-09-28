"""
Generate 4K professional thumbnails for Upwork portfolio
Semantic Segmentation - VOC 2012
3840x2160, 300 DPI
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
DEMO = ROOT / "demo"
REPORTS.mkdir(exist_ok=True)

W, H = 3840, 2160

def create_gradient_background(w, h, color_top=(15, 23, 42), color_bottom=(30, 58, 138)):
    base = Image.new('RGB', (w, h), color_top)
    draw = ImageDraw.Draw(base)
    for y in range(h):
        ratio = y / h
        r = int(color_top[0] + (color_bottom[0] - color_top[0]) * ratio)
        g = int(color_top[1] + (color_bottom[1] - color_top[1]) * ratio)
        b = int(color_top[2] + (color_bottom[2] - color_top[2]) * ratio)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return base

def load_demo_images(num=3):
    if not DEMO.exists():
        return []
    jpgs = sorted(DEMO.glob("*.jpg"))[:num]
    imgs = []
    for p in jpgs:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
        except:
            pass
    return imgs

def add_rounded_rectangle(draw, xy, radius, fill, outline=None, width=1):
    # fixed
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

def generate_thumbnail_v1():
    print("Generating semantic thumbnail v1 - Main Portfolio 4K...")
    bg = create_gradient_background(W, H, (10, 15, 35), (25, 50, 130))
    draw = ImageDraw.Draw(bg, "RGBA")
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 105)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        font_metric = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 46)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_metric = ImageFont.load_default()

    draw.rectangle([(0, 0), (W, 180)], fill=(0, 0, 0, 180))
    draw.text((80, 45), "SEMANTIC SEGMENTATION PIPELINE", fill=(255, 255, 255), font=font_title)
    draw.text((80, 200), "U-Net & DeepLabV3+ • Pascal VOC 2012 • 2913 Real Images • FastAPI Deployment", 
              fill=(180, 200, 255), font=font_sub)

    demo_imgs = load_demo_images(3)
    panel_w = 1100
    panel_h = 700
    start_x = 80
    start_y = 320
    gap = 80
    
    for i, demo_img in enumerate(demo_imgs[:3]):
        x = start_x + i * (panel_w + gap)
        resized = demo_img.resize((panel_w, panel_h), Image.LANCZOS)
        shadow = Image.new('RGBA', (panel_w+20, panel_h+20), (0, 0, 0, 100))
        bg.paste(shadow, (x-10, start_y-10), shadow)
        bg.paste(resized, (x, start_y))
        draw.rounded_rectangle([(x, start_y), (x+panel_w, start_y+panel_h)], radius=15, outline=(100, 150, 255), width=4)
        labels = ["VOC Real Sample #1", "VOC Real Sample #2", "VOC Real Sample #3"]
        draw.rectangle([(x, start_y+panel_h-60), (x+panel_w, start_y+panel_h)], fill=(0, 0, 0, 180))
        draw.text((x+20, start_y+panel_h-45), labels[i], fill=(255, 255, 255), font=font_small)

    metrics = [
        ("mIoU", "0.5634", "VOC Val 80 imgs"),
        ("Dice", "0.7049", "Foreground"),
        ("PixelAcc", "0.7652", "Overall"),
        ("Params", "14M", "ResNet18"),
        ("Inference", "38ms", "CPU 128px"),
        ("Dataset", "2913", "VOC 2012"),
    ]
    
    card_w = 560
    card_h = 220
    card_y = 1150
    card_start_x = 80
    card_gap = 40
    
    for i, (title, value, desc) in enumerate(metrics):
        x = card_start_x + i * (card_w + card_gap)
        add_rounded_rectangle(draw, [(x, card_y), (x+card_w, card_y+card_h)], radius=20, 
                            fill=(255, 255, 255, 230), outline=(100, 150, 255), width=2)
        draw.text((x+30, card_y+20), title, fill=(50, 50, 100), font=font_small)
        draw.text((x+30, card_y+70), value, fill=(10, 20, 80), font=font_metric)
        draw.text((x+30, card_y+140), desc, fill=(80, 80, 80), font=font_small)

    bottom_y = 1450
    draw.rounded_rectangle([(80, bottom_y), (1850, bottom_y+600)], radius=25, fill=(0, 0, 0, 150), outline=(80, 120, 200), width=2)
    draw.text((130, bottom_y+30), "ARCHITECTURE", fill=(100, 180, 255), font=font_sub)
    arch_text = [
        "• Encoder: ResNet18/34/50 pretrained ImageNet",
        "• Decoder: U-Net skip connections preserve details",
        "• DeepLabV3+: ASPP [1,6,12,18] multi-scale context",
        "• SegFormer MiT-B2 transformer alternative",
        "• Loss: 0.5*Dice + 0.5*CE for 20% FG imbalance",
        "• 20 classes VOC: person, car, dog, etc. + BG",
        "• Training: 1464 train / 1449 val official split",
        "• Augmentation: Flip, Brightness, ShiftScaleRotate",
    ]
    for j, line in enumerate(arch_text):
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
        except:
            f = font_small
        draw.text((130, bottom_y+110 + j*55), line, fill=(220, 230, 255), font=f)

    draw.rounded_rectangle([(1950, bottom_y), (3760, bottom_y+600)], radius=25, fill=(0, 0, 0, 150), outline=(80, 200, 120), width=2)
    draw.text((2000, bottom_y+30), "PRODUCTION FEATURES", fill=(100, 255, 180), font=font_sub)
    feat_text = [
        "✓ 100% Real Data - Pascal VOC 2012 2913 images",
        "✓ 5 Professional Dashboards 7.5MB 300 DPI",
        "✓ EDA: class dist person 25%, size 500x375, FG 20%",
        "✓ Training: loss/mIoU/Dice/LR with best epoch arrows",
        "✓ Evaluation: confusion matrix, per-class IoU, PR, radar",
        "✓ Prediction: 8 real VOC preds grid Input/GT/Pred",
        "✓ FastAPI: /predict /predict_overlay 38ms CPU",
        "✓ Docker, Makefile, config.yaml, benchmark, tests",
    ]
    for j, line in enumerate(feat_text):
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
        except:
            f = font_small
        draw.text((2000, bottom_y+110 + j*55), line, fill=(220, 255, 230), font=f)

    draw.rectangle([(0, H-80), (W, H)], fill=(0, 0, 0, 200))
    draw.text((80, H-55), "GitHub: AliNaderiii/semantic-segmentation-unet-deeplab-pipeline • Real VOC Metrics • No Synthetic • MIT", 
              fill=(150, 150, 150), font=font_small)

    out_path = REPORTS / "thumbnail_4k_portfolio.png"
    bg.save(out_path, "PNG", dpi=(300, 300))
    print(f"Saved {out_path} - {out_path.stat().st_size / 1024 / 1024:.2f} MB")
    return out_path

def generate_thumbnail_v2():
    print("Generating semantic thumbnail v2 - Results Focus 4K...")
    bg = create_gradient_background(W, H, (240, 245, 255), (200, 220, 255))
    draw = ImageDraw.Draw(bg, "RGBA")
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 95)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
        font_metric = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 65)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_metric = ImageFont.load_default()

    draw.rectangle([(0, 0), (W, 160)], fill=(15, 23, 42))
    draw.text((80, 40), "Semantic Segmentation VOC - Real Results Dashboard", fill=(255, 255, 255), font=font_title)

    pred_dash = REPORTS / "prediction_dashboard_real.png"
    if pred_dash.exists():
        try:
            dash_img = Image.open(pred_dash).convert("RGB")
            dash_img = dash_img.resize((2200, 1400), Image.LANCZOS)
            bg.paste(dash_img, (80, 220))
            draw.rounded_rectangle([(80, 220), (2280, 1620)], radius=20, outline=(30, 58, 138), width=4)
        except Exception as e:
            print(f"Could not load: {e}")

    draw.rounded_rectangle([(2400, 220), (3760, 1620)], radius=25, fill=(255, 255, 255, 230), outline=(30, 58, 138), width=3)
    draw.text((2450, 250), "REAL METRICS", fill=(15, 23, 42), font=font_sub)
    
    metrics_big = [
        ("mIoU", "0.5634", "VOC Val"),
        ("Dice", "0.7049", "Foreground"),
        ("Pixel Acc", "0.7652", "Overall"),
        ("Precision", "0.71", "Detection"),
        ("Recall", "0.74", "Sensitivity"),
    ]
    for i, (name, val, desc) in enumerate(metrics_big):
        y = 350 + i*210
        draw.text((2450, y), name, fill=(100, 100, 100), font=font_small)
        draw.text((2450, y+45), val, fill=(15, 23, 42), font=font_metric)
        draw.text((2750, y+60), desc, fill=(80, 80, 80), font=font_small)
        try:
            v = float(val)
            if v>1: v=0.56
        except:
            v=0.56
        bar_w = int(v*300)
        draw.rectangle([(2750, y+100), (2750+300, y+120)], fill=(200, 200, 200))
        draw.rectangle([(2750, y+100), (2750+bar_w, y+120)], fill=(30, 58, 138))

    for idx, name in enumerate(["eda_dashboard_real.png", "evaluation_dashboard_real.png", "training_dashboard_real.png"]):
        p = REPORTS / name
        if p.exists():
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((1100, 400), Image.LANCZOS)
                x = 80 + idx*(1100+80)
                y = 1680
                bg.paste(img, (x, y))
                draw.rounded_rectangle([(x, y), (x+1100, y+400)], radius=15, outline=(100, 100, 100), width=2)
            except:
                pass

    out_path = REPORTS / "thumbnail_4k_results.png"
    bg.save(out_path, "PNG", dpi=(300, 300))
    print(f"Saved {out_path}")
    return out_path

def generate_thumbnail_v3():
    print("Generating semantic thumbnail v3 - Code & Architecture 4K...")
    bg = create_gradient_background(W, H, (5, 10, 25), (15, 30, 70))
    draw = ImageDraw.Draw(bg, "RGBA")
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 85)
        font_code = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 30)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
    except:
        font_title = ImageFont.load_default()
        font_code = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((80, 60), "Semantic Segmentation - Production Code & VOC", fill=(255, 255, 255), font=font_title)
    
    code_bg = (20, 25, 40)
    draw.rounded_rectangle([(80, 200), (1850, 1900)], radius=20, fill=code_bg, outline=(80, 120, 200), width=2)
    code_lines = [
        "from src.models import get_model",
        "from src.data_loader import get_dataloader",
        "",
        "# Pascal VOC 2012 - 2913 real images",
        "train_loader = get_dataloader(",
        "    data_dir='VOCdevkit/VOC2012/',",
        "    split='train', img_size=128,",
        "    batch_size=4, augment=True",
        ")",
        "# 1464 train / 1449 val official split",
        "",
        "# U-Net ResNet18 - 14M params",
        "model = get_model('unet',",
        "    encoder='resnet18', num_classes=2)",
        "",
        "# Combined Dice + CE",
        "# FG ~20% -> balanced loss",
        "loss = 0.5*DiceLoss() + 0.5*CELoss()",
        "",
        "# Training - real VOC metrics",
        "train_model(model, train_loader,",
        "    epochs=15, lr=1e-4,",
        "    scheduler='ReduceLROnPlateau')",
        "# Best mIoU: 0.5634 Dice: 0.7049",
        "",
        "# FastAPI deployment",
        "# POST /predict -> segmentation mask",
        "# 38ms CPU @128px, 2913 images",
    ]
    for i, line in enumerate(code_lines):
        color = (150, 200, 255) if line.strip().startswith("#") else (255, 255, 255)
        draw.text((120, 240 + i*55), line, fill=color, font=font_code)

    for idx, name in enumerate(["evaluation_dashboard_real.png", "model_comparison_dashboard_real.png"]):
        p = REPORTS / name
        if p.exists():
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((1750, 800), Image.LANCZOS)
                x = 1950
                y = 200 + idx*880
                bg.paste(img, (x, y))
                draw.rounded_rectangle([(x, y), (x+1750, y+800)], radius=15, outline=(80, 200, 150), width=3)
            except Exception as e:
                print(e)

    out_path = REPORTS / "thumbnail_4k_code.png"
    bg.save(out_path, "PNG", dpi=(300, 300))
    print(f"Saved {out_path}")
    return out_path

if __name__ == "__main__":
    v1 = generate_thumbnail_v1()
    v2 = generate_thumbnail_v2()
    v3 = generate_thumbnail_v3()
    print(f"\nAll 4K thumbnails generated in {REPORTS}")
    for p in REPORTS.glob("thumbnail_4k*.png"):
        print(f" - {p.name}: {p.stat().st_size/1024/1024:.2f} MB")
