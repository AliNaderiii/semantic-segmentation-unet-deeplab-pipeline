"""Benchmark inference time and model size"""
import torch
import time
from pathlib import Path
from models import get_model, count_parameters
import numpy as np

def benchmark(model_name='unet', encoder='resnet18', img_size=256, num_runs=50):
    device = torch.device('cpu')
    model = get_model(model_name, num_classes=2, encoder=encoder)
    model = model.to(device)
    model.eval()
    
    params = count_parameters(model)
    print(f"{model_name} {encoder}: {params:.2f}M params")
    
    # Dummy input
    x = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(x)
        times.append((time.time() - start) * 1000)  # ms
    
    avg = np.mean(times)
    std = np.std(times)
    print(f"Inference @ {img_size}x{img_size}: {avg:.1f} ± {std:.1f} ms (CPU, {num_runs} runs)")
    return params, avg

if __name__ == "__main__":
    for model_name, encoder in [('unet','resnet18'), ('unet','resnet34'), ('deeplabv3plus','resnet50'), ('fpn','resnet34')]:
        benchmark(model_name, encoder, img_size=128, num_runs=20)
