"""Benchmark a model architecture without silently downloading pretrained weights."""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from .models import count_parameters, get_model


def benchmark(model_name: str, encoder: str, image_size: int, runs: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, num_classes=2, encoder=encoder, pretrained=False).to(device).eval()
    input_tensor = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        for _ in range(10):
            model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    print(
        f"{model_name}/{encoder}: {count_parameters(model):.2f}M parameters; "
        f"{statistics.mean(times):.2f} ± {statistics.stdev(times):.2f} ms on {device}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unet")
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    benchmark(args.model, args.encoder, args.image_size, args.runs)
