import argparse

parser = argparse.ArgumentParser(description="Measure ViT-B-16 FP32 latency")
parser.add_argument(
    "--iters", type=int, default=50,
    help="Number of forward passes to average (use 1 in CI smoke-test)"
)
args = parser.parse_args()
iters = args.iters

import time, torch, torchvision
from rich import print



# 1) Pick the best device
if   torch.cuda.is_available():
    device = torch.device("cuda")
    backend = "CUDA"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    backend = "MPS"
else:
    device = torch.device("cpu")
    backend = "CPU"

# 2) Load model & dummy input
model = torchvision.models.vit_b_16(weights="DEFAULT").to(device).eval()
dummy = torch.randn(1, 3, 224, 224, device=device)

# 3) Warm-up
warmups = 20
for _ in range(warmups):
    _ = model(dummy)
if backend == "CUDA": torch.cuda.synchronize()
if backend == "MPS":  torch.mps.synchronize()

# 4) Timing loop
iters = 50
start = time.perf_counter()
with torch.inference_mode():
    for _ in range(iters):
        _ = model(dummy)
if backend == "CUDA": torch.cuda.synchronize()
if backend == "MPS":  torch.mps.synchronize()
end = time.perf_counter()

lat_ms = (end - start) / iters * 1e3
print(f"[bold green]FP32 latency:[/] {lat_ms:.2f} ms on {backend}")
