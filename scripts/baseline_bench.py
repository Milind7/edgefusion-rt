#!/usr/bin/env python3
import argparse, time, torch, torchvision
from rich import print

# ─── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Measure ViT-B-16 FP32 latency")
parser.add_argument("--iters", type=int, default=50,
                    help="Number of forward passes to average")
parser.add_argument("--device", choices=["auto","cpu","mps","cuda"],
                    default="auto",
                    help="Force a device (CI will use cpu)")
args = parser.parse_args()

# ─── Device selection ──────────────────────────────────────────────────────────
if args.device == "auto":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device(args.device)

print(f"[blue]Using device:[/] {device}")

# ─── Model & dummy input ───────────────────────────────────────────────────────
model = torchvision.models.vit_b_16(weights="DEFAULT").to(device).eval()
dummy = torch.randn(1, 3, 224, 224, device=device)

# ─── Warm-up & Synchronize ─────────────────────────────────────────────────────
warmups = min(20, args.iters)
for _ in range(warmups):
    _ = model(dummy)
if device.type == "cuda":
    torch.cuda.synchronize()
elif device.type == "mps":
    torch.mps.synchronize()

# ─── Timing loop ───────────────────────────────────────────────────────────────
start = time.perf_counter()
with torch.inference_mode():
    for _ in range(args.iters):
        _ = model(dummy)
if device.type == "cuda":
    torch.cuda.synchronize()
elif device.type == "mps":
    torch.mps.synchronize()
end = time.perf_counter()

lat_ms = (end - start) / args.iters * 1e3
print(f"[bold green]FP32 latency:[/] {lat_ms:.2f} ms on {device.type}")
