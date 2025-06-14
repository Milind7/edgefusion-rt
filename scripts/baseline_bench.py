import time, torch, torchvision
from rich import print

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = torchvision.models.vit_b_16(weights="DEFAULT").to(device).eval()
dummy = torch.randn(1, 3, 224, 224, device=device)

# Warm-up
for _ in range(5):
    _ = model(dummy)

iters, start = 10, time.perf_counter()
with torch.inference_mode():
    for _ in range(iters):
        _ = model(dummy)
end = time.perf_counter()

lat_ms = (end - start) / iters * 1e3
print(f"[bold green]FP32 latency:[/bold green] {lat_ms:.2f} ms on {device}")
