# EdgeFusion-RT

A minimal benchmark to measure ViT-B-16 inference latency.

| CI | Baseline Latency (ViT-B-16, FP32) |
|----|-----------------------------------|
| ![CI](https://github.com/<your-user>/edgefusion-rt/actions/workflows/ci.yml/badge.svg) | **26.24** ms on MPS |

## Usage

```bash
python3 scripts/baseline_bench.py
