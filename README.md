# EdgeFusion-RT

A minimal benchmark to measure ViT-B-16 inference latency.

| CI | Baseline Latency (ViT-B-16, FP32)    |
|----|--------------------------------------|
| ![build-test](https://github.com/Milind7/edgefusion-rt/actions/workflows/ci.yml/badge.svg) | **14.83 ms on MPS** |


## Usage

```bash
python3 scripts/baseline_bench.py
