# EdgeFusion-RT

A minimal benchmark to measure ViT-B-16 inference latency.

| CI                                                                                                                                   | FP32 PyTorch (FP32)       | FP32 ONNX-Runtime (FP32) |
|-------------------------------------------------------------------------------------------------------------------------------------|---------------------------|--------------------------|
| ![build-test](https://github.com/Milind7/edgefusion-rt/actions/workflows/ci.yml/badge.svg)                                         |                           |                          |
| **Apple MPS (Local)**                                                                                                               | **15.20 ms**              | **TBD**                  |
| **Google Colab GPU (NVIDIA A100-SXM4-40GB)**                                                                                                     | **5.13 ms**              | **TBD**                  |



## Usage

```bash
python3 scripts/baseline_bench.py
