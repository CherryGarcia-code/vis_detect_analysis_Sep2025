"""Fail-fast GPU/torch check for DeepUnitMatch training (plan Task 0).

Run on a cluster A100 node under the ceph `unitmatch` env before launching training.
Locally it prints CPU torch (expected); on the cluster it must print cuda_available True.
"""
import torch

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
