"""Shared helpers for DeepUnitMatch fine-tuning (tracked; vendor stays read-only)."""
from __future__ import annotations
import random
import numpy as np


class AverageMeter:
    """Running average (matches upstream DeepUnitMatch utils.metric.AverageMeter)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def augment_channel_roll(data, choice=None, rng=random):
    """Port of upstream NeuropixelsDataset._augment_original (non-mutating).

    data: array [T, C]. Randomly shifts odd/even channel columns up or down by one
    electrode row (simulates a single-row probe drift), or leaves it unchanged.
    """
    data = np.asarray(data).copy()
    if choice is None:
        choice = rng.choice(["roll_up", "roll_down", "none"])
    C = data.shape[1]
    if choice == "roll_up":
        odd = np.arange(0, C - 1, 2)
        even = np.arange(1, C - 1, 2)
        if len(odd) > 1:
            data[:, odd[:-1]] = data[:, odd[1:]]
        if len(even) > 1:
            data[:, even[:-1]] = data[:, even[1:]]
    elif choice == "roll_down":
        odd = np.arange(2, C, 2)
        even = np.arange(3, C, 2)
        if len(odd) > 0:
            data[:, odd] = data[:, odd - 2]
        if len(even) > 0:
            data[:, even] = data[:, even - 2]
    return data
