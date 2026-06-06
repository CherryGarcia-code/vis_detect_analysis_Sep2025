#!/usr/bin/env python3
"""Stage-2 CLIP fine-tune of the DeepUnitMatch encoder on BG_046.

Self-supervised: positives = (aug CV-half-1, aug CV-half-2) of the same unit;
negatives = other units in the same-session batch. Loss = CLIP on projector outputs.
Inference later uses the 256-dim encoder via clip_sim, so fine-tuning FcBlock reshapes
the embedding space tracking consumes.

Run under unitmatch_env:
  conda run -n unitmatch_env --no-capture-output python train_deepum_clip.py \
    --train-root <processed_waveforms> --out-dir <out> --init shipped --freeze fcblock
"""
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_deepum_common import (add_deepum_to_path, shipped_checkpoint_path,
                                 patch_augmentation, build_export_checkpoint,
                                 within_session_top1_accuracy, AverageMeter)


def build_model(device):
    add_deepum_to_path()
    from DeepUnitMatch.utils.mymodel import SpatioTemporalCNN_V2
    from DeepUnitMatch.utils.losses import CustomClipLoss, Projector
    model = SpatioTemporalCNN_V2(n_channel=30, n_time=60, n_output=256).to(device).double()
    clip_loss = CustomClipLoss().to(device)
    projector = Projector(256, 128, 128, 1, 0.1).to(device).double()
    return model, clip_loss, projector


def init_weights(model, clip_loss, init, device):
    if init == "shipped":
        ck = torch.load(shipped_checkpoint_path(), map_location=device)
        model.load_state_dict(ck["model"])
        if "clip_loss" in ck:
            clip_loss.load_state_dict(ck["clip_loss"])
        print(f"  init: shipped checkpoint", flush=True)
    else:
        ck = torch.load(init, map_location=device)
        model.load_state_dict(ck["encoder"])
        print(f"  init: AE encoder from {init}", flush=True)


def apply_freeze(model, freeze):
    if freeze == "fcblock":
        for name, p in model.named_parameters():
            p.requires_grad = "FcBlock" in name
    elif freeze == "none":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown freeze policy {freeze}")
    n_train = sum(p.requires_grad for p in model.parameters())
    print(f"  freeze={freeze}: {n_train} encoder params trainable", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init", default="shipped", help="'shipped' or path to AE ckpt")
    ap.add_argument("--freeze", default="fcblock", choices=["fcblock", "none"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--lr-proj", type=float, default=1.1e-4)
    ap.add_argument("--save-freq", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    add_deepum_to_path()
    from DeepUnitMatch.utils.npdataset import (NeuropixelsDataset,
                                               TrainExperimentBatchSampler,
                                               ValidationExperimentBatchSampler)
    patch_augmentation(NeuropixelsDataset)

    model, clip_loss, projector = build_model(device)
    init_weights(model, clip_loss, args.init, device)
    apply_freeze(model, args.freeze)

    enc_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam([
        {"params": enc_params, "lr": args.lr_enc},
        {"params": list(projector.parameters()) + list(clip_loss.parameters()),
         "lr": args.lr_proj},
    ])

    train_ds = NeuropixelsDataset(args.train_root, batch_size=args.batch, mode="train")
    train_loader = DataLoader(
        train_ds, batch_sampler=TrainExperimentBatchSampler(train_ds, args.batch, shuffle=True))
    val_ds = NeuropixelsDataset(args.train_root, batch_size=args.batch, mode="val")
    val_loader = DataLoader(
        val_ds, batch_sampler=ValidationExperimentBatchSampler(val_ds, shuffle=False))

    proxy_log = []
    for epoch in range(args.epochs):
        model.train(); clip_loss.train()
        losses = AverageMeter()
        for estimates, candidates, *_ in train_loader:
            estimates = estimates.to(device).double()
            candidates = candidates.to(device).double()
            optimizer.zero_grad()
            loss = clip_loss(projector(model(estimates)), projector(model(candidates)))
            loss.backward(); optimizer.step()
            losses.update(loss.item(), estimates.shape[0])
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "clip_loss": clip_loss.state_dict(), "epoch": epoch},
                       os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pt"))
            torch.save(build_export_checkpoint(model, clip_loss, projector),
                       os.path.join(args.out_dir, f"export_epoch_{epoch}.pt"))
            acc = within_session_top1_accuracy(model, val_loader, device)
            proxy_log.append({"epoch": epoch, "train_loss": losses.avg, "val_top1": acc})
            print(f"epoch {epoch}: train_loss={losses.avg:.6f} val_top1={acc:.4f}", flush=True)
    with open(os.path.join(args.out_dir, "proxy_log.json"), "w") as f:
        json.dump({"init": args.init, "freeze": args.freeze, "log": proxy_log}, f, indent=2)


if __name__ == "__main__":
    main()
