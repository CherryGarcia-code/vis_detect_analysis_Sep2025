#!/usr/bin/env python3
"""Stage-1 autoencoder pretrain for DeepUnitMatch (from-scratch rung only).

Reconstructs each CV half with SpatioTemporalAutoEncoder_V2 (AELoss lambda1=0,
lambda2=1.0 = pure MSE). Saves checkpoints with an 'encoder' key that
train_deepum_clip.py --init <ckpt> loads into the SpatioTemporalCNN_V2 encoder.
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_deepum_common import add_deepum_to_path, AverageMeter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-freq", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    add_deepum_to_path()
    from DeepUnitMatch.utils.mymodel import SpatioTemporalAutoEncoder_V2
    from DeepUnitMatch.utils.losses import AELoss
    from DeepUnitMatch.utils.npdataset import (NeuropixelsDataset,
                                               TrainExperimentBatchSampler)

    ae = SpatioTemporalAutoEncoder_V2(n_channel=30, n_time=60, n_output=256).to(device).double()
    ae_loss = AELoss(lambda1=0.0, lambda2=1.0)
    optimizer = optim.Adam(ae.parameters(), lr=args.lr)

    ds = NeuropixelsDataset(args.train_root, batch_size=args.batch, mode="val")  # no aug for AE target
    loader = DataLoader(ds, batch_sampler=TrainExperimentBatchSampler(ds, args.batch, shuffle=True))

    log = []
    for epoch in range(args.epochs):
        ae.train()
        losses = AverageMeter()
        for estimates, candidates, *_ in loader:
            for x in (estimates, candidates):  # reconstruct both CV halves
                x = x.to(device).double()
                optimizer.zero_grad()
                loss = ae_loss(ae(x), x)
                loss.backward(); optimizer.step()
                losses.update(loss.item(), x.shape[0])
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            torch.save({"model": ae.state_dict(), "encoder": ae.encoder.state_dict(),
                        "optimizer": optimizer.state_dict(), "epoch": epoch},
                       os.path.join(args.out_dir, f"ae_epoch_{epoch}.pt"))
            log.append({"epoch": epoch, "recon_loss": losses.avg})
            print(f"AE epoch {epoch}: recon_loss={losses.avg:.6f}", flush=True)
    with open(os.path.join(args.out_dir, "ae_log.json"), "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
