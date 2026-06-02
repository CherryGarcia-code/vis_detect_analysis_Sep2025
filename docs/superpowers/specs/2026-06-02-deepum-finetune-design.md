# DeepUnitMatch fine-tuning — design spec

**Date:** 2026-06-02
**Branch:** `feature/deepum-finetune` (worktree off local `main` @ 7f438d4)
**Author context:** continues `memory/neuron_tracking_may2026.md` ("Fine-tuning DeepUM — starting points")

---

## 1. Goal

Fine-tune the DeepUnitMatch (DeepUM) waveform encoder on BG_046's striatal Neuropixels
2.0 data so that cross-session neuron tracking beats **stock DeepUM** and ideally
approaches/beats the classical **UnitMatch (UM 3.2.9)** baseline.

Headline metric: percentage of tracked global IDs spanning **≥2 sessions** in the
`cell_registry.csv` produced by the existing tracking pipeline, on the same 42 BG_046
sessions.

| Method | ≥2 sess | ≥5 | ≥10 | ≥15 | ≥20 | max span | Naive∩Expert |
|---|---|---|---|---|---|---|---|
| UM 3.2.9 (target) | 19.8% | 4.9% | 1.6% | 0.9% | 0.5% | 28/42 | 19 (14 hi-conf) |
| DeepUM stock (beat) | 6.3% | 0.4% | 0.03% | 0 | 0 | 14/42 | 32 (unvalidated) |

- **Must:** clearly beat stock DeepUM's 6.3% ≥2-sess.
- **Target:** approach/beat UM's 19.8% ≥2-sess and recover long tracks (≥20-sess > 0).

This is positioned as a **scalable per-subject tracker** for future subjects; UM remains
primary for current science until/unless DeepUM demonstrably wins.

---

## 2. Background: what exists, what's missing

The vendored DeepUM source lives at
`_DeepUnitMatch_repo/UnitMatchPy/DeepUnitMatch/` (a copy of `EnnyvanBeest/UnitMatch@main`;
**gitignored** in the parent repo — see §7).

**Present and reusable** (in `utils/`):
- `mymodel.py` — `SpatioTemporalCNN_V2` encoder (n_channel=30, n_time=60, n_output=256),
  `SpatioTemporalAutoEncoder_V2` (encoder+decoder), `Decoder_SpatioTemporalCNN_V2`.
- `losses.py` — `CustomClipLoss` (CLIP contrastive, learnable `temp_tau`), `AELoss`
  (λ₁·L1 + λ₂·L2), `Projector`, `clip_sim`, `clip_prob`.
- `npdataset.py` — `NeuropixelsDataset` (loads `(60,30,2)` HDF5 snippets, returns the two
  CV halves), `TrainExperimentBatchSampler` (one batch per session, resamples to fill),
  `ValidationExperimentBatchSampler`.
- `testing/test.py` — `load_trained_model` (loads shipped checkpoint, builds encoder),
  `inference` (similarity matrix via `clip_sim` on **256-dim encoder** outputs),
  `get_matches`, drift-corrected distance filtering.
- Shipped pre-trained checkpoint: `utils/model` (single file; `load_trained_model` reads
  keys `model` + `clip_loss` only — does **not** read a `projector` key).

**Missing from the vendored copy** (present upstream, must be ported):
- `utils/npdataset.py::_augment_original` — called in `mode='train'` but **undefined**.
  Upstream: random channel **roll up / roll down by one electrode row** (simulates a
  single-row probe shift) or `none`. Verbatim port.
- `utils/metric.py` — upstream training imports `from utils import metric` (`AverageMeter`).
  Not present. Port `AverageMeter`.
- `train/train_AE.py`, `train/train_finetune.py` — the training loops. **Not present.**
  We do **not** copy these verbatim; we reuse their model/loss/loop logic in our own
  cleaner entry scripts (upstream hard-codes a `ModelExp/...` layout, spawns TensorBoard
  subprocesses, and kills `tensorboard.exe` via `psutil` — a Windows-ism we drop).

**Training data (already on cluster):** HDF5 cache written by job 3026330 at
`/ceph/.../UnitMatchPy/DeepUnitMatch/processed_waveforms/` — 6679 units across 42
sessions, one dir per session, each `Unit{i}.npy` an HDF5 with `waveform (60,30,2)` +
`MaxSitepos`. The two CV halves are `waveform[...,0]` and `waveform[...,1]`. **No
`get_snippets` re-run needed** — point training at this dir.

---

## 3. Method recap (self-supervised, no cross-session labels)

Each unit's snippet has two cross-validation (CV) halves = two waveform views of the same
neuron from the first vs second half of one recording. The encoder maps a `(60,30)` view
to a 256-dim embedding. Training objective:

- **Positive pair:** (augmented CV-half-1, augmented CV-half-2) of the **same** unit.
- **Negatives:** all **other units in the same session batch**.
- **Loss:** `CustomClipLoss` (CLIP contrastive) computed on **projector** outputs
  (128-dim); a learnable temperature scales the logits.
- At **inference** the projector is discarded; tracking similarity = `clip_sim` on the
  **256-dim encoder** output. Therefore fine-tuning the encoder's final block (`FcBlock`)
  is what re-shapes the embedding space that tracking actually consumes.

No cross-session match labels are ever used, so evaluating cross-session tracking on the
same 42 sessions is **not label leakage** (the model cannot memorise cross-day identity).
The only residual risk is overfitting to the subject's general waveform statistics — which
is exactly what a per-subject tracker should do. We therefore train on all 42 sessions and
report in-sample tracking (the deployment use case is tracking these exact units).

Two stages (author pipeline):
1. **AE pretrain** (`SpatioTemporalAutoEncoder_V2`, `AELoss(λ1=0, λ2=1.0)` = pure MSE
   reconstruction) → produces an `encoder` checkpoint. Teaches general waveform structure.
2. **CLIP fine-tune** (`train_finetune` logic) → loads an encoder, freezes per policy,
   trains the contrastive objective.

---

## 4. Experiment matrix — three rungs, run all up front

Two independent knobs:
- **Init** (where weights start): shipped cortex-lab model (**warm-start**) vs.
  AE-pretrained-on-BG_046 (**from-scratch**).
- **Freeze** (what may move): only the encoder `FcBlock` (`--freeze fcblock`) vs. the
  whole encoder (`--freeze none`).

| Rung | Init | Freeze | Question it answers |
|---|---|---|---|
| **1** (primary deliverable) | shipped (warm-start) | FcBlock only | Does a light final-layer re-tune of the released model help? |
| **2** | shipped (warm-start) | unfreeze all | Does striatum need *deeper* adaptation than a final-layer tweak? |
| **3** (control) | AE-on-BG_046 (scratch) | FcBlock only | Did the cortex-lab pretraining help at all vs. BG_046 alone? |

All three are trained and tracking-evaluated up front; the comparison matrix is the
deliverable. Rung 3 requires the AE-pretrain step first; rungs 1–2 do not.

**Hyperparameters** (author defaults, exposed as CLI args):
`lr_enc=2e-5` (encoder FcBlock), `lr_proj=1.1e-4` (projector + temperature), Adam,
`batch=40`, `epochs=50`, `save_freq=5`. Rung 2 uses `lr_enc` for the whole encoder
(may revisit if it diverges). AE pretrain: Adam `lr=1e-5`, reconstruction MSE.

---

## 5. Components — files added / modified

### 5.1 Port into the vendored repo (gitignored; edits sync to cluster — see §7)
- `_DeepUnitMatch_repo/.../utils/npdataset.py` — add `_augment_original` to
  `NeuropixelsDataset` (verbatim upstream channel roll up/down/none).
- `_DeepUnitMatch_repo/.../utils/metric.py` — **new**; port `AverageMeter`.

### 5.2 New scripts we own — `scripts/pipelines/tracking/`
- `train_deepum_ae.py` — Stage-1 AE pretrain on BG_046 (rung 3 only). Saves per-epoch
  checkpoints `{'model','encoder','optimizer','epoch'}`; the `encoder` key feeds the CLIP
  stage.
- `train_deepum_clip.py` — Stage-2 CLIP fine-tune. Args:
  `--init {shipped|<ae_ckpt_path>}`, `--freeze {fcblock|none}`, `--train-root`,
  `--out-dir`, `--epochs`, `--batch`, `--lr-enc`, `--lr-proj`, `--save-freq`.
  - `--init shipped` loads encoder from `utils/model['model']` and temperature from
    `['clip_loss']`; fresh `Projector`.
  - `--init <ae_ckpt>` loads `checkpoint['encoder']`.
  - `--freeze fcblock` sets `requires_grad=False` for every encoder param whose name lacks
    `FcBlock`; `--freeze none` trains the whole encoder.
  - Per-epoch: save training-state ckpt **and** an inference-format export
    `export_epoch_{N}.pt` = `{'model','clip_loss','projector'}` (§6).
  - Per-epoch proxy validation: within-session top-1 matching accuracy + CLIP val loss
    on a held-back fraction of units (fast; **not** the headline metric).
- `eval_deepum_checkpoints.py` — for each export checkpoint, invoke the existing tracking
  pipeline with `--ckpt`, collect `run_summary.json`, and write one comparison table
  (`finetune_comparison.csv`) across {stock, UM ref, rung1/2/3 candidates}. To save A100
  time, evaluates a **subset** of epochs (default: every 10th + the proxy-best), not all 50.
- `slurm/train_deepum.sbatch` — A100 submission (§7).

### 5.3 One backward-compatible edit to the vendored runner / inference
- `_DeepUnitMatch_repo/.../testing/test.py::load_trained_model(device, ckpt_path=None)` —
  optional path; default unchanged (shipped `utils/model`).
- `scripts/pipelines/tracking/run_deepunitmatch_all.py` — add `--ckpt <path>` threaded into
  `load_trained_model(device, ckpt_path=args.ckpt)`. Default `None` reproduces the stock
  run exactly. (This file is currently **untracked** — see §7.)

Nothing existing is deleted or behavior-changed when the new flags are omitted.

---

## 6. Checkpoint reconciliation

- **Training-state** (resumable): `{'model','optimizer','clip_loss','epoch'}`.
- **Inference-format** (what `load_trained_model` reads): needs `model` + `clip_loss`;
  we also include `projector` for completeness → `{'model','clip_loss','projector'}`.
- `train_deepum_clip.py` writes both each `save_freq` epochs. `eval_deepum_checkpoints.py`
  and `run_deepunitmatch_all.py --ckpt` consume only the export files.

Selection rule: pick the export checkpoint with the best ≥2-sess (tie-break ≥5 then ≥10),
sanity-checked against the proxy-accuracy curve to avoid an obviously over-fit epoch.

---

## 7. Cluster execution & repo-state prerequisites

### 7.1 Repo-state reality (must be handled before implementation)
- `_DeepUnitMatch_repo/` is **gitignored** in the parent repo. Edits to it (port augmentation,
  add `metric.py`, `--ckpt` in `testing/test.py`) are **not** parent-repo git operations;
  they must be propagated to the cluster's ceph copy
  (`/ceph/.../UnitMatchPy/DeepUnitMatch/`) by the existing sync mechanism (rsync/scp or the
  cluster's own git for the vendored repo).
- `run_deepunitmatch_all.py` and `validate_long_tracks.py` are **untracked**, present only in
  the *main* workspace working tree (not on `main`, not in this worktree). To modify
  `run_deepunitmatch_all.py` on this branch we must first bring it into the worktree and
  `git add` it (first-time tracking). **Decision required from user** before doing so, since
  it originated in a parallel tracking-work chat (do not unilaterally commit another chat's
  WIP). Until then, the `--ckpt` edit can be applied to the cluster copy directly.

### 7.2 Prerequisite — GPU-enabled PyTorch (BLOCKING, verify first)
Prior DeepUM inference jobs ran **CPU** torch. Training (esp. AE pretrain + rung 3) wants
the A100. **Step 0:** on a GPU node, confirm `torch.cuda.is_available()` in the ceph
`unitmatch` env; if `False`, install a CUDA build before training. (Rung 1 FcBlock-only is
light enough to fall back to CPU if absolutely necessary.)

### 7.3 SLURM
- Partition `a100`, `--gres=gpu:1`, `--mem=100G`, `--time=06:00:00`, ceph-conda pattern
  `export PATH="${CONDA_ENV}/bin:${PATH}"` (mirrors KS4 retry + UM run scripts).
- Train job: optional AE (rung 3) → CLIP (rungs 1/2/3) → export checkpoints.
- Eval job (can be CPU, ~40 min like the inference job): `eval_deepum_checkpoints.py`.
- `--train-root /ceph/.../DeepUnitMatch/processed_waveforms`.
- Outputs → `data/unit_match/output/BG_046_finetune/` (checkpoints, proxy curves,
  `finetune_comparison.csv`, logs) — separate from stock `all42_deep/`.

---

## 8. Testing strategy (TDD-lite — plumbing, not science)

Local CPU guardrails before burning A100 time:
1. `_augment_original`: `roll_up`/`roll_down` preserve `(T,C)` shape and shift the expected
   channels; `none` is identity.
2. Dataset: a sample returns two `(60,30)` halves + a `MaxSitepos`.
3. **Integration smoke:** run `train_deepum_clip.py --init shipped --freeze fcblock` for 1
   epoch on a 2-session subset; assert it writes an `export_epoch_*.pt` that
   `load_trained_model(ckpt_path=...)` loads and that `inference` produces a finite
   similarity matrix. Catches checkpoint-format and freeze-policy bugs cheaply.
4. Checkpoint export round-trips through `load_trained_model`.

---

## 9. Success criteria

- **Must:** rung-1 (warm-start, FcBlock) ≥2-sess clearly > 6.3%.
- **Target:** any rung approaches/beats UM's 19.8% ≥2-sess; ≥20-sess tracks > 0.
- **Scientific read-outs:**
  - rung 1 vs rung 2 → does striatum need deeper adaptation than a final-layer tweak?
  - rung 1 vs rung 3 → how much did cortex-lab pretraining transfer to striatum?
- Cross-check winners against the ISI-fingerprint validation
  (`validate_long_tracks.py`, paper Fig 4 method) as a functional sanity check, as done for
  the UM cohort.

---

## 10. Scope / non-goals (YAGNI)

- **In scope:** BG_046 fine-tuning (3 rungs), tracking eval, comparison matrix.
- **Out of scope:** generalising the CLI to arbitrary new subjects (paths are args, but no
  multi-subject driver); hyperparameter sweeps beyond freeze policy + the given LRs; the
  concat-sort A100 triangulation (separate strategic option in the memory). Scripts must not
  hard-code BG_046, but multi-subject is explicitly deferred.

---

## 11. Open items / risks

1. **GPU torch** in the ceph `unitmatch` env — verify/Install (§7.2). Blocking.
2. **Untracked `run_deepunitmatch_all.py`** — confirm with user how it should be brought
   onto this branch vs. patched on the cluster directly (§7.1).
3. **Vendored-repo edit propagation** — confirm the exact sync path to ceph for the
   gitignored `_DeepUnitMatch_repo/` changes (§7.1).
4. **Rung-2 stability** — unfreezing the whole encoder at `lr_enc=2e-5` may need a smaller
   LR or fewer epochs; watch the proxy curve.
5. **In-sample reporting caveat** — documented as acceptable (§3); state it plainly in any
   write-up.
