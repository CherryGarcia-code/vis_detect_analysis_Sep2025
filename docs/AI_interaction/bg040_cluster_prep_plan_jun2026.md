# BG_040 UM-prep — run it cluster-side (not over the Samba mount)

**Date:** 2026-06-22
**Why:** the local TPrime→pkl→extract prep was run over the `X:` Samba gateway (`ceph-gw02`) and helped lock it up (150 MB/s → 120 kB/s; SWC-confirmed). KS4 + UnitMatch already run on HPC compute nodes (native CephFS) and were immune. This plan extends that **same cluster-native pattern** to the prep so we never touch the Samba gateway for compute again. See `[[feedback-no-compute-over-samba-gateway]]`.

---

## BG_040 current state (ready to prep)
- **KS4 16/16 complete.**
- **Behavior organized** — 69 JSONs placed into `Raw data/<session>/Session/` (0 truncation); `organize_subject_data.py BG_040 --execute` already run.
- **Manifest written:** `data/BG_040_um_extract_manifest.csv` (16 sessions, all 8-digit names).
- **Paused:** the local prep wedged on the degraded gateway; the partial session-1 adj was deleted, so a clean re-run rebuilds it.
- Session list (16): `02052025 09052025 10062025 12062025 13052025 15052025 16062025 18062025 19052025 21052025 24042025 27062025 28042025 28052025 30042025 30052025` (all prefixed `BG_040_`).

---

## One-time cluster setup (just like KS4 — clone + env + sbatch, NOT "build a repo")
The KS4 jobs ran from `~/Documents/ks4` with a conda env on ceph. Same pattern here, with **one** extra: KS4's runner only imported `kilosort` (a pip package), but the prep scripts `import visdetect`, so the repo **code** must be on the cluster.
1. **`git clone` the repo to the cluster** — `git clone git@github.com:CherryGarcia-code/vis_detect_analysis_Sep2025.git ~/vis_detect` (code only — data is gitignored, so small/fast). Gives `src/visdetect` + the prep scripts. *(This is the only thing KS4 didn't need; it's a clone, not a build.)*
2. **Reuse an existing conda env** (e.g. `/ceph/…/conda_envs/unitmatch`) — already has numpy/scipy/pandas; `pip install` any missing deps (`h5py`, `tqdm`). Make visdetect importable with **`export PYTHONPATH=~/vis_detect/src`** in the sbatch — **no editable install needed.** (Same trick as the worktree PYTHONPATH note.)
3. **Linux TPrime binary** ⚠️ — `run_tprime.py` defaults to the Windows `G:\…\TPrime\TPrime.exe`. Need SpikeGLX's **Linux** CatGT/TPrime build on ceph, passed via `--tprime-exe`. **VERIFY whether the lab already has a Linux TPrime on ceph; if not, download the SpikeGLX Linux release.**
4. **Manifest paths** — regenerate `BG_040_um_extract_manifest.csv` with the ceph pkl out-dir (or just pass `--out-dir <ceph>` at run time).

---

## The prep Slurm job (sketch) — `run_prep_subject.sbatch <SUBJECT>`
Run on a `cpu` partition compute node (TPrime/ingest are CPU/IO, not GPU). All paths are **`/ceph/mrsic_flogel/...`** (native CephFS — no Samba):
```
conda activate /ceph/.../conda_envs/unitmatch     # reuse existing env (+ h5py/tqdm if missing)
export PYTHONPATH=~/vis_detect/src                # make visdetect importable — no install
cd ~/vis_detect
PROC=/ceph/.../wEPhys/<SUB>/Processed data ;  RAW=/ceph/.../wEPhys/<SUB>/Raw data
# Stage 0 — TPrime (abort-on-fail so no uncorrected pkls):
for S in $SESSIONS: python scripts/pipelines/run_tprime.py --processed-root "$PROC" --session "$S" \
        --tprime-exe /ceph/.../tools/TPrime --timeout 1800
# Stage 1 — pkl (resumable; behavior now attaches; write pkls to ceph):
for S: python scripts/conversion/raw_to_pkl.py --raw-root "$RAW" --processed-root "$PROC" \
        --out-dir /ceph/.../data/pkls/<SUB> --session "$S"
# Stage 2 — extract waveforms straight into the UM input dir:
python scripts/analysis/prep_unitmatch_full_trial_waveforms.py --subject <SUB> \
        --manifest <ceph manifest> --output /ceph/.../wEPhys/<SUB>/unit_match/input --n_workers 16
```
- Native CephFS → can use **more workers** (16) safely; TPrime's big adj write is local-to-ceph (fast).
- Staged (all TPrime → all pkl → extract) + resumable (skip-if-adj / skip-if-pkl / `_extraction_complete.txt`).

## Then (already cluster-native)
- Stage 2 writes waveforms **directly to `unit_match/input/BG_040`** on ceph → **no separate ceph-stage step needed.**
- Submit UM: `sbatch run_unitmatch_subject.sbatch BG_040` (existing).
- Then BG_040 slots into the `--subject`-aware curation pipeline.

## Getting results to local `E:` for analysis
The pkls/waveforms can be **bulk-copied to `E:` once** (robocopy/rsync) when needed for local analysis — that's a *file transfer* (Samba's intended use), not live compute. Better long-term: move curation/analysis to the cluster too.

---

## Open items to verify (do NOT verify over gw02 while it's degraded)
- **Linux TPrime** present on ceph + `run_tprime`'s subprocess cmd works on Linux (it already uses forward-slash paths).
- `visdetect` + deps install cleanly in a ceph conda env.
- ingest/extraction are path-agnostic (Path-based; spot-check no `X:` literals reached at runtime — defaults are overridden by args).
- Decide pkl output location (ceph project `data/pkls` vs `/ceph/scratch`) and whether to mirror to `E:`.

## Note
This is the template for moving **all** future ingest/extraction (and ideally curation) off the Samba mount. Local prep over `X:` is retired.
