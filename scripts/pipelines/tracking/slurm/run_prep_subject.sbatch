#!/usr/bin/env bash
#
# run_prep_subject.sbatch <SUBJECT>
# =================================
# Runs the UnitMatch-INPUT PREP (TPrime -> raw_to_pkl -> waveform extraction) for
# ONE subject ON AN HPC COMPUTE NODE, reading/writing CephFS NATIVELY (no Samba
# gateway). This REPLACES running the prep locally over the `X:` mount, which
# floods/locks the ceph-gw02 Samba gateway. (KS4 + UnitMatch already run this way,
# which is why they were immune to the gw02 outage.)
# See docs/AI_interaction/bg040_cluster_prep_plan_jun2026.md + memory
# feedback-no-compute-over-samba-gateway.
#
# WHY a compute node and not your laptop terminal: it's the *filesystem path* that
# matters, not where the process runs. Compute nodes mount CephFS directly (no
# Samba); your Windows `X:` mount goes through gw02. Same data, different path.
#
# ── ONE-TIME PREREQUISITES (cluster login node) ──────────────────────────────
#   1. Make sure origin/main is up to date, then clone:
#        (on your dev box)  git push origin main          # origin is behind local main
#        (on the cluster)   git clone git@github.com:CherryGarcia-code/vis_detect_analysis_Sep2025.git ~/vis_detect
#   2. Env with visdetect deps — reuse the unitmatch env, add any missing:
#        conda activate /ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/conda_envs/unitmatch
#        pip install h5py tqdm        # (only if `python -c "import h5py,tqdm"` fails)
#   3. SpikeGLX *Linux* TPrime unpacked at $CEPH/wEPhys/TPrime-linux (DONE). It
#        cannot be run directly; this script auto-runs its install.sh (builds the
#        links/ libs dir + chmods), and run_tprime.py invokes it via the bundled
#        loader (ld-linux ... --library-path links TPrime). No manual steps.
#   4. Behavioural JSONs already in Raw data/<session>/Session/  (organize_subject_data.py).
#        For BG_040 this is DONE.
#
# ── RUN ──────────────────────────────────────────────────────────────────────
#   sbatch ~/vis_detect/scripts/pipelines/tracking/slurm/run_prep_subject.sbatch BG_040
#   # then, when it finishes:
#   cd $CEPH/wEPhys/BG_040/unit_match && sbatch $CEPH/wEPhys/run_unitmatch_subject.sbatch BG_040
#
#SBATCH -J um_prep
#SBATCH -o um_prep-%A.out
#SBATCH -e um_prep-%A.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-12:00
#SBATCH -p cpu
#SBATCH --mem=96G

set -u
SUBJECT="${1:?usage: sbatch run_prep_subject.sbatch <SUBJECT>}"

# ===== EDIT THESE FOR YOUR SETUP =====
REPO="${REPO:-$HOME/vis_detect}"                                       # where you cloned the repo
CEPH="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect"
CONDA_ENV="${CONDA_ENV:-$CEPH/conda_envs/unitmatch}"                   # env with visdetect deps
TPRIME_DIR="${TPRIME_DIR:-$CEPH/wEPhys/TPrime-linux}"                  # unpacked SpikeGLX Linux TPrime folder (you placed it here)
TPRIME_EXE="${TPRIME_EXE:-$TPRIME_DIR/TPrime}"                         # the TPrime binary inside that folder (run via its bundled loader)
PKL_DIR="${PKL_DIR:-$CEPH/derivatives/pkls/$SUBJECT}"                  # where pkls are written (ceph)
NWORKERS="${NWORKERS:-16}"
# =====================================

PROC="$CEPH/wEPhys/$SUBJECT/Processed data"
RAW="$CEPH/wEPhys/$SUBJECT/Raw data"
UM_DIR="$CEPH/wEPhys/$SUBJECT/unit_match"
UM_INPUT="$UM_DIR/input"
MANIFEST="$UM_DIR/${SUBJECT}_um_extract_manifest.csv"

# Use the env's python DIRECTLY via a PATH prepend -- do NOT rely on `conda activate`.
# Under `module load miniconda`, `conda activate "$CONDA_ENV"` set CONDA_PREFIX + the
# shell prompt but left the BASE miniconda python first on PATH, so the job ran on a
# python WITHOUT the env's deps (ModuleNotFoundError: numpy) even though the env has
# them all. Prepending $CONDA_ENV/bin is bulletproof and needs neither `module load`
# nor activation. (Falls back to activation only if the env bin is somehow absent.)
if [ -x "$CONDA_ENV/bin/python" ]; then
  export PATH="$CONDA_ENV/bin:$PATH"
  hash -r 2>/dev/null || true
else
  echo "WARN: $CONDA_ENV/bin/python not found; falling back to conda activate"
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
fi
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
echo "python -> $(command -v python)  ($(python --version 2>&1))"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "$REPO"
mkdir -p "$PKL_DIR" "$UM_INPUT"

echo "==== UM PREP $SUBJECT on $(hostname) @ $(date) ===="
echo "REPO=$REPO  ENV=$CONDA_ENV  TPRIME=$TPRIME_EXE  PKL_DIR=$PKL_DIR"
python -c "import visdetect, numpy, scipy, pandas, h5py; print('imports OK; numpy', numpy.__version__)" \
  || { echo "!!! env missing deps (need visdetect importable via PYTHONPATH + numpy/scipy/pandas/h5py)"; exit 3; }
# SpikeGLX Linux TPrime can't be run directly — it needs its bundled libs under
# links/ (created by the shipped install.sh) and is invoked via that loader
# (run_tprime.py auto-detects links/ and uses it). Baked in: run install.sh once
# if links/ is missing, and ensure the binaries are executable.
if [ ! -e "$TPRIME_DIR/links/ld-linux-x86-64.so.2" ]; then
  echo "Setting up TPrime libs via install.sh ..."
  ( cd "$TPRIME_DIR" && sh ./install.sh ) || { echo "!!! TPrime install.sh failed"; exit 3; }
fi
chmod +x "$TPRIME_EXE" "$TPRIME_DIR/links/ld-linux-x86-64.so.2" 2>/dev/null || true
[ -e "$TPRIME_DIR/links/ld-linux-x86-64.so.2" ] || { echo "!!! TPrime libs not set up under $TPRIME_DIR/links"; exit 3; }

# ── 0) pre-flight + manifest (KS + edge + meta + nidaq + Session has *trials.json) ──
python - "$SUBJECT" "$PROC" "$RAW" "$PKL_DIR" "$MANIFEST" <<'PY'
import os, sys, glob
sub, proc, raw, pkl_dir, manifest = sys.argv[1:6]
ready=[]
for s in sorted(os.listdir(proc)):
    if not os.path.isdir(os.path.join(proc,s)): continue
    imecs=glob.glob(os.path.join(proc,s,"Kilosort&Phy","*_imec0"))
    if not imecs: continue
    im=imecs[0]
    ok = ((os.path.exists(os.path.join(im,"kilosort4","spike_times.npy")) or os.path.exists(os.path.join(im,"spike_times.npy")))
          and glob.glob(os.path.join(im,"*tcat.imec*.ap.xd_*.txt"))
          and glob.glob(os.path.join(im,"*.ap.meta"))
          and os.path.isdir(os.path.join(proc,s,"Nidaq"))
          and glob.glob(os.path.join(raw,s,"Session","*trials.json")))
    ready.append(s) if ok else print(f"  NOT READY: {s}")
with open(manifest,"w") as f:
    f.write("session_name,path\n")
    for s in ready: f.write(f"{s},{os.path.join(pkl_dir, s)}.pkl\n")
open(manifest+".sessions","w").write(" ".join(ready))
print(f"READY {len(ready)} sessions -> {manifest}")
PY
[ -s "${MANIFEST}.sessions" ] || { echo "!!! no ready sessions"; exit 4; }
SESSIONS="$(cat "${MANIFEST}.sessions")"
echo "Sessions: $SESSIONS"

# ── 1) TPrime  (abort-on-fail; staged so pkls only build after ALL TPrime ok = no uncorrected pkls) ──
for S in $SESSIONS; do
  echo "-- tprime $S @ $(date +%H:%M:%S) --"
  python scripts/pipelines/run_tprime.py --processed-root "$PROC" --session "$S" --tprime-exe "$TPRIME_EXE" --timeout 1800 \
    || { echo "!!! TPRIME FAIL $S -- aborting"; exit 1; }
done

# ── 2) raw_to_pkl  (resumable: skips sessions whose pkl already exists) ──
for S in $SESSIONS; do
  echo "-- raw_to_pkl $S @ $(date +%H:%M:%S) --"
  python scripts/conversion/raw_to_pkl.py --raw-root "$RAW" --processed-root "$PROC" --out-dir "$PKL_DIR" --session "$S" \
    || { echo "!!! PKL FAIL $S -- aborting"; exit 1; }
done

# ── 3) waveform extraction -> UM input on ceph  (resumable via _extraction_complete.txt) ──
# --processed-root is REQUIRED on the cluster: the extractor otherwise defaults to a
# hardcoded Windows X:\ path that doesn't exist on Linux -> every session skipped
# "No binary data found" (it now exits nonzero if zero sessions yield waveforms).
python scripts/analysis/prep_unitmatch_full_trial_waveforms.py --subject "$SUBJECT" \
  --processed-root "$PROC" \
  --manifest "$MANIFEST" --output "$UM_INPUT" --n_workers "$NWORKERS" \
  || { echo "!!! EXTRACT FAIL -- aborting"; exit 1; }

echo "==== PREP DONE @ $(date).  UM input -> $UM_INPUT ===="
echo "NEXT: cd $UM_DIR && sbatch $CEPH/wEPhys/run_unitmatch_subject.sbatch $SUBJECT"
