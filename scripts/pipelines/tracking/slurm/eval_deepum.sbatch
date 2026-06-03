#!/usr/bin/env bash
#
# eval_deepum.sbatch -- evaluate one fine-tuned rung's checkpoints by running the
# full tracking pipeline (run_deepunitmatch_all.py --ckpt) per checkpoint and
# tabulating the span distribution vs UM/stock baselines.
#
# This is the BIG-RAM, CPU profile (NOT the GPU training profile): the pipeline's
# UnitMatch extract_metric_scores + Bayes fusion peaks ~180-230 GB and is CPU-only
# -- identical resource shape to the stock run (run_deepunitmatch_cluster.sh).
# Each checkpoint takes ~79 min, run sequentially, so pick STRIDE to bound runtime
# (stride 25 over save_freq=5 -> ~3 checkpoints ~4 h).
#
# Submit from UM_DIR after the matching training rung has finished:
#   sbatch slurm/eval_deepum.sbatch rung1_warm_frozen 25
#
#SBATCH -J deepum_eval
#SBATCH -o /ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/unit_match/logs/deepum_eval-%A.out
#SBATCH -e /ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/unit_match/logs/deepum_eval-%A.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-12:00
#SBATCH -p a100
#SBATCH --mem=400G

set -e

CEPH=/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect
UM_DIR=${CEPH}/wEPhys/BG_046/unit_match
CONDA_ENV=${CEPH}/conda_envs/unitmatch
OUT_BASE=${UM_DIR}/output/BG_046_finetune
TRACK="${DEEPUM_TRACK:-${UM_DIR}}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

RUNG="${1:-rung1_warm_frozen}"   # rung1_warm_frozen | rung2_warm_unfrozen | rung3_scratch
STRIDE="${2:-25}"                # eval every Nth epoch + last

echo "════════════════════════════════════════════════════════════"
echo "SLURM job ${SLURM_JOB_ID:-?} on $(hostname)   eval ${RUNG} stride ${STRIDE}"
echo "════════════════════════════════════════════════════════════"
[ -d "${OUT_BASE}/${RUNG}" ] || { echo "MISSING checkpoint dir: ${OUT_BASE}/${RUNG}"; exit 3; }

python -u "${TRACK}/eval_deepum_checkpoints.py" \
    --ckpt-dir "${OUT_BASE}/${RUNG}" \
    --input    "${UM_DIR}/input" \
    --out-root "${OUT_BASE}/${RUNG}_eval" \
    --stride   "${STRIDE}" \
    --label-prefix "${RUNG}"

echo "done eval ${RUNG} -- see ${OUT_BASE}/${RUNG}_eval/finetune_comparison.csv"
