#!/bin/bash
#SBATCH --job-name=fastpulse_ephys
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s.kuroda@ucl.ac.uk

source /etc/profile.d/modules.sh

echo "SLURM job info:"
scontrol show job $SLURM_JOB_ID

module load miniconda
conda activate pupilsense

python run_fastpulse_ephys.py
