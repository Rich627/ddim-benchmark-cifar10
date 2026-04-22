#!/bin/bash
#SBATCH -A class_eee59838068spring2026
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-01:00:00
#SBATCH -o fid_%j.out
#SBATCH -e fid_%j.err
#SBATCH --export=NONE

module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate ddim_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$HOME/ddim-benchmark"
python -u benchmark.py --fid-only --resume
