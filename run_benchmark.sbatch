#!/bin/bash
#SBATCH -A class_eee59838068spring2026
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-16:00:00
#SBATCH -o benchmark_%j.out
#SBATCH -e benchmark_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE

module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate ddim_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$HOME/ddim-benchmark"
python -u benchmark.py --resume
