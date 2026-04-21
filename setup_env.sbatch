#!/bin/bash
#SBATCH -A class_eee59838068spring2026
#SBATCH -p general
#SBATCH -q class
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 0-01:00:00
#SBATCH -o setup_env_%j.out
#SBATCH -e setup_env_%j.err
#SBATCH --export=NONE

module load mamba/latest
cd "$HOME/ddim-benchmark"
mamba env create -f environment.yml
echo "ddim_env created successfully"
