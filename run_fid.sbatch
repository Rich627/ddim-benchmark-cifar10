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
source activate ddim_env

cd "/home/juichili/EEE 598 Generative AI Theory and Practice final project"
python -u benchmark.py --fid-only
