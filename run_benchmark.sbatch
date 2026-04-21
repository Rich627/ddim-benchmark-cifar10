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
source activate ddim_env

cd "/home/juichili/EEE 598 Generative AI Theory and Practice final project"
python -u benchmark.py --resume
