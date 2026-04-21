#!/bin/bash
# create mamba env on SOL
# NOTE: run this inside an interactive session, NOT on login node
#   interactive -p general -q class -t 0-00:30:00
set -e
module load mamba/latest
mamba env create -f environment.yml
echo "done. activate with: source activate ddim_env"
