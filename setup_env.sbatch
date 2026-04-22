#!/bin/bash
#SBATCH -A class_eee59838068spring2026
#SBATCH -p general
#SBATCH -q class
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 0-01:00:00
#SBATCH -o setup_env_%j.out
#SBATCH -e setup_env_%j.err
#SBATCH --export=NONE

module load mamba/latest

cd "$HOME/ddim-benchmark"

# remove old env if exists
mamba env remove -n ddim_env -y 2>/dev/null
rm -rf ~/.conda/envs/ddim_env

# create from environment.yml
mamba env create -f environment.yml

# verify
eval "$(conda shell.bash hook)"
conda activate ddim_env
python -c "import torch; print('torch OK, cuda:', torch.cuda.is_available())"
python -c "import pytorch_fid; print('pytorch_fid OK')"

echo "ddim_env setup complete"
