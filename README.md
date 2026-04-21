# EEE 598 Generative AI Theory and Practice - Final Project

## Denoising Diffusion Implicit Models: Accelerated Sampling and Quality Trade-offs

Benchmark comparison of DDIM vs DDPM sampling on CIFAR-10, evaluating FID scores
and inference time across different step counts and stochasticity (eta) values.

**Reference Paper:** Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models.* ICLR 2021. arXiv:2010.02502

## Setup on ASU SOL

### 1. Upload project to SOL

```bash
# from local machine (includes ddim/ repo)
scp -r "/Users/rich/Desktop/EEE 598 Generative AI Theory and Practice/sol_env/final_project/"* \
    juichili@sol.asu.edu:"/home/juichili/EEE 598 Generative AI Theory and Practice final project/"
```

### 2. Create environment

Login node has memory limits, so submit env setup as a batch job:

```bash
sbatch setup_env.sbatch
# check when done
squeue -u juichili
cat setup_env_*.out
```

Or use an interactive session:

```bash
interactive -p general -q class -t 0-01:00:00
bash setup_env.sh
exit
```

### 3. Re-upload after local edits (skips ddim/ repo)

```bash
# from local machine
scp "/Users/rich/Desktop/EEE 598 Generative AI Theory and Practice/sol_env/final_project/benchmark.py" \
    "/Users/rich/Desktop/EEE 598 Generative AI Theory and Practice/sol_env/final_project/run_benchmark.sbatch" \
    "/Users/rich/Desktop/EEE 598 Generative AI Theory and Practice/sol_env/final_project/environment.yml" \
    "/Users/rich/Desktop/EEE 598 Generative AI Theory and Practice/sol_env/final_project/setup_env.sh" \
    "/Users/rich/Desktop/EEE 598 Generative AI Theory and Practice/sol_env/final_project/setup_env.sbatch" \
    "/Users/rich/Desktop/EEE 598 Generative AI Theory and Practice/sol_env/final_project/README.md" \
    juichili@sol.asu.edu:"/home/juichili/EEE 598 Generative AI Theory and Practice final project/"
```

### 4. Run benchmark

```bash
sbatch run_benchmark.sbatch      # submit
squeue -u juichili               # check status
tail -f benchmark_*.out          # watch output
```

Figures are auto-generated when benchmark finishes, saved to `figures/`.

### 5. benchmark.py options

```bash
python -u benchmark.py                # full benchmark (auto-plots at end)
python -u benchmark.py --resume       # resume interrupted run
python -u benchmark.py --fid-only     # recompute FID for existing samples
python -u benchmark.py --plot-only    # regenerate figures from results json
```

### 6. Download results to local machine

```bash
# from local machine — download results, figures, and logs
scp juichili@sol.asu.edu:"/home/juichili/EEE 598 Generative AI Theory and Practice final project/benchmark_results.json" \
    "/Users/rich/Desktop/EEE 598 Generative AI Theory and Practice/sol_env/final_project/"

scp -r juichili@sol.asu.edu:"/home/juichili/EEE 598 Generative AI Theory and Practice final project/figures" \
    "/Users/rich/Desktop/EEE 598 Generative AI Theory and Practice/sol_env/final_project/"
```

## Experiments

| Experiment | Steps | Eta | Description |
|-----------|-------|-----|-------------|
| DDIM vs DDPM | 10, 25, 50, 100, 250, 1000 | 0.0 / 1.0 | FID and time comparison |
| Eta sweep | 50 | 0.0, 0.25, 0.5, 0.75, 1.0 | Effect of stochasticity |

## SOL Resource Limits

- Course limit lifted — use `public` QOS (no GPU minute cap)
- Max wall time: 7 days (public partition)
- Using: A100 GPU, public partition, public QOS

## Team

- Anish Verma (Lead)
- Jui-Chi Liu (Coding)
- Aaldin Kesavan Helen
