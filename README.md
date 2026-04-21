# EEE 598 Generative AI Theory and Practice - Final Project

## Denoising Diffusion Implicit Models: Accelerated Sampling and Quality Trade-offs

Benchmark comparison of DDIM vs DDPM sampling on CIFAR-10, evaluating FID scores
and inference time across different step counts and stochasticity (eta) values.

**Reference Paper:** Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models.* ICLR 2021. arXiv:2010.02502

## Setup on ASU SOL

### 1. Clone and upload to SOL

```bash
git clone https://github.com/Rich627/ddim-benchmark-cifar10.git
cd ddim-benchmark-cifar10
git clone https://github.com/ermongroup/ddim.git

# upload to SOL (replace <asurite> with your ASURITE)
scp -r ./* <asurite>@sol.asu.edu:"~/ddim-benchmark/"
```

### 2. Create environment

Login node has memory limits, so submit env setup as a batch job:

```bash
ssh <asurite>@sol.asu.edu
cd ~/ddim-benchmark
sbatch setup_env.sbatch
squeue -u $USER
cat setup_env_*.out
```

### 3. Run benchmark

```bash
sbatch run_benchmark.sbatch      # submit
squeue -u $USER                  # check status
tail -f benchmark_*.out          # watch output
```

Figures are auto-generated when benchmark finishes, saved to `figures/`.

### 4. benchmark.py options

```bash
python -u benchmark.py                # full benchmark (auto-plots at end)
python -u benchmark.py --resume       # resume interrupted run
python -u benchmark.py --fid-only     # recompute FID for existing samples
python -u benchmark.py --plot-only    # regenerate figures from results json
```

### 5. Download results to local machine

```bash
scp <asurite>@sol.asu.edu:"~/ddim-benchmark/benchmark_results.json" .
scp -r <asurite>@sol.asu.edu:"~/ddim-benchmark/figures" .
```

## Experiments

| Experiment | Steps | Eta | Description |
|-----------|-------|-----|-------------|
| DDIM vs DDPM | 10, 25, 50, 100, 250, 1000 | 0.0 / 1.0 | FID and time comparison |
| Eta sweep | 50 | 0.0, 0.25, 0.5, 0.75, 1.0 | Effect of stochasticity |

## SOL Resource Limits

- Account: `class_eee59838068spring2026`
- Using: A100 GPU, public partition, public QOS

## Team

- Anish Verma (Lead)
- Jui-Chi Liu (Coding)
- Aaldin Kesavan Helen
