import json
import os
import subprocess
import sys
import time

import matplotlib

matplotlib.use('Agg')

fig_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(fig_dir, exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

# fix ddim checkpoint path (default hardcodes /atlas/)
os.environ.setdefault("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))

script_dir = os.path.dirname(os.path.abspath(__file__))
ddim_repo = os.path.join(script_dir, "ddim")
results_dir = os.path.join(script_dir, "results")
results_json = os.path.join(script_dir, "benchmark_results.json")

steps_list = [10, 25, 50, 100, 250, 1000]
eta_sweep_steps = 50
eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
skip_type = "quad"
n_fid_samples = 50000


def build_experiments():
    exps = []
    for s in steps_list:
        exps.append((f"ddim_steps{s}", s, 0.0))
        exps.append((f"ddpm_steps{s}", s, 1.0))
    for eta in eta_values:
        if any(e[1] == eta_sweep_steps and e[2] == eta for e in exps):
            continue
        exps.append((f"eta{eta:.2f}_steps{eta_sweep_steps}", eta_sweep_steps, eta))
    return exps


def run_sampling(run_name, steps, eta):
    main_py = os.path.join(ddim_repo, "main.py")
    config = os.path.join(ddim_repo, "configs", "cifar10.yml")
    cmd = [
        sys.executable, main_py,
        "--config", config,
        "--exp", results_dir,
        "--doc", run_name,
        "--use_pretrained",
        "--sample", "--fid",
        "-i", run_name,
        "--timesteps", str(steps),
        "--eta", str(eta),
        "--skip_type", skip_type,
        "--ni",
    ]
    print(f"{run_name} | steps={steps}, eta={eta}")
    t0 = time.time()
    ret = subprocess.run(cmd)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"ERROR: {run_name} failed (exit {ret.returncode})")
        return None
    print(f"done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return elapsed


def compute_fid(run_name):
    for sub in ["image_samples", "samples"]:
        samples = os.path.join(results_dir, sub, run_name)
        if os.path.isdir(samples):
            break
    else:
        print(f"sample dir not found for {run_name}")
        return None

    cifar_dir = os.path.join(script_dir, "cifar10_train_images")
    if not os.path.isdir(cifar_dir):
        prepare_cifar10_ref()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cmd = [sys.executable, "-m", "pytorch_fid", samples, cifar_dir, "--device", dev]
    print(f"computing FID for {run_name} ({dev})...")
    try:
        ret = subprocess.run(cmd, capture_output=True, text=True)
        out = ret.stdout.strip()
        print(f"{out}")
        if ret.returncode != 0:
            print(f"stderr: {ret.stderr.strip()}")
            return None
        for line in out.split("\n"):
            if "FID" in line.upper():
                return float(line.split(":")[-1].strip())
    except Exception as e:
        print(f"FID error: {e}")
    return None


def prepare_cifar10_ref():
    import torchvision
    from torchvision.utils import save_image
    print("saving cifar10 train images for FID reference...")
    out = os.path.join(script_dir, "cifar10_train_images")
    os.makedirs(out, exist_ok=True)
    ds = torchvision.datasets.CIFAR10(
        root=os.path.join(script_dir, "data"), train=True, download=True,
        transform=torchvision.transforms.ToTensor())
    for i, (img, _) in enumerate(ds):
        save_image(img, os.path.join(out, f"{i:05d}.png"))
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(ds)}")


def load_results():
    if os.path.isfile(results_json):
        with open(results_json) as f:
            return json.load(f)
    return {}


def save_results(res):
    with open(results_json, "w") as f:
        json.dump(res, f, indent=2)

def plot_fid_vs_steps(results):
    ddim, ddpm = {}, {}
    for name, d in results.items():
        if d.get("fid") is None:
            continue
        if d["eta"] == 0.0 and name.startswith("ddim_"):
            ddim[d["steps"]] = d["fid"]
        elif d["eta"] == 1.0 and name.startswith("ddpm_"):
            ddpm[d["steps"]] = d["fid"]
    if not ddim and not ddpm:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if ddim:
        s = sorted(ddim)
        ax.plot(s, [ddim[x] for x in s], "o-", label=r"DDIM ($\eta=0$)")
    if ddpm:
        s = sorted(ddpm)
        ax.plot(s, [ddpm[x] for x in s], "s--", label=r"DDPM ($\eta=1$)")
    ax.set_xlabel("Sampling Steps")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("FID vs Sampling Steps (CIFAR-10)")
    ax.set_xscale("log")
    ax.set_xticks(steps_list)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/fid_vs_steps.png", dpi=150)
    plt.close()


def plot_fid_vs_eta(results):
    data = {}
    for name, d in results.items():
        if d.get("fid") is None:
            continue
        if d["steps"] == eta_sweep_steps:
            data[d["eta"]] = d["fid"]
    if len(data) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    etas = sorted(data)
    ax.plot(etas, [data[e] for e in etas], "o-")
    for e in etas:
        label = "DDIM" if e == 0.0 else ("DDPM" if e == 1.0 else "")
        if label:
            ax.annotate(label, (e, data[e]), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=10)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title(f"FID vs $\\eta$ at {eta_sweep_steps} Steps (CIFAR-10)")
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/fid_vs_eta.png", dpi=150)
    plt.close()


def plot_time_vs_steps(results):
    ddim, ddpm = {}, {}
    for name, d in results.items():
        if d.get("sampling_time_sec") is None:
            continue
        if d["eta"] == 0.0 and name.startswith("ddim_"):
            ddim[d["steps"]] = d["sampling_time_sec"] / 60
        elif d["eta"] == 1.0 and name.startswith("ddpm_"):
            ddpm[d["steps"]] = d["sampling_time_sec"] / 60
    if not ddim and not ddpm:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if ddim:
        s = sorted(ddim)
        ax.plot(s, [ddim[x] for x in s], "o-", label=r"DDIM ($\eta=0$)")
    if ddpm:
        s = sorted(ddpm)
        ax.plot(s, [ddpm[x] for x in s], "s--", label=r"DDPM ($\eta=1$)")
    ax.set_xlabel("Sampling Steps")
    ax.set_ylabel("Inference Time (min)")
    ax.set_title("Inference Time vs Sampling Steps (50K samples)")
    ax.set_xscale("log")
    ax.set_xticks(steps_list)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/time_vs_steps.png", dpi=150)
    plt.close()


def plot_sample_grid():
    configs = [
        ("ddim_steps10", "DDIM 10 steps"),
        ("ddim_steps50", "DDIM 50 steps"),
        ("ddim_steps100", "DDIM 100 steps"),
        ("ddpm_steps10", "DDPM 10 steps"),
        ("ddpm_steps50", "DDPM 50 steps"),
        ("ddpm_steps100", "DDPM 100 steps"),
    ]
    rows = []
    for run_name, label in configs:
        for sub in ["image_samples", "samples"]:
            d = os.path.join(results_dir, sub, run_name)
            if os.path.isdir(d):
                pngs = sorted([f for f in os.listdir(d) if f.endswith(".png")])[:8]
                if pngs:
                    rows.append((label, [os.path.join(d, p) for p in pngs]))
                    break
    if not rows:
        return

    n_rows = len(rows)
    n_cols = min(8, min(len(imgs) for _, imgs in rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    for row, (label, imgs) in enumerate(rows):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.imshow(plt.imread(imgs[col]))
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(label, fontsize=10)
    fig.suptitle("Generated Samples: DDIM vs DDPM", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/sample_grid.png", dpi=150, bbox_inches="tight")
    plt.close()


def _find_sample_dir(run_name):
    for sub in ["image_samples", "samples"]:
        d = os.path.join(results_dir, sub, run_name)
        if os.path.isdir(d):
            return d
    return None


def _get_pngs(d, indices):
    pngs = sorted([f for f in os.listdir(d) if f.endswith(".png")])
    return [os.path.join(d, pngs[i]) for i in indices if i < len(pngs)]


def plot_steps_comparison():
    step_list = [10, 25, 50, 100]
    indices = list(range(8))
    sections = [
        ("DDIM", 0.0, "ddim_steps{}"),
        ("DDPM", 1.0, "ddpm_steps{}"),
    ]
    n_cols = len(indices)
    n_rows = len(step_list)
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 2 * 1.3, n_rows * 1.5))

    for sec_i, (sec_label, eta, fmt) in enumerate(sections):
        for row, s in enumerate(step_list):
            d = _find_sample_dir(fmt.format(s))
            if not d:
                continue
            imgs = _get_pngs(d, indices)
            for col, img_path in enumerate(imgs):
                ax = axes[row, sec_i * n_cols + col]
                ax.imshow(plt.imread(img_path))
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"{s} steps", fontsize=9)
                if row == 0:
                    if col == n_cols // 2:
                        ax.set_title(sec_label, fontsize=11, fontweight="bold")

    fig.suptitle("Same Seed Across Steps: DDIM vs DDPM", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/sample_steps_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_eta_grid():
    etas = [0.0, 0.25, 0.5, 0.75, 1.0]
    indices = list(range(8))
    n_cols = len(indices)
    n_rows = len(etas)

    rows = []
    for eta in etas:
        if eta == 0.0:
            name = "ddim_steps50"
        elif eta == 1.0:
            name = "ddpm_steps50"
        else:
            name = f"eta{eta:.2f}_steps50"
        d = _find_sample_dir(name)
        if not d:
            continue
        imgs = _get_pngs(d, indices)
        if imgs:
            label = f"$\\eta={eta:.2f}$"
            if eta == 0.0:
                label += " (DDIM)"
            elif eta == 1.0:
                label += " (DDPM)"
            rows.append((label, imgs))

    if len(rows) < 2:
        return

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    for row, (label, imgs) in enumerate(rows):
        for col in range(min(n_cols, len(imgs))):
            ax = axes[row, col]
            ax.imshow(plt.imread(imgs[col]))
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(label, fontsize=10)
    fig.suptitle("Effect of $\\eta$ on Sample Quality (50 Steps)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/sample_eta_grid.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_all(results):
    plot_fid_vs_steps(results)
    plot_fid_vs_eta(results)
    plot_time_vs_steps(results)
    plot_sample_grid()
    plot_steps_comparison()
    plot_eta_grid()
    print(f"figures saved to {fig_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fid-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(ddim_repo, "main.py")):
        print("ERROR: ddim repo not found. run setup_env.sh first.")
        sys.exit(1)

    experiments = build_experiments()
    print(f"experiments: {len(experiments)}")
    for name, s, e in experiments:
        print(f"{name}: steps={s}, eta={e}")

    results = load_results() if (args.resume or args.fid_only) else {}

    if args.plot_only:
        results = load_results()
        plot_all(results)
        sys.exit(0)

    t_total = time.time()

    for i, (run_name, steps, eta) in enumerate(experiments):
        print(f"[{i+1}/{len(experiments)}] {run_name}")

        if run_name in results and results[run_name].get("fid") is not None:
            print(f"skip (FID={results[run_name]['fid']:.2f})")
            continue

        if not args.fid_only:
            t_sample = run_sampling(run_name, steps, eta)
        else:
            t_sample = results.get(run_name, {}).get("sampling_time_sec")

        fid = compute_fid(run_name)

        results[run_name] = {
            "steps": steps, "eta": eta, "skip_type": skip_type,
            "sampling_time_sec": t_sample, "fid": fid,
        }
        save_results(results)

    elapsed = time.time() - t_total
    print(f"all done in {elapsed:.1f}s ({elapsed/3600:.1f} hrs)")

    print(f"{'Name':<25} {'Steps':>6} {'Eta':>5} {'Time(s)':>8} {'FID':>8}")
    for name in sorted(results):
        d = results[name]
        t = f"{d['sampling_time_sec']:.0f}" if d.get("sampling_time_sec") else "N/A"
        f = f"{d['fid']:.2f}" if d.get("fid") else "N/A"
        print(f"{name:<25} {d['steps']:>6} {d['eta']:>5.2f} {t:>8} {f:>8}")

    plot_all(results)
