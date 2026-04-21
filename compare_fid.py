import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Paper Table 1 (CIFAR-10, quadratic skip)
# From Song et al. 2021, Table 1
paper_fid = {
    "ddim": {10: 13.36, 20: 6.84, 50: 4.67, 100: 4.16, 1000: 3.17},
    "ddpm": {10: 72.19, 20: 23.42, 50: 7.23, 100: 5.21, 1000: 3.17},
}

# Load our results
results_json = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
with open(results_json) as f:
    results = json.load(f)

our_ddim, our_ddpm = {}, {}
for name, d in results.items():
    if d.get("fid") is None:
        continue
    if d["eta"] == 0.0 and name.startswith("ddim_"):
        our_ddim[d["steps"]] = d["fid"]
    elif d["eta"] == 1.0 and name.startswith("ddpm_"):
        our_ddpm[d["steps"]] = d["fid"]

# Print comparison table
print(f"{'Steps':>6}  {'Paper DDIM':>11}  {'Ours DDIM':>10}  {'Paper DDPM':>11}  {'Ours DDPM':>10}")
print("-" * 60)
all_steps = sorted(set(list(paper_fid["ddim"].keys()) + list(our_ddim.keys()) + list(our_ddpm.keys())))
for s in all_steps:
    p_ddim = f"{paper_fid['ddim'][s]:.2f}" if s in paper_fid["ddim"] else "-"
    o_ddim = f"{our_ddim[s]:.2f}" if s in our_ddim else "-"
    p_ddpm = f"{paper_fid['ddpm'][s]:.2f}" if s in paper_fid["ddpm"] else "-"
    o_ddpm = f"{our_ddpm[s]:.2f}" if s in our_ddpm else "-"
    print(f"{s:>6}  {p_ddim:>11}  {o_ddim:>10}  {p_ddpm:>11}  {o_ddpm:>10}")

# Plot comparison
fig_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(fig_dir, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# DDIM comparison
ps = sorted(paper_fid["ddim"])
ax1.plot(ps, [paper_fid["ddim"][s] for s in ps], "o--", label="Paper DDIM", color="C0", alpha=0.6)
if our_ddim:
    os_ = sorted(our_ddim)
    ax1.plot(os_, [our_ddim[s] for s in os_], "o-", label="Ours DDIM", color="C0")
ps = sorted(paper_fid["ddpm"])
ax1.plot(ps, [paper_fid["ddpm"][s] for s in ps], "s--", label="Paper DDPM", color="C1", alpha=0.6)
if our_ddpm:
    os_ = sorted(our_ddpm)
    ax1.plot(os_, [our_ddpm[s] for s in os_], "s-", label="Ours DDPM", color="C1")
ax1.set_xlabel("Sampling Steps")
ax1.set_ylabel("FID (lower is better)")
ax1.set_title("FID: Ours vs Paper (CIFAR-10)")
ax1.set_xscale("log")
ax1.set_xticks([10, 20, 25, 50, 100, 250, 1000])
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.legend()

# Eta sweep (ours only)
eta_data = {}
for name, d in results.items():
    if d.get("fid") is None:
        continue
    if d["steps"] == 50:
        eta_data[d["eta"]] = d["fid"]
if len(eta_data) >= 2:
    etas = sorted(eta_data)
    ax2.plot(etas, [eta_data[e] for e in etas], "o-", color="C2")
    for e in etas:
        label = "DDIM" if e == 0.0 else ("DDPM" if e == 1.0 else "")
        if label:
            ax2.annotate(label, (e, eta_data[e]), textcoords="offset points",
                         xytext=(0, 12), ha="center", fontsize=10)
    ax2.set_xlabel(r"$\eta$")
    ax2.set_ylabel("FID (lower is better)")
    ax2.set_title("FID vs $\\eta$ at 50 Steps (Ours)")
    ax2.set_xlim(-0.05, 1.05)
else:
    ax2.text(0.5, 0.5, "FID not computed yet", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("FID vs $\\eta$ (waiting for data)")

plt.tight_layout()
plt.savefig(f"{fig_dir}/fid_comparison.png", dpi=150)
plt.close()
print(f"\nfigure saved to {fig_dir}/fid_comparison.png")
