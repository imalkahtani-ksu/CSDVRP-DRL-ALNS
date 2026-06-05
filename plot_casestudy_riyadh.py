"""
plot_casestudy_riyadh.py
Generates Figure 9 (case study) for CSDVRP-CS-Riyadh:
  Panel A: % cost reduction (CW / ALNS / DRL-ALNS bar chart)
  Panel B: Run-to-run consistency (5 independent runs)
  Panel C: Sensitivity to delta (emission penalty)
  Panel D: Sensitivity to U_max (Tier-1 visit cap)
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

BASE    = Path(__file__).parent.parent
OUT_DIR = BASE / "results" / "riyadh_casestudy"
PLOT    = BASE / "plots"
PLOT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linestyle": "--",
})

COLORS = {"CW": "#9ca3af", "ClassicalALNS": "#3b82f6", "DRL-ALNS": "#059669"}

def load():
    main  = pd.read_csv(OUT_DIR / "main_results.csv")
    sd    = pd.read_csv(OUT_DIR / "sensitivity_delta.csv")
    su    = pd.read_csv(OUT_DIR / "sensitivity_umax.csv")
    return main, sd, su

main, sd, su = load()

fig, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
fig.suptitle(
    "Case Study: NUPCO–Riyadh Medical Supply Distribution ($n\\!=\\!30$ hospitals)",
    fontweight="bold", fontsize=12)

cw_mean   = main[main["method"]=="CW"]["Z"].mean()
alns_mean = main[main["method"]=="ClassicalALNS"]["Z"].mean()
drl_mean  = main[main["method"]=="DRL-ALNS"]["Z"].mean()
alns_pct  = (cw_mean - alns_mean) / cw_mean * 100
drl_pct   = (cw_mean - drl_mean)  / cw_mean * 100

# Panel A — cost comparison
ax = axes[0]
labels = ["CW\n(baseline)", "Classical\nALNS", "DRL-ALNS\n(proposed)"]
values = [cw_mean, alns_mean, drl_mean]
colors = [COLORS["CW"], COLORS["ClassicalALNS"], COLORS["DRL-ALNS"]]
bars = ax.bar(labels, values, color=colors, edgecolor="white",
              linewidth=0.5, alpha=0.9, width=0.55)
for bar, v in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, v*1.01, f"{v:.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Mean Objective $Z^*$")
ax.set_title("Solution Quality", fontweight="bold")
ax.text(1, alns_mean + (cw_mean-alns_mean)*0.5,
        f"−{alns_pct:.1f}%", ha="center", color=COLORS["ClassicalALNS"],
        fontsize=9, fontweight="bold")
ax.text(2, drl_mean + (cw_mean-drl_mean)*0.5,
        f"−{drl_pct:.1f}%", ha="center", color=COLORS["DRL-ALNS"],
        fontsize=9, fontweight="bold")

# Panel B — run consistency
ax = axes[1]
alns_runs = main[main["method"]=="ClassicalALNS"]["Z"].values
drl_runs  = main[main["method"]=="DRL-ALNS"]["Z"].values
rng = np.random.default_rng(42)
for i, (vals, col, lab) in enumerate(zip(
        [alns_runs, drl_runs],
        [COLORS["ClassicalALNS"], COLORS["DRL-ALNS"]],
        ["Classical ALNS", "DRL-ALNS"])):
    jit = rng.uniform(-0.12, 0.12, len(vals))
    ax.scatter(np.full(len(vals), i)+jit, vals, color=col, s=80,
               alpha=0.85, zorder=3, edgecolors="white", lw=0.4, label=lab)
    ax.hlines(np.median(vals), i-0.22, i+0.22, color=col, lw=2.5, zorder=4)
ax.axhline(cw_mean, color=COLORS["CW"], ls="--", lw=1.5, label=f"CW baseline ({cw_mean:.0f})")
ax.set_xticks([0,1]); ax.set_xticklabels(["Classical\nALNS", "DRL-ALNS"])
ax.set_ylabel("$Z^*$"); ax.set_title("Run Consistency (5 runs)", fontweight="bold")
ax.legend(fontsize=8, framealpha=0.7)

# Panel C — sensitivity to delta
ax = axes[2]
for method, col in [("Z_ALNS", COLORS["ClassicalALNS"]),
                     ("Z_DRL",  COLORS["DRL-ALNS"]),
                     ("Z_CW",   COLORS["CW"])]:
    label = {"Z_ALNS": "Classical ALNS", "Z_DRL": "DRL-ALNS", "Z_CW": "CW"}[method]
    ls    = {"Z_ALNS": "-", "Z_DRL": "-", "Z_CW": "--"}[method]
    ax.plot(sd["delta"], sd[method], color=col, lw=2.0,
            marker="o", ms=5, ls=ls, label=label)
ax.axvline(15, color="gray", ls=":", lw=1, alpha=0.7, label="Baseline $\\delta=15$")
ax.set_xlabel("Emission penalty $\\delta$")
ax.set_ylabel("$Z^*$")
ax.set_title("Sensitivity to $\\delta$", fontweight="bold")
ax.legend(fontsize=8, framealpha=0.7)

# Panel D — sensitivity to U_max
ax = axes[3]
x = su["U_max_T1"].values
w = 0.28
xi = np.arange(len(x))
ax.bar(xi-w, su["Z_CW"],   w, color=COLORS["CW"],           alpha=0.85, label="CW", edgecolor="white")
ax.bar(xi,   su["Z_ALNS"], w, color=COLORS["ClassicalALNS"],alpha=0.85, label="ALNS", edgecolor="white")
ax.bar(xi+w, su["Z_DRL"],  w, color=COLORS["DRL-ALNS"],     alpha=0.85, label="DRL-ALNS", edgecolor="white")
ax.set_xticks(xi)
ax.set_xticklabels([f"$U_{{\\max}}$={u}" for u in x])
ax.set_ylabel("$Z^*$")
ax.set_title("Sensitivity to $U_{\\max}$ (Tier-1)", fontweight="bold")
ax.legend(fontsize=8, framealpha=0.7)

out_path = PLOT / "Fig9_casestudy.png"
fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# Print summary for paper
print(f"\n=== VALUES FOR PAPER ===")
print(f"CW Z*:      {cw_mean:.1f}")
print(f"ALNS Z*:    {alns_mean:.1f}  (−{alns_pct:.1f}% vs CW)")
print(f"DRL-ALNS Z*:{drl_mean:.1f}  (−{drl_pct:.1f}% vs CW)")
print(f"DRL vs ALNS: {(alns_mean-drl_mean)/alns_mean*100:.2f}%")
print(f"DRL wins: {int((drl_runs<alns_runs).sum())}/5 runs")
print(f"\nDelta sensitivity:")
print(sd.to_string(index=False))
print(f"\nUmax sensitivity:")
print(su.to_string(index=False))
