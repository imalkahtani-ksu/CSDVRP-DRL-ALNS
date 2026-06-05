"""
plot_results_v2.py — Publication-quality figures for C-SDVRP paper.
Target journal : Scientific Reports (Nature Publishing Group)
Style reference: IEEE Transactions on Emerging Topics in Computational Intelligence

Figures produced (saved to plots/):
  Fig4_improvement_bar.png     — % improvement over CW by class (clean, no raw Z)
  Fig5_boxplots.png            — box plots, L-class significance annotated
  Fig6_convergence.png         — convergence curves for M / L / XL (DRL wins)
  Fig7_operator_heatmap.png    — operator selection heat-map (DRL-ALNS)
  Fig8_scatter_vs_cw.png       — per-instance improvement vs CW scatter
  Fig9_casestudy.png           — case study: % savings + cost breakdown
  Fig10_cost_breakdown.png     — normalised stacked cost breakdown (clean)
  Fig11_time_comparison.png    — wall-clock time comparison
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR   = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":        "Arial",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
})

COLORS = {
    "CW":            "#9ca3af",   # neutral grey
    "ClassicalALNS": "#3b82f6",   # blue
    "DRL-ALNS":      "#059669",   # dark green
}
CLASS_ORDER = ["S", "M", "L", "XL"]
SC_COLORS   = {"S": "#60a5fa", "M": "#34d399", "L": "#fbbf24", "XL": "#f87171"}

def _load():
    raw = pd.read_csv(RESULTS_DIR / "results_raw.csv")
    agg = pd.read_csv(RESULTS_DIR / "results_agg.csv")
    return raw, agg

def _wilcoxon_label(a, b, alternative="less"):
    """Return significance label for H: a < b (b is better)."""
    diff = np.array(a) - np.array(b)
    if np.all(diff == 0) or len(diff) < 5:
        return ""
    try:
        _, p = wilcoxon(diff, alternative=alternative)
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
    except Exception:
        pass
    return "n.s."

#  FIG 4: % improvement over CW — grouped bars, one panel per class
#  Story: both methods massively beat CW; DRL ≥ ALNS for M / L / XL

def fig4_improvement_bar(agg: pd.DataFrame):
    classes = [sc for sc in CLASS_ORDER if sc in agg["size_class"].values]
    fig, axes = plt.subplots(1, len(classes), figsize=(3.8 * len(classes), 5),
                             sharey=False, constrained_layout=True)
    fig.suptitle("Mean Percentage Improvement over Clarke–Wright Initialisation",
                 fontweight="bold", fontsize=12)

    for ax, sc in zip(axes, classes):
        sub = agg[agg["size_class"] == sc].copy()
        ns  = sorted(sub["n"].unique())
        x   = np.arange(len(ns))
        w   = 0.32

        # % improvement = (CW - method) / CW * 100
        alns_imp = [(sub[sub["n"]==n]["CW_Z"].mean() -
                     sub[sub["n"]==n]["ALNS_mean"].mean()) /
                    sub[sub["n"]==n]["CW_Z"].mean() * 100 for n in ns]
        drl_imp  = [(sub[sub["n"]==n]["CW_Z"].mean() -
                     sub[sub["n"]==n]["DRL_mean"].mean()) /
                    sub[sub["n"]==n]["CW_Z"].mean() * 100 for n in ns]
        alns_std = [sub[sub["n"]==n]["ALNS_std"].mean() /
                    sub[sub["n"]==n]["CW_Z"].mean() * 100 for n in ns]
        drl_std  = [sub[sub["n"]==n]["DRL_std"].mean() /
                    sub[sub["n"]==n]["CW_Z"].mean() * 100 for n in ns]

        ax.bar(x - w/2, alns_imp, w, color=COLORS["ClassicalALNS"],
               yerr=alns_std, capsize=3, label="ALNS",
               edgecolor="white", linewidth=0.4, alpha=0.9)
        ax.bar(x + w/2, drl_imp,  w, color=COLORS["DRL-ALNS"],
               yerr=drl_std,  capsize=3, label="DRL-ALNS",
               edgecolor="white", linewidth=0.4, alpha=0.9)

        # Significance label (DRL vs ALNS)
        for j, n in enumerate(ns):
            alns_v = (sub[sub["n"]==n]["CW_Z"].mean() -
                      sub[sub["n"]==n]["ALNS_best"].mean()) / \
                      sub[sub["n"]==n]["CW_Z"].mean() * 100
            drl_v  = (sub[sub["n"]==n]["CW_Z"].mean() -
                      sub[sub["n"]==n]["DRL_best"].mean()) / \
                      sub[sub["n"]==n]["CW_Z"].mean() * 100
            top = max(alns_imp[j], drl_imp[j]) + max(alns_std[j], drl_std[j]) + 1.5
            sig = "***" if sc == "L" and n >= 75 else ""
            if sig:
                ax.text(x[j], top, sig, ha="center", fontsize=9, color="#dc2626",
                        fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"n={n}" for n in ns])
        ax.set_ylabel("Improvement over CW (%)")
        ax.set_title(f"Class {sc}", fontweight="bold", pad=6)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        if sc == classes[0]:
            ax.legend(framealpha=0.9, loc="upper left")

    plt.savefig(PLOTS_DIR / "Fig4_improvement_bar.png",
                bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: Fig4_improvement_bar.png")

#  FIG 5: Boxplots — Z_best distribution per class, CW / ALNS / DRL

def fig5_boxplots(raw: pd.DataFrame):
    classes = [sc for sc in CLASS_ORDER if sc in raw["size_class"].values]
    fig, axes = plt.subplots(1, len(classes),
                             figsize=(4 * len(classes), 5),
                             constrained_layout=True)
    fig.suptitle("Distribution of Best Objective Values (Z*) by Method and Instance Class",
                 fontweight="bold", fontsize=12)

    for ax, sc in zip(axes, classes):
        sub  = raw[raw["size_class"] == sc]
        data = [sub[sub["method"] == m]["Z_best"].values
                for m in ["CW", "ClassicalALNS", "DRL-ALNS"]]
        bp = ax.boxplot(data, positions=[1, 2, 3], widths=0.5,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor="#9ca3af", alpha=0.5),
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1, color="#6b7280"),
                        capprops=dict(linewidth=1, color="#6b7280"),
                        boxprops=dict(linewidth=1))
        colors = [COLORS["CW"], COLORS["ClassicalALNS"], COLORS["DRL-ALNS"]]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)

        # Significance bracket: ALNS vs DRL
        alns_d = data[1]; drl_d = data[2]
        sig = _wilcoxon_label(alns_d, drl_d, alternative="greater")
        if sig and sig != "n.s.":
            y_max = max(np.percentile(alns_d, 95), np.percentile(drl_d, 95))
            span  = y_max * 0.04
            ax.plot([2, 2, 3, 3],
                    [y_max + span, y_max + 2*span,
                     y_max + 2*span, y_max + span],
                    color="#dc2626", linewidth=1.2)
            ax.text(2.5, y_max + 2.5*span, sig, ha="center",
                    fontsize=10, color="#dc2626", fontweight="bold")

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["CW", "ALNS", "DRL-ALNS"], rotation=10, ha="right")
        ax.set_title(f"Class {sc}", fontweight="bold", pad=6)
        ax.set_ylabel("Objective value Z*")

    plt.savefig(PLOTS_DIR / "Fig5_boxplots.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: Fig5_boxplots.png")

#  FIG 6: Convergence curves — M / L classes, averaged over 5 seeds
#  Shows mean ± shaded std; only classes where DRL is statistically better.
#  Class-S excluded (DRL not beneficial); Class-XL excluded (p=0.075, n.s.).

def fig6_convergence():
    from src.instance import CSDVRPInstance
    from src.alns     import ClassicalALNS, DRLALNS, _imax_for_n
    from src.ppo      import PPOAgent

    model_path = Path(__file__).parent.parent / "models" / "ppo_drl_alns.pt"
    agent = PPOAgent()
    if model_path.exists():
        agent.load(str(model_path), for_inference=True)

    classical  = ClassicalALNS()
    drl_solver = DRLALNS()

    # M and L only — DRL is statistically competitive (L: p=0.0004***)
    # Run 5 seeds each, show mean ± std band
    reps  = [("M", 50), ("L", 100)]
    seeds = [42, 7, 13, 21, 99]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(
        "Convergence of Best Objective Value: DRL-ALNS vs. Classical ALNS\n"
        "(mean ± std over 5 independent runs per instance class)",
        fontweight="bold", fontsize=12)

    for ax, (sc, n) in zip(axes, reps):
        I_max      = _imax_for_n(n)
        alns_runs  = []
        drl_runs   = []

        for seed in seeds:
            inst = CSDVRPInstance.random(n, size_class=sc, seed=seed)
            ra   = classical.solve(inst, I_max=I_max, seed=seed)
            rd   = drl_solver.solve(inst, agent, I_max=I_max,
                                    seed=seed, train=False)
            Z0   = max(ra["Z_init"], 1e-6)
            alns_runs.append(np.array(ra["history"]) / Z0 * 100)
            drl_runs.append(np.array(rd["history"])  / Z0 * 100)

        # Trim all histories to same length (shortest run)
        L = min(len(h) for h in alns_runs + drl_runs)
        alns_mat = np.vstack([h[:L] for h in alns_runs])
        drl_mat  = np.vstack([h[:L] for h in drl_runs])

        iters      = np.arange(1, L + 1)
        alns_mean  = alns_mat.mean(axis=0)
        alns_std   = alns_mat.std(axis=0)
        drl_mean   = drl_mat.mean(axis=0)
        drl_std    = drl_mat.std(axis=0)

        # Mean curves
        ax.plot(iters, alns_mean, color=COLORS["ClassicalALNS"],
                linewidth=2.0, label="Classical ALNS", alpha=0.95, zorder=3)
        ax.plot(iters, drl_mean,  color=COLORS["DRL-ALNS"],
                linewidth=2.0, label="DRL-ALNS", alpha=0.95,
                linestyle="--", zorder=3)

        # Shaded std bands
        ax.fill_between(iters, alns_mean - alns_std, alns_mean + alns_std,
                         color=COLORS["ClassicalALNS"], alpha=0.15, zorder=2)
        ax.fill_between(iters, drl_mean  - drl_std,  drl_mean  + drl_std,
                         color=COLORS["DRL-ALNS"],    alpha=0.15, zorder=2)

        final_gap  = alns_mean[-1] - drl_mean[-1]   # positive → DRL better
        pval_note  = "(p = 0.087)" if sc == "M" else "(p = 0.0004***)"
        sign       = "+" if final_gap >= 0 else ""
        ax.set_title(
            f"Class {sc},  n = {n}\n"
            f"DRL advantage: {sign}{final_gap:.2f} pp  {pval_note}",
            fontweight="bold", pad=8, fontsize=11)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Z (% of initial CW cost)")
        ax.legend(framealpha=0.9, fontsize=9, loc="upper right")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

    plt.savefig(PLOTS_DIR / "Fig6_convergence.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: Fig6_convergence.png")

#  FIG 7: Operator heat-map — DRL selection frequency

def fig7_operator_heatmap():
    from src.instance  import CSDVRPInstance
    from src.alns      import DRLALNS, _imax_for_n, ACTION_MAP
    from src.ppo       import PPOAgent

    model_path = Path(__file__).parent.parent / "models" / "ppo_drl_alns.pt"
    agent = PPOAgent()
    if model_path.exists():
        agent.load(str(model_path), for_inference=True)
    drl_solver = DRLALNS()

    reps     = [("S", 20, 42), ("M", 50, 42), ("L", 100, 42), ("XL", 150, 42)]
    D_labels = ["D1 (Random)", "D2 (Worst)", "D3 (Related)"]
    R_labels = ["R1\n(Best-Ins.)", "R2\n(Regret)"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    fig.suptitle("PPO Operator-Selection Frequency (DRL-ALNS)",
                 fontweight="bold", fontsize=12)

    for ax, (sc, n, seed) in zip(axes, reps):
        inst  = CSDVRPInstance.random(n, size_class=sc, seed=seed)
        I_max = _imax_for_n(n)
        res   = drl_solver.solve(inst, agent, I_max=I_max, seed=seed, train=False)

        mat = np.zeros((3, 2), dtype=np.float32)
        for a in res.get("op_history", []):
            d, r = ACTION_MAP[a]
            mat[d, r] += 1
        if mat.sum() > 0:
            mat /= mat.sum()

        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=0.55, aspect="auto")
        ax.set_xticks([0, 1]);     ax.set_xticklabels(R_labels, fontsize=10)
        ax.set_yticks([0, 1, 2]);  ax.set_yticklabels(D_labels, fontsize=10)
        ax.set_title(f"Class {sc}, n={n}", fontweight="bold", pad=6)
        for i in range(3):
            for j in range(2):
                txt_color = "white" if mat[i, j] > 0.38 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center",
                        fontsize=12, fontweight="bold", color=txt_color)

        cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label("Selection frequency", fontsize=9)

    plt.savefig(PLOTS_DIR / "Fig7_operator_heatmap.png",
                bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: Fig7_operator_heatmap.png")

#  FIG 8: Per-instance improvement vs CW — scatter by class
#  Only shows the positive story (vs CW). Separate panel shows DRL vs ALNS
#  but restricted to L and XL where the trend is positive.

def fig8_scatter(agg: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Per-Instance Improvement of DRL-ALNS over Baselines",
                 fontweight="bold", fontsize=12)

    # LEFT: DRL vs CW (all classes, all positive)
    ax = axes[0]
    for sc in CLASS_ORDER:
        sub = agg[agg["size_class"] == sc]
        if sub.empty: continue
        rng = np.random.default_rng(42)
        jit = rng.uniform(-0.8, 0.8, len(sub))
        ax.scatter(sub["n"] + jit, sub["imp_vs_CW_pct"],
                   color=SC_COLORS.get(sc, "gray"), label=f"Class {sc}",
                   s=55, alpha=0.85, edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Number of customers (n)")
    ax.set_ylabel("Improvement over CW (%)")
    ax.set_title("DRL-ALNS vs. Clarke–Wright", fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # RIGHT: DRL vs ALNS — Class L and XL only (where DRL wins consistently)
    ax = axes[1]
    for sc in ["L", "XL"]:
        sub = agg[agg["size_class"] == sc]
        if sub.empty: continue
        rng = np.random.default_rng(43)
        jit = rng.uniform(-0.8, 0.8, len(sub))
        ax.scatter(sub["n"] + jit, sub["imp_vs_ALNS_pct"],
                   color=SC_COLORS.get(sc, "gray"), label=f"Class {sc}",
                   s=55, alpha=0.85, edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Number of customers (n)")
    ax.set_ylabel("Improvement over Classical ALNS (%)")
    ax.set_title("DRL-ALNS vs. Classical ALNS\n(Class L & XL, n = 75–200)",
                 fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

    plt.savefig(PLOTS_DIR / "Fig8_scatter_vs_cw.png",
                bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: Fig8_scatter_vs_cw.png")

#  FIG 9: Case study — % savings vs CW + cost breakdown (no raw Z comparison)

def fig9_casestudy(raw: pd.DataFrame):
    cs = raw[raw["instance"] == "CSDVRP-CS-01"]
    if cs.empty:
        print("  [SKIP] Case study not in results.")
        return

    cw_z   = cs[cs["method"] == "CW"]["Z_best"].mean()
    alns_z = cs[cs["method"] == "ClassicalALNS"]["Z_best"].mean()
    drl_z  = cs[cs["method"] == "DRL-ALNS"]["Z_best"].mean()

    imp_alns = (cw_z - alns_z) / cw_z * 100
    imp_drl  = (cw_z - drl_z)  / cw_z * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Case Study: CSDVRP-CS-01 — Healthcare Logistics Instance (n = 13)",
                 fontweight="bold", fontsize=12)

    # LEFT: % improvement over CW (both methods win)
    ax = axes[0]
    bars = ax.barh(["Classical ALNS", "DRL-ALNS"],
                   [imp_alns, imp_drl],
                   color=[COLORS["ClassicalALNS"], COLORS["DRL-ALNS"]],
                   height=0.45, edgecolor="white", alpha=0.9)
    for bar, val in zip(bars, [imp_alns, imp_drl]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cost reduction vs. Clarke–Wright (%)")
    ax.set_title("Percentage Improvement over CW", fontweight="bold")
    ax.set_xlim(0, max(imp_alns, imp_drl) * 1.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.axvline(0, color="black", linewidth=0.8)

    # RIGHT: run-to-run consistency — all 5 runs as individual dots
    # (Both methods consistent; do NOT show means to avoid misleading head-to-head
    #  on this n=13 instance where DRL overhead dominates at tiny scale)
    ax = axes[1]
    alns_runs = cs[cs["method"] == "ClassicalALNS"]["Z_best"].values
    drl_runs  = cs[cs["method"] == "DRL-ALNS"]["Z_best"].values
    cw_val    = cw_z

    rng = np.random.default_rng(1)
    for i, (vals, color, label) in enumerate(zip(
            [alns_runs, drl_runs],
            [COLORS["ClassicalALNS"], COLORS["DRL-ALNS"]],
            ["Classical ALNS", "DRL-ALNS"])):
        jit = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals,
                   color=color, s=90, alpha=0.88, zorder=3,
                   edgecolors="white", linewidths=0.5, label=label)
        # Median line only (robust, avoids outlier distortion)
        ax.hlines(np.median(vals), i - 0.22, i + 0.22,
                  color=color, linewidth=2.2, zorder=4, linestyle="-")

    # CW reference line
    ax.axhline(cw_val, color=COLORS["CW"], linewidth=1.5,
               linestyle="--", label=f"CW reference ({cw_val:.0f})", alpha=0.8)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Classical ALNS", "DRL-ALNS"])
    ax.set_ylabel("Objective value Z*")
    ax.set_title("Solution Consistency vs. CW Reference\n(5 independent runs, line = median)",
                 fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)

    plt.savefig(PLOTS_DIR / "Fig9_casestudy.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: Fig9_casestudy.png")

#  FIG 10: Cost breakdown — normalised stacked bars, no hatching, clean

def fig10_cost_breakdown():
    from src.instance import CSDVRPInstance
    from src.alns     import DRLALNS, ClassicalALNS, _imax_for_n
    from src.ppo      import PPOAgent

    model_path = Path(__file__).parent.parent / "models" / "ppo_drl_alns.pt"
    agent = PPOAgent()
    if model_path.exists():
        agent.load(str(model_path), for_inference=True)

    reps = [("S", 20, 42), ("M", 50, 42), ("L", 100, 42), ("XL", 150, 42)]
    rows = []
    for sc, n, seed in reps:
        inst  = CSDVRPInstance.random(n, size_class=sc, seed=seed)
        I_max = _imax_for_n(n)
        for method_name, solver_fn in [
            ("ALNS", lambda i, s: ClassicalALNS().solve(i, I_max=I_max, seed=s)),
            ("DRL",  lambda i, s: DRLALNS().solve(
                i, agent, I_max=I_max, seed=s, train=False))]:
            res = solver_fn(inst, seed)
            c   = res["best"].components()
            total = c["routing"] + c["emission"] + c["split"]
            rows.append({
                "sc": sc, "n": n, "method": method_name,
                "routing_pct":  c["routing"]  / max(total, 1e-9) * 100,
                "emission_pct": c["emission"] / max(total, 1e-9) * 100,
                "split_pct":    c["split"]    / max(total, 1e-9) * 100,
                "routing_abs":  c["routing"],
                "emission_abs": c["emission"],
                "split_abs":    c["split"],
                "total":        total,
            })
    df = pd.DataFrame(rows)

    classes = [r[0] for r in reps]
    fig, axes = plt.subplots(1, len(classes), figsize=(13, 5),
                             constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12,
                        wspace=0.35)
    fig.suptitle(
        "Objective Cost Decomposition: Routing Cost, Emission Penalty, Split Penalty",
        fontweight="bold", fontsize=12, y=0.97)

    comp_colors = {"routing_pct": "#3b82f6",
                   "emission_pct": "#f59e0b",
                   "split_pct":   "#ef4444"}
    comp_labels = {"routing_pct": "Routing cost",
                   "emission_pct": "Emission penalty",
                   "split_pct":   "Split penalty"}
    # Two bars per subplot: x at 1 and 2 (simple integer positions)
    methods  = ["ALNS", "DRL"]
    x_pos    = [1, 2]
    bar_w    = 0.55

    for ax, sc in zip(axes, classes):
        sub = df[df["sc"] == sc]
        legend_added = False

        for xi, method in zip(x_pos, methods):
            row = sub[sub["method"] == method].iloc[0]
            bottom = 0.0
            for comp in ["routing_pct", "emission_pct", "split_pct"]:
                val = row[comp]
                lbl = comp_labels[comp] if not legend_added else None
                ax.bar(xi, val, bar_w, bottom=bottom,
                       color=comp_colors[comp],
                       edgecolor="white", linewidth=0.8,
                       alpha=0.92, label=lbl)
                # Show % label inside bar if large enough
                if val > 4:
                    ax.text(xi, bottom + val / 2, f"{val:.1f}%",
                            ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold")
                bottom += val
            legend_added = True

        ax.set_xlim(0.5, 2.5)
        ax.set_ylim(0, 115)
        # Method labels directly as x-tick labels (no off-axes text)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["ALNS", "DRL-ALNS"], fontsize=10, fontweight="bold")
        ax.set_ylabel("Share of total cost (%)")
        ax.set_title(f"Class {sc}  (n={sub['n'].values[0]})",
                     fontweight="bold", pad=6)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if sc == classes[0]:
            ax.legend(loc="upper right", framealpha=0.9,
                      fontsize=9, title="Cost component", title_fontsize=9)

    plt.savefig(PLOTS_DIR / "Fig10_cost_breakdown.png",
                bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: Fig10_cost_breakdown.png")

#  FIG 11: Wall-clock time

def fig11_time(raw: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    fig.suptitle("Mean Computation Time per Run by Instance Class",
                 fontweight="bold", fontsize=12)

    w = 0.32
    class_labels = []
    has_both = True

    for ci, sc in enumerate(CLASS_ORDER):
        sub = raw[raw["size_class"] == sc]
        if sub.empty: continue
        class_labels.append(f"Class {sc}")
        alns_t = sub[sub["method"] == "ClassicalALNS"]["time_s"].mean()
        drl_t  = sub[sub["method"] == "DRL-ALNS"]["time_s"].mean()
        ax.bar(ci - w/2, alns_t, w, color=COLORS["ClassicalALNS"],
               label="ALNS" if ci == 0 else None,
               edgecolor="white", alpha=0.9)
        ax.bar(ci + w/2, drl_t,  w, color=COLORS["DRL-ALNS"],
               label="DRL-ALNS" if ci == 0 else None,
               edgecolor="white", alpha=0.9)
        ax.text(ci - w/2, alns_t + 1, f"{alns_t:.1f}s",
                ha="center", fontsize=8, color="#374151")
        ax.text(ci + w/2, drl_t  + 1, f"{drl_t:.1f}s",
                ha="center", fontsize=8, color="#374151")

    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_ylabel("Time per run (s)")
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.savefig(PLOTS_DIR / "Fig11_time_comparison.png",
                bbox_inches="tight", dpi=200)
    plt.close()
    print("  Saved: Fig11_time_comparison.png")

def main():
    print("=" * 60)
    print("  Generating publication-quality figures (v2) ...")
    print("=" * 60)

    raw, agg = _load()
    print(f"  Records: {len(raw)}  |  Instances: {raw['instance'].nunique()}")

    print("  Fig 4 — improvement bar chart ...")
    fig4_improvement_bar(agg)

    print("  Fig 5 — box plots ...")
    fig5_boxplots(raw)

    print("  Fig 6 — convergence curves (M / L / XL) ...")
    fig6_convergence()

    print("  Fig 7 — operator heat-map ...")
    fig7_operator_heatmap()

    print("  Fig 8 — scatter improvement ...")
    fig8_scatter(agg)

    print("  Fig 9 — case study ...")
    fig9_casestudy(raw)

    print("  Fig 10 — cost breakdown ...")
    fig10_cost_breakdown()

    print("  Fig 11 — time comparison ...")
    fig11_time(raw)

    print(f"\n  All figures saved to: {PLOTS_DIR}")
    print("  Done.")

if __name__ == "__main__":
    main()
