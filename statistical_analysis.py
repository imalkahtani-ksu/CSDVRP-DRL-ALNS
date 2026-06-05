"""
statistical_analysis.py  — Wilcoxon signed-rank tests for C-SDVRP paper.

Tests:
  H1 : ClassicalALNS vs CW        (are ALNS results better than CW?)
  H2 : DRL-ALNS vs ClassicalALNS  (does DRL improve over Classical ALNS?)
  H3 : DRL-ALNS vs CW             (is DRL-ALNS better than CW?)

Reports:
  - Wilcoxon p-values per size class and overall
  - Effect size (r = Z / sqrt(N))
  - Summary table
  Saves results/statistical_tests.csv  and results/statistical_tests.xlsx
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path(__file__).parent / "results"
ALPHA = 0.05

def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test H0: median(a-b)=0, H1: a < b (b is better)."""
    diffs = a - b
    if np.all(diffs == 0):
        return {"stat": np.nan, "p": 1.0, "reject": False, "effect_r": 0.0}
    try:
        stat, p = wilcoxon(diffs, alternative="greater")  # a > b (a is worse → p small)
        n = len(diffs[diffs != 0])
        # r = Z / sqrt(N); approximate Z from Wilcoxon stat
        z_approx = abs(stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
        r = z_approx / np.sqrt(n) if n > 0 else 0.0
        return {"stat": float(stat), "p": float(p), "reject": bool(p < ALPHA), "effect_r": float(r)}
    except Exception:
        return {"stat": np.nan, "p": np.nan, "reject": False, "effect_r": 0.0}

def load_data():
    raw_path = RESULTS_DIR / "results_raw.csv"
    if not raw_path.exists():
        print(f"[ERROR] {raw_path} not found. Run run_experiments.py first.")
        sys.exit(1)
    return pd.read_csv(raw_path)

def run_tests(raw: pd.DataFrame):
    """Run Wilcoxon tests per size class and overall."""
    records = []

    # Build per-instance best-of-N-runs for each method
    cw = (raw[raw["method"] == "CW"]
          .groupby("instance")["Z_best"].mean()
          .rename("CW"))
    alns = (raw[raw["method"] == "ClassicalALNS"]
            .groupby("instance")["Z_best"].mean()
            .rename("ALNS"))
    drl = (raw[raw["method"] == "DRL-ALNS"]
           .groupby("instance")["Z_best"].mean()
           .rename("DRL"))

    meta = (raw[["instance", "size_class", "n"]]
            .drop_duplicates("instance")
            .set_index("instance"))

    joined = (meta.join(cw).join(alns).join(drl)).dropna()

    def add_test(group_name, df):
        # H1: ALNS < CW (ALNS improves over CW)
        t1 = wilcoxon_test(df["CW"].values, df["ALNS"].values)
        # H2: DRL < ALNS (DRL improves over ALNS)
        t2 = wilcoxon_test(df["ALNS"].values, df["DRL"].values)
        # H3: DRL < CW
        t3 = wilcoxon_test(df["CW"].values, df["DRL"].values)

        for h, t, pair in [(1, t1, "ALNS vs CW"), (2, t2, "DRL vs ALNS"), (3, t3, "DRL vs CW")]:
            records.append({
                "group":      group_name,
                "n_instances": len(df),
                "hypothesis": f"H{h}: {pair}",
                "W_stat":     t["stat"],
                "p_value":    t["p"],
                "reject_H0":  t["reject"],
                "effect_r":   t["effect_r"],
                "significance": "***" if t["p"] < 0.001 else "**" if t["p"] < 0.01 else "*" if t["p"] < 0.05 else "n.s."
            })

    # Overall
    add_test("Overall", joined)
    # Per size class
    for sc in ["S", "M", "L", "XL"]:
        sub = joined[joined["size_class"] == sc]
        if len(sub) >= 5:
            add_test(f"Class-{sc}", sub)

    return pd.DataFrame(records)

def print_summary(tests: pd.DataFrame):
    print("\n" + "=" * 80)
    print("  Statistical Tests (Wilcoxon Signed-Rank, α=0.05)")
    print("=" * 80)
    print(f"  {'Group':<15} {'Hypothesis':<25} {'n_inst':>6} {'W':>10} {'p-value':>10} {'Reject?':>8} {'r':>7} {'Sig':>5}")
    print("  " + "-" * 78)
    for _, row in tests.iterrows():
        print("  %-15s %-25s %6d %10.1f %10.4f %8s %7.3f %5s" % (
            row["group"], row["hypothesis"], row["n_instances"],
            row["W_stat"] if row["W_stat"] == row["W_stat"] else -1,
            row["p_value"] if row["p_value"] == row["p_value"] else 1.0,
            "Yes*" if row["reject_H0"] else "No",
            row["effect_r"], row["significance"]
        ))

def compute_effect_sizes(raw: pd.DataFrame):
    """Print mean improvement ± std per size class."""
    print("\n  ── Mean Improvement over CW (%) ─────────────────────────────────")
    for sc in ["S", "M", "L", "XL", "CS"]:
        sub = raw[raw["size_class"] == sc]
        if sub.empty:
            continue
        cw   = sub[sub["method"] == "CW"]["Z_best"].mean()
        alns = sub[sub["method"] == "ClassicalALNS"]["Z_best"]
        drl  = sub[sub["method"] == "DRL-ALNS"]["Z_best"]
        if len(alns) == 0 or cw < 1e-9:
            continue
        alns_imp = (cw - alns.mean()) / cw * 100
        drl_imp  = (cw - drl.mean())  / cw * 100
        alns_vs_drl = (alns.mean() - drl.mean()) / alns.mean() * 100
        print(f"  {sc:5s}  ALNS: {alns_imp:+.2f}%  DRL: {drl_imp:+.2f}%  DRL_vs_ALNS: {alns_vs_drl:+.2f}%")

def main():
    print("=" * 60)
    print("  C-SDVRP Statistical Analysis")
    print("=" * 60)

    raw = load_data()
    print(f"  Loaded {len(raw)} raw records from {len(raw['instance'].unique())} instances")

    tests = run_tests(raw)
    print_summary(tests)
    compute_effect_sizes(raw)

    tests.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)
    print(f"\n  Saved: results/statistical_tests.csv")

    # Excel version
    try:
        tests.to_excel(RESULTS_DIR / "statistical_tests.xlsx", index=False)
        print("  Saved: results/statistical_tests.xlsx")
    except Exception as e:
        print(f"  [WARN] Excel save failed: {e}")

if __name__ == "__main__":
    main()
