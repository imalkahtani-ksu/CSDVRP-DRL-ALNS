"""
run_riyadh_casestudy.py
Runs CW, Classical ALNS, and DRL-ALNS on the CSDVRP-CS-Riyadh (n=30)
instance, then performs sensitivity analysis on delta and U_max.
"""
import sys, os, time, json
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.instance      import CSDVRPInstance
from src.alns          import ClassicalALNS, DRLALNS, _imax_for_n
from src.ppo           import PPOAgent
from src.clarke_wright import clarke_wright

BASE     = os.path.dirname(os.path.abspath(__file__))
CS_DIR   = os.path.join(BASE, "C_SDVRP_Benchmark", "case_study_riyadh")
OUT_DIR  = os.path.join(BASE, "results", "riyadh_casestudy")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE, "models", "ppo_drl_alns.pt")
N_RUNS     = 5
I_MAX      = 3000   # Class-M budget for n=30

def load_instance():
    nodes    = pd.read_csv(os.path.join(CS_DIR, "nodes.csv"))
    vehicles = pd.read_csv(os.path.join(CS_DIR, "vehicles.csv"))
    with open(os.path.join(CS_DIR, "params.json")) as f:
        params = json.load(f)
    params["n"] = int(nodes[nodes["node_type"]=="customer"].shape[0])
    return CSDVRPInstance(nodes, vehicles, params)

def load_agent():
    agent = PPOAgent()
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH, for_inference=True)
        print(f"  Loaded model: {MODEL_PATH}")
    else:
        print(f"  WARNING: model not found at {MODEL_PATH}, using untrained agent")
    return agent

print("=" * 60)
print("  CSDVRP-CS-Riyadh: Main Experiments")
print("=" * 60)

inst  = load_instance()
agent = load_agent()
print(f"  n={inst.n}, K={len(inst.vtypes)}, Q={inst.vtypes[0]['capacity']}")
print(f"  Total demand: {inst.demand[1:].sum():.1f}\n")

# ── Main experiment (5 runs per method) ──────────────────────────────────────
rows = []
for method_name, solver in [("CW", None),
                              ("ClassicalALNS", ClassicalALNS()),
                              ("DRL-ALNS",      None)]:  # DRL filled below
    for run in range(N_RUNS):
        seed = run
        t0   = time.perf_counter()
        if method_name == "CW":
            sol  = clarke_wright(inst, seed=seed)
            Z    = sol.objective()
            rows.append({"method": "CW", "run": run, "Z": Z,
                         "time_s": time.perf_counter()-t0})
            if run == 0:
                print(f"  CW: Z = {Z:.1f}")
            continue
        if method_name == "ClassicalALNS":
            res = solver.solve(inst, I_max=I_MAX, seed=seed)
        else:
            res = DRLALNS().solve(inst, agent, I_max=I_MAX, seed=seed, train=False)
        rows.append({"method": method_name, "run": run,
                     "Z": res["Z_best"], "Z_init": res["Z_init"],
                     "time_s": res["time_s"]})
        print(f"  {method_name} run {run}: Z={res['Z_best']:.1f}  "
              f"({(res['Z_init']-res['Z_best'])/res['Z_init']*100:.1f}% impr)  "
              f"{res['time_s']:.1f}s")

df_main = pd.DataFrame(rows)
df_main.to_csv(os.path.join(OUT_DIR, "main_results.csv"), index=False)

# Summary
cw_z   = df_main[df_main["method"]=="CW"]["Z"].mean()
alns_z = df_main[df_main["method"]=="ClassicalALNS"]["Z"].mean()
drl_z  = df_main[df_main["method"]=="DRL-ALNS"]["Z"].mean()
print(f"\n  Summary:")
print(f"  CW:          Z = {cw_z:.1f}")
print(f"  ALNS:        Z = {alns_z:.1f}  ({(cw_z-alns_z)/cw_z*100:.1f}% vs CW)")
print(f"  DRL-ALNS:    Z = {drl_z:.1f}  ({(cw_z-drl_z)/cw_z*100:.1f}% vs CW)")
print(f"  DRL vs ALNS: {(alns_z-drl_z)/alns_z*100:.2f}%")
drl_wins = int((df_main[df_main["method"]=="DRL-ALNS"]["Z"].values <
                df_main[df_main["method"]=="ClassicalALNS"]["Z"].values).sum())
print(f"  DRL wins: {drl_wins}/{N_RUNS} runs")

# ── Sensitivity analysis on delta ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Sensitivity analysis: emission penalty delta")
print(f"{'='*60}")

DELTA_VALUES = [0, 5, 10, 15, 20, 25, 30]
sens_rows = []
base_params = inst.params.copy()

for delta in DELTA_VALUES:
    inst.params["delta"] = delta
    res_a = ClassicalALNS().solve(inst, I_max=I_MAX, seed=0)
    res_d = DRLALNS().solve(inst, agent, I_max=I_MAX, seed=0, train=False)
    cw_sol = clarke_wright(inst, seed=0)
    sens_rows.append({
        "delta": delta,
        "Z_CW":   round(cw_sol.objective(), 2),
        "Z_ALNS": round(res_a["Z_best"], 2),
        "Z_DRL":  round(res_d["Z_best"], 2),
    })
    print(f"  delta={delta:3d}: CW={cw_sol.objective():.1f}  "
          f"ALNS={res_a['Z_best']:.1f}  DRL={res_d['Z_best']:.1f}")

inst.params["delta"] = base_params["delta"]   # restore

df_sens_d = pd.DataFrame(sens_rows)
df_sens_d.to_csv(os.path.join(OUT_DIR, "sensitivity_delta.csv"), index=False)

# ── Sensitivity analysis on U_max ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Sensitivity analysis: visit cap U_max (Tier-1 nodes)")
print(f"{'='*60}")

import copy
UMAX_VALUES = [1, 2, 3]
umax_rows = []
orig_umax = inst.U_max.copy()

for umax_t1 in UMAX_VALUES:
    nodes_tmp = pd.read_csv(os.path.join(CS_DIR, "nodes.csv"))
    nodes_tmp["U_max"] = nodes_tmp.apply(
        lambda r: umax_t1 if r["node_type"]=="customer" and r["U_max"]==3
                  else r["U_max"], axis=1)
    inst_tmp = CSDVRPInstance(nodes_tmp,
                              pd.read_csv(os.path.join(CS_DIR, "vehicles.csv")),
                              base_params.copy())
    inst_tmp.params["n"] = 30

    res_a = ClassicalALNS().solve(inst_tmp, I_max=I_MAX, seed=0)
    res_d = DRLALNS().solve(inst_tmp, agent, I_max=I_MAX, seed=0, train=False)
    cw_sol = clarke_wright(inst_tmp, seed=0)
    n_splits = sum(1 for i in range(1, 31)
                   if any(inst_tmp.U_max[i] > 1
                          for _ in [None]))   # just flag
    umax_rows.append({
        "U_max_T1":  umax_t1,
        "Z_CW":      round(cw_sol.objective(), 2),
        "Z_ALNS":    round(res_a["Z_best"], 2),
        "Z_DRL":     round(res_d["Z_best"], 2),
        "ALNS_pct":  round((cw_sol.objective()-res_a["Z_best"])/cw_sol.objective()*100, 1),
        "DRL_pct":   round((cw_sol.objective()-res_d["Z_best"])/cw_sol.objective()*100, 1),
    })
    print(f"  U_max_T1={umax_t1}: CW={cw_sol.objective():.1f}  "
          f"ALNS={res_a['Z_best']:.1f} ({(cw_sol.objective()-res_a['Z_best'])/cw_sol.objective()*100:.1f}%)  "
          f"DRL={res_d['Z_best']:.1f} ({(cw_sol.objective()-res_d['Z_best'])/cw_sol.objective()*100:.1f}%)")

df_sens_u = pd.DataFrame(umax_rows)
df_sens_u.to_csv(os.path.join(OUT_DIR, "sensitivity_umax.csv"), index=False)

print(f"\n  All results saved to {OUT_DIR}")
print("  Done.")
