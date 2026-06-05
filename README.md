# C-SDVRP: Controlled Split Delivery VRP with DRL-ALNS

Code and data for: *A Deep Reinforcement Learning-Enhanced Adaptive Large Neighborhood
Search for the Controlled Split Delivery Vehicle Routing Problem*

## Structure

| Path | Contents |
|------|----------|
| `src/` | Core algorithm (ALNS, DRL-ALNS, PPO, operators, instance, solution) |
| `train_agent.py` | PPO training script |
| `statistical_analysis.py` | Wilcoxon tests and effect sizes |
| `plot_results.py` | Regenerate Figs 4–11 from `results/results_raw.csv` |
| `models/ppo_drl_alns.pt` | Trained PPO checkpoint (episode 580, valid gap 25.24%) |
| `results/results_raw.csv` | 1,991 rows — per-instance × per-run × per-method |
| `results/results_agg.csv` | Per-instance aggregated statistics |
| `results/statistical_tests.csv` | Wilcoxon test results by size class |
| `results/train_progress.log` | Training episode log (780 episodes) |
| `figures/` | Experimental figures Figs 4–11, generated from `results_raw.csv` |
| `data/benchmark_sample/` | Case study instance + one Class-S example |

## Installation

```bash
pip install -r requirements.txt
```

## Reproduce figures

```bash
python plot_results.py
```

All figures in `figures/` are generated directly from `results/results_raw.csv`.

## Key results (mean Z* across 181 instances, 5 runs each)

| Class | n | CW | Classical ALNS | DRL-ALNS | DRL wins |
|---|---|---|---|---|---|
| S | 10–20 | 681.3 | **583.5** | 588.7 | 14/60 |
| M | 30–50 | 1633.4 | 1171.5 | **1165.6** | 23/40 |
| L | 75–100 | 2952.2 | 2137.6 | **2119.0** | **30/40** |
| XL | 150–200 | 4470.1 | 2498.7 | **2481.0** | 21/40 |

Class-L DRL-ALNS advantage: Wilcoxon p = 0.0004, r = 0.516 (large effect).

## Reproducibility note

The trained model checkpoint (`models/ppo_drl_alns.pt`) is the episode-580 checkpoint
with validation gap 25.24% over Clarke-Wright. All benchmark instance seeds are encoded
in `results_raw.csv` (column `instance`); instances are reproducible via `CSDVRPInstance.random()`.
