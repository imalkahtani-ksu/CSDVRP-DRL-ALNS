"""
train_agent.py — Train the PPO agent for DRL-ALNS on randomly generated instances.

Strategy:
  - Generate fresh random training instances each episode
  - Mix of sizes S/M/L/XL (n in {10,15,20,30,50,75,100,150,200})
  - Per-episode I_max = _imax_for_n(inst.n) — MATCHES evaluation budget exactly,
    so the agent's it/I_max state feature is always in [0,1] and in-distribution
  - Run DRL-ALNS in training mode (agent.store + agent.update per episode)
  - Validate every 20 episodes on a held-out validation set
  - Save best model based on validation reward
"""

import sys
import os
import time
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

_LOG_FILE = Path(__file__).parent / "train_progress.log"
_log_fh   = open(_LOG_FILE, "w", encoding="utf-8", buffering=1)  # line-buffered

def _log(msg: str) -> None:
    """Print to stdout AND write immediately to progress log file."""
    print(msg, flush=True)
    _log_fh.write(msg + "\n")
    _log_fh.flush()

from src.instance import CSDVRPInstance
from src.ppo      import PPOAgent
from src.alns     import DRLALNS, _imax_for_n

TRAIN_EPISODES  = 1000     # total training episodes (increased for full I_max coverage)
# NOTE: I_max per episode = _imax_for_n(inst.n) — no fixed TRAIN_I_MAX constant.
#       This ensures the agent trains at the EXACT iteration budget it will face
#       at evaluation time, eliminating the previous train/eval mismatch.
VALID_EVERY     = 20       # validate every N episodes
N_VALID         = 10       # number of validation instances
MODEL_SAVE_DIR  = Path(__file__).parent / "models"
MODEL_SAVE_DIR.mkdir(exist_ok=True)
MODEL_PATH      = MODEL_SAVE_DIR / "ppo_drl_alns.pt"

# Training instance sizes — includes XL-200 so agent learns the 6000-iter distribution
TRAIN_SIZES = [
    ("S",  10), ("S",  15), ("S",  20),
    ("M",  30), ("M",  50),
    ("L",  75), ("L", 100),
    ("XL", 150), ("XL", 200),   # both XL sizes: 5000 and 6000 iters
]

MASTER_SEED = 42

def make_train_instance(episode: int) -> CSDVRPInstance:
    size_class, n = TRAIN_SIZES[episode % len(TRAIN_SIZES)]
    seed = MASTER_SEED + episode * 97 + 3
    return CSDVRPInstance.random(n, size_class=size_class, seed=seed)

def make_valid_set():
    insts = []
    for i in range(N_VALID):
        size_class, n = TRAIN_SIZES[i % len(TRAIN_SIZES)]
        seed = 9000 + i * 13
        insts.append(CSDVRPInstance.random(n, size_class=size_class, seed=seed))
    return insts

def main():
    _log("=" * 60)
    _log("  C-SDVRP DRL-ALNS — PPO Training (v2: matched I_max)")
    _log("=" * 60)
    _log(f"  Device   : {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}")
    _log(f"  Episodes : {TRAIN_EPISODES}")
    _log(f"  I_max/ep : _imax_for_n(n)  [1500/3000/5000/6000 per size]")
    _log(f"  Sizes    : {[f'{sc}-{n}' for sc,n in TRAIN_SIZES]}")
    _log(f"  Model    : {MODEL_PATH}")
    _log("=" * 60)

    agent      = PPOAgent()
    solver     = DRLALNS()
    valid_set  = make_valid_set()

    best_valid_reward = -float("inf")

    for ep in range(TRAIN_EPISODES):
        inst   = make_train_instance(ep)
        seed   = MASTER_SEED + ep
        I_max  = _imax_for_n(inst.n)   # ← matches evaluation budget exactly
        agent.net.train()

        t0 = time.perf_counter()
        result = solver.solve(inst, agent, I_max=I_max, seed=seed, train=True)
        elapsed = time.perf_counter() - t0

        gap = (result["Z_best"] - result["Z_init"]) / max(result["Z_init"], 1e-6) * 100

        # Validation
        if (ep + 1) % VALID_EVERY == 0:
            agent.net.eval()
            valid_gaps = []
            for vi in valid_set:
                v_imax = _imax_for_n(vi.n)   # ← per-instance budget in validation too
                vr = solver.solve(vi, agent, I_max=v_imax, seed=0, train=False)
                valid_gaps.append(
                    (vr["Z_init"] - vr["Z_best"]) / max(vr["Z_init"], 1e-6) * 100
                )
            avg_valid = float(np.mean(valid_gaps))
            marker = ""
            if avg_valid > best_valid_reward:
                best_valid_reward = avg_valid
                agent.save(str(MODEL_PATH))
                marker = " *** SAVED"

            _log(f"  Ep {ep+1:4d}/{TRAIN_EPISODES} | "
                 f"n={inst.n:3d} I={I_max:4d} | gap={gap:6.2f}% | "
                 f"valid={avg_valid:6.2f}% | {elapsed:.1f}s{marker}")
        else:
            if (ep + 1) % 10 == 0:
                _log(f"  Ep {ep+1:4d}/{TRAIN_EPISODES} | "
                     f"n={inst.n:3d} I={I_max:4d} | gap={gap:6.2f}% | {elapsed:.1f}s")

    # Final save if no best found
    if not MODEL_PATH.exists():
        agent.save(str(MODEL_PATH))

    _log("\n  Training complete.")
    _log(f"  Best validation improvement: {best_valid_reward:.2f}%")
    _log(f"  Model saved to: {MODEL_PATH}")
    _log_fh.close()

if __name__ == "__main__":
    main()
