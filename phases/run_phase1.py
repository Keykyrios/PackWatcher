"""
phases/run_phase1.py — Data generation + Part A/B/C/E training (CPU).

Phase 1 sequence:
    1. Generate synthetic data from all mock scenarios
    2. Train SAE + healthy centroid (Part A)
    3. Train world model (Part B)
    4. Build attack database + train coherence scorer (Part C)
    5. Run curriculum continual learning (Part E)

All steps run on CPU with small model configs for fast iteration.
Total expected time: < 30 minutes on modern hardware.

Usage:
    python -m phases.run_phase1 --n-episodes 5 --n-epochs 20
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Windows STATUS_STACK_BUFFER_OVERRUN — process crashed at exit (e.g. PyTorch
# weight_norm C cleanup on Python 3.13) but may have written output correctly.
_WINDOWS_CRASH_CODES = {3221226505}   # 0xC0000409


def _run_step(
    name: str,
    cmd: list[str],
    success_files: Optional[list[str]] = None,
) -> bool:
    """
    Run a sub-command and return True if it succeeds.

    If the process exits with a Windows crash code but all `success_files`
    exist on disk, the step is treated as successful (the crash happened
    during Python/C cleanup AFTER the actual work was done).
    """
    log.info("--- %s ---", name)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode == 0:
        log.info("%s OK (%.1fs)", name, elapsed)
        return True

    # Check for Windows crash-at-exit: if outputs exist, treat as success
    if result.returncode in _WINDOWS_CRASH_CODES and success_files:
        all_exist = all(Path(f).exists() for f in success_files)
        if all_exist:
            log.warning(
                "%s: process exited with crash code %d (likely C-extension cleanup "
                "on Python 3.13/Windows) but all output files are present — "
                "treating as OK (%.1fs)",
                name, result.returncode, elapsed,
            )
            return True

    log.error("%s FAILED (exit code %d) in %.1fs", name, result.returncode, elapsed)
    return False


def main() -> bool:
    parser = argparse.ArgumentParser(description="Pack Watcher Phase 1: data generation + training")
    parser.add_argument("--n-episodes", type=int, default=5,
                        help="Episodes per scenario for data generation")
    parser.add_argument("--n-epochs",   type=int, default=20,
                        help="Training epochs for each model")
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--model-dir",  default="models")
    parser.add_argument("--scenarios-dir", default="sim/scenarios")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("Pack Watcher — Phase 1 Training Pipeline")
    log.info("=" * 55)

    base_py = [sys.executable, "-m"]
    md = args.model_dir
    dd = args.data_dir

    steps = [
        ("Generate data",
         base_py + ["sim.data_gen",
                     "--scenarios-dir", args.scenarios_dir,
                     "--output-dir",    dd,
                     "--n-episodes",    str(args.n_episodes)],
         [f"{dd}/activations.npy", f"{dd}/labels.npy"]),

        ("Train Part A (SAE + centroid)",
         base_py + ["train.train_part_a",
                     "--data-dir",  dd,
                     "--model-dir", md,
                     "--n-epochs",  str(args.n_epochs)],
         [f"{md}/sae.pt", f"{md}/order_param_centroid.npy"]),

        ("Train Part B (World Model)",
         base_py + ["train.train_part_b",
                     "--data-dir",  dd,
                     "--model-dir", md,
                     "--n-epochs",  str(args.n_epochs)],
         [f"{md}/world_model.pt", f"{md}/wm_config.json"]),

        ("Train Part C (Attack DB + Coherence)",
         base_py + ["train.train_part_c",
                     "--data-dir",      dd,
                     "--model-dir",     md,
                     "--scenarios-dir", args.scenarios_dir,
                     "--n-epochs",      str(args.n_epochs)],
         [f"{md}/attack_db.json"]),

        ("Train Part E (Curriculum CNL+RaPO)",
         base_py + ["train.train_part_e",
                     "--data-dir",  dd,
                     "--model-dir", md,
                     "--n-epochs",  str(min(args.n_epochs, 10))],
         [f"{md}/cnl_scar_library.db"]),
    ]

    all_ok = True
    for name, cmd, success_files in steps:
        ok = _run_step(name, cmd, success_files=success_files)
        if not ok:
            all_ok = False
            log.error("Phase 1 stopping at failed step: %s", name)
            break

    if all_ok:
        log.info("=" * 55)
        log.info("Phase 1 COMPLETE. All models saved to %s/", md)
        log.info("Next: run `python -m phases.run_phase3` for full benchmark evaluation.")
    else:
        log.error("=" * 55)
        log.error("Phase 1 FAILED. Check logs above.")

    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
