"""Run independent seeds concurrently on the two Kaggle T4s (one process per GPU)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def launch(config, cpu=False):
    cfg = json.loads(Path(config).read_text())
    if len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Seeds must be distinct.")
    import torch
    if cpu:
        if cfg.get("backend") != "torch" or not cfg.get("allow_cpu"):
            raise ValueError("CPU tests require backend='torch', allow_cpu=true.")
        devices = [None]
    else:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Select GPU T4 x2 in Kaggle accelerator settings.")
        devices = [0, 1]
    out = Path(cfg["out"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(config).resolve()
    snapshot = out/"config.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != cfg:
        raise ValueError("Output belongs to a different configuration; choose a new out path.")
    snapshot.write_text(json.dumps(cfg, indent=2))
    def queue(slot):
        failures = []
        for seed in cfg["seeds"][slot::len(devices)]:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if cpu else str(devices[slot]),
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
            log = out/f"seed_{seed}.log"
            with log.open("w") as stream:
                status = subprocess.run([sys.executable, str(Path(__file__).with_name("experiment.py")),
                                         str(source), "--seed", str(seed)], env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
            print(f"seed={seed}, GPU={devices[slot]}, exit={status.returncode}, log={log}", flush=True)
            if status.returncode:
                failures.append((seed, log.read_text()[-3500:]))
        return failures
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        errors = sum(pool.map(queue, range(len(devices))), [])
    if errors:
        raise RuntimeError(f"Some runs failed; no successful-run substitution: {errors}")
    subprocess.run([sys.executable, str(Path(__file__).with_name("plots.py")), str(out)], check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    launch(a.config, a.cpu)
