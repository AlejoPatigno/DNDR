"""Eigenvalue spectrum of the frozen DMAP operator for several beta factors.

    python spectrum.py config_spectrum_emnist.json

Same data, anchors (training split) and input standardization as experiment.py, and
beta = factor * base_beta as resolve_beta does, so each curve is the spectrum the encoder of a run
with that beta_factor would truncate.  Writes spectrum.npz and spectrum.png/.pdf to cfg['out']:
(a) lambda_k against k, one curve per beta; (b) the same values as a beta x k "spectrogram".
"""
import json
from pathlib import Path
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from data import load_data
from geometry import base_beta
from spectral import dmap_spectrum

FLOOR = 1e-16   # plotting floor only: rounding can leave the far tail at or below zero


def run(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, raw_f, truth, split, meta = load_data(cfg)
    u = raw_f[split[0]].astype(np.float64)
    u = (u-u.mean(0))/max(float(np.sqrt(np.mean((u-u.mean(0))**2))), 1e-12)
    d, factors = int(cfg.get("d", 1000)), [float(f) for f in cfg["factors"]]
    b0 = base_beta(u)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    values = []
    for f in factors:
        started = time.perf_counter()
        values.append(dmap_spectrum(u, d, f*b0, cfg.get("alpha", 1.), cfg.get("backend", "flash"), device))
        print(f"factor={f:g} beta={f*b0:.4g} lambda_1={values[-1][1]:.6f} lambda_{d}={values[-1][-1]:.3e} "
              f"({time.perf_counter()-started:.0f} s)", flush=True)
    values = np.array(values)
    np.savez_compressed(out/"spectrum.npz", eigenvalues=values, factors=factors, betas=np.array(factors)*b0,
                        base_beta=b0, n_anchors=len(u), d=d)
    (out/"spectrum_meta.json").write_text(json.dumps({"config": cfg, "data": meta}, indent=2, default=str))
    figure(values, factors, b0, len(u), cfg, out/"spectrum")
    return values


def figure(values, factors, b0, n, cfg, stem):
    k = np.arange(1, values.shape[1])
    lam = np.clip(values[:, 1:], FLOOR, None)          # drop the Perron value lambda_0 = 1
    colors = plt.cm.viridis(np.linspace(0, 1, len(factors)))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
    for row, f, c in zip(lam, factors, colors):
        ax.plot(k, row, color=c, lw=1.2, label=f"factor {f:g} (beta={f*b0:.3g})")
    ax.set(xscale="log", yscale="log", xlabel="mode k", ylabel=r"$\lambda_k$",
           title=f"(a) DMAP spectrum, d={len(k)}")
    ax.legend(fontsize=7, loc="lower left")
    im = bx.imshow(np.log10(lam), aspect="auto", origin="lower", cmap="magma", interpolation="nearest",
                   extent=[.5, len(k)+.5, -.5, len(factors)-.5])
    bx.set_yticks(range(len(factors)), [f"{f:g}" for f in factors])
    bx.set(xlabel="mode k", ylabel="beta factor (x base beta)", title=r"(b) $\log_{10}\lambda_k$ by beta")
    fig.colorbar(im, ax=bx, label=r"$\log_{10}\lambda_k$")
    fig.suptitle(f"{cfg['dataset']}{'/'+cfg['split'] if cfg.get('split') else ''}: N={n} anchors, "
                 f"alpha={cfg.get('alpha', 1.)}, base beta={b0:.3g} (median heuristic)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
