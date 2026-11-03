"""Eigenvalue spectrum of the frozen DMAP operator for several beta factors.

    python spectrum.py config_spectrum_emnist.json

Same data, anchors (training split) and input standardization as experiment.py, and
beta = factor * base_beta as resolve_beta does, so each curve is the spectrum the encoder of a run
with that beta_factor would truncate.  Writes spectrum.npz and spectrum.png/.pdf to cfg['out']:
(a) lambda_k against k, one curve per beta; (b) the same values as a beta x k "spectrogram"; (c) the
local Weyl dimension d(i).  Weyl's law lambda_i ~ i^(2/d) with lambda_i = -ln mu_i (diffusion time only
shifts the intercept) gives d = 2 / slope of ln lambda_i against ln i over cfg['weyl_window'].
method "dense" is exact (N <~ 10^4); "krylov" is matrix-free, so all_train runs on every training image.
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
from spectral import Gaussian, dmap_spectrum, krylov_spectrum

FLOOR = 1e-16   # plotting floor only: rounding can leave the far tail at or below zero


def weyl_dimension(mu, window=(10, 300), valid=None):
    """(d, slope) from the log-log fit of lambda_i = -ln mu_i over indices in window; mu[0] is the Perron value.
    Only 0 < mu_i < 1 (and valid, if given) count; NaN with fewer than 5 points."""
    i = np.arange(len(mu))
    use = (i >= window[0]) & (i <= window[1]) & (mu > 0) & (mu < 1)
    if valid is not None:
        use &= valid
    if use.sum() < 5:
        return np.nan, np.nan
    slope = np.polyfit(np.log(i[use]), np.log(-np.log(mu[use])), 1)[0]
    return 2/slope, slope


def local_dimension(mu, valid=None):
    """(centres, d_local): Weyl dimension on log-spaced windows [i, 2i]."""
    centres = np.unique(np.geomspace(2, len(mu)/2, 40).astype(int))
    return centres, np.array([weyl_dimension(mu, (i, 2*i), valid)[0] for i in centres])


def run(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, raw_f, truth, split, meta = load_data(cfg)
    u = raw_f[np.concatenate(split[:2]) if cfg.get("all_train") else split[0]].astype(np.float64)
    u = (u-u.mean(0))/max(float(np.sqrt(np.mean((u-u.mean(0))**2))), 1e-12)
    k, factors = int(cfg.get("k", 1000)), [float(f) for f in cfg["factors"]]
    method, window, tol = cfg.get("method", "dense"), tuple(cfg.get("weyl_window", (10, 300))), cfg.get("tol", 1e-4)
    b0 = base_beta(u)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    backend, alpha = cfg.get("backend", "flash"), cfg.get("alpha", 1.)
    values, residuals, valid, dims, window_res = [], [], [], [], []
    if method == "krylov":
        probe = Gaussian(u - u.mean(0), b0, backend, device, cfg.get("tile", 4096))
        probe.apply(np.ones(len(u)))                    # warm-up: compilation / allocation
        started = time.perf_counter()
        probe.apply(np.random.default_rng(0).normal(size=len(u)))
        one = time.perf_counter()-started
        m = (cfg.get("block") or k + 1 + max(10, (k+1)//10)) * cfg.get("steps", 6)
        passes = m if backend == "flash" else cfg.get("steps", 6)
        print(f"one kernel pass: {one:.2f} s, projected {one*passes/60:.1f} min per beta (m={m}, {backend})", flush=True)
    for f in factors:
        started = time.perf_counter()
        if method == "krylov":
            mu, res = krylov_spectrum(u, k, f*b0, alpha, backend, device, cfg.get("block"),
                                      cfg.get("steps", 6), cfg.get("tile", 4096))
        else:
            mu = dmap_spectrum(u, k, f*b0, alpha, backend, device)
            res = np.zeros_like(mu)
        values.append(mu), residuals.append(res), valid.append(res < tol)
        dims.append(weyl_dimension(mu, window, valid[-1])[0])
        window_res.append(float(res[window[0]:window[1]+1].max()))
        print(f"factor={f:g} beta={f*b0:.4g} lambda_1={mu[1]:.6f} lambda_{k}={mu[-1]:.3e} weyl_d={dims[-1]:.2f} "
              f"max_residual[{window[0]},{window[1]}]={window_res[-1]:.1e} ({time.perf_counter()-started:.0f} s)",
              flush=True)
    values, residuals, valid = np.array(values), np.array(residuals), np.array(valid)
    np.savez_compressed(out/"spectrum.npz", eigenvalues=values, residuals=residuals, weyl_d=dims,
                        weyl_window=window, method=method, backend=backend, k=k, factors=factors, betas=np.array(factors)*b0,
                        base_beta=b0, n_anchors=len(u))
    table = [dict(factor=f, beta=f*b0, weyl_d=dm, max_window_residual=r) for f, dm, r in zip(factors, dims, window_res)]
    (out/"spectrum_meta.json").write_text(json.dumps({"config": cfg, "data": meta, "n_anchors": len(u),
                                                      "method": method, "backend": backend, "weyl": table}, indent=2, default=str))
    figure(values, valid, dims, window, factors, b0, len(u), cfg, out/"spectrum")
    return values


def figure(values, valid, dims, window, factors, b0, n, cfg, stem):
    k = np.arange(1, values.shape[1])
    lam = np.clip(values[:, 1:], FLOOR, None)          # drop the Perron value lambda_0 = 1
    colors = plt.cm.viridis(np.linspace(0, 1, len(factors)))
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    for row, f, c, dm in zip(lam, factors, colors, dims):
        ax.plot(k, row, color=c, lw=1.2, label=f"factor {f:g}, d={dm:.1f}")
    ax.set(xscale="log", yscale="log", xlabel="mode k", ylabel=r"$\lambda_k$",
           title=f"(a) DMAP spectrum, k={len(k)}")
    ax.legend(fontsize=7, loc="lower left")
    im = bx.imshow(np.log10(lam), aspect="auto", origin="lower", cmap="magma", interpolation="nearest",
                   extent=[.5, len(k)+.5, -.5, len(factors)-.5])
    bx.set_yticks(range(len(factors)), [f"{f:g}" for f in factors])
    bx.set(xlabel="mode k", ylabel="beta factor (x base beta)", title=r"(b) $\log_{10}\lambda_k$ by beta")
    fig.colorbar(im, ax=bx, label=r"$\log_{10}\lambda_k$")
    for mu, ok, f, c in zip(values, valid, factors, colors):
        cx.plot(*local_dimension(mu, ok), color=c, lw=1.2, label=f"{f:g}")
    cx.axvspan(*window, color="0.8", alpha=.5, lw=0)
    cx.set(xscale="log", ylim=(0, 40), xlabel="mode i", ylabel="local d(i)",
           title=r"(c) $d=2/$slope of $\ln\lambda$ vs $\ln i$ on $[i,2i]$")
    fig.suptitle(f"{cfg['dataset']}{'/'+cfg['split'] if cfg.get('split') else ''}: N={n} anchors, "
                 f"alpha={cfg.get('alpha', 1.)}, base beta={b0:.3g} (median heuristic)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
