"""Eigenvalue spectrum of the frozen DMAP operator for several beta factors.

    python spectrum.py config_spectrum_emnist.json

Same data, anchors (training split) and input standardization as experiment.py, and
beta = factor * base_beta as resolve_beta does, so each curve is the spectrum the encoder of a run
with that beta_factor would truncate.  Writes spectrum.npz and three separate figures (.png/.pdf) to
cfg['out'], see figures().  Weyl's law lambda_i ~ i^(2/d) with lambda_i = -ln mu_i (diffusion time only
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
import matplotlib.ticker
import numpy as np
import torch
from data import load_data
from geometry import base_beta
from spectral import Gaussian, dmap_spectrum, krylov_spectrum

FLOOR = 1e-16   # plotting floor only: rounding can leave the far tail at or below zero


def weyl_dimension(mu, window=(10, 300), valid=None):
    """(d, slope, intercept) from the log-log fit ln(-ln mu_i) = slope ln i + intercept over indices in
    window; mu[0] is the Perron value.  Only 0 < mu_i < 1 (and valid, if given) count; NaN with < 5 points."""
    i = np.arange(len(mu))
    use = (i >= window[0]) & (i <= window[1]) & (mu > 0) & (mu < 1)
    if valid is not None:
        use &= valid
    if use.sum() < 5:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(np.log(i[use]), np.log(-np.log(mu[use])), 1)
    return 2/slope, slope, intercept


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
        # Lower bound only: each beta also pays the block products and the N x m reorthogonalization on CPU.
        passes = m if backend == "flash" else cfg.get("steps", 6)
        print(f"one kernel pass: {one:.2f} s, >= {one*passes/60:.1f} min per beta (m={m}, {backend})", flush=True)
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
    figures(out, cfg)
    return values


MARKERS = "osv^Dp<>h*"


def _style(ax):
    ax.set(xscale="log", yscale="log")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0e}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0e}"))


def _save(fig, stem):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def figures(out, cfg=None):
    """Separate figures from out/spectrum.npz (also: python spectrum.py --plot <out>):
    spectrum_weyl:    lambda_n = -4 beta ln mu_n against n/N, log-log, one curve per beta, dashed effective
                      Weyl scaling (n/N)^(2/d) fitted on the window.  The Gaussian kernel exp(-beta r^2)
                      with alpha = 1 approximates the heat semigroup at t = 1/(4 beta), so this rescaling
                      makes curves of different beta collapse where the Weyl regime holds.
    spectrum_heatmap: log10 mu_k over beta x k.
    spectrum_local_d: local Weyl dimension on windows [i, 2i]."""
    out = Path(out)
    a = np.load(out/"spectrum.npz")
    cfg = cfg or json.loads((out/"spectrum_meta.json").read_text())["config"]
    values, factors, betas = a["eigenvalues"], a["factors"], a["betas"]
    n = int(a["n_anchors"])
    res = a["residuals"] if "residuals" in a.files else np.zeros_like(values)
    valid = res < cfg.get("tol", 1e-4)
    window = tuple(cfg.get("weyl_window", (10, 300)))
    colors = plt.cm.viridis(np.linspace(0, .9, len(factors)))
    title = f"{cfg['dataset']}{'/'+cfg['split'] if cfg.get('split') else ''}, N = {n}"

    fig, ax = plt.subplots(figsize=(6.4, 5))
    for j, (mu, ok, f, b, c) in enumerate(zip(values, valid, factors, betas, colors)):
        i = np.arange(len(mu))
        use = (i > 0) & ok & (mu > 0) & (mu < 1)
        lam = -4*b*np.log(mu[use])
        x = i[use]/n
        if len(x) < 2:                                  # no converged mode: nothing to draw
            continue
        d, slope, cut = weyl_dimension(mu, window, ok)
        ax.plot(x, lam, color=c, lw=.8, marker=MARKERS[j % len(MARKERS)], ms=4, mfc="none",
                markevery=np.unique(np.geomspace(1, len(x), 30).astype(int)-1),
                label=f"factor {f:g} ($\\beta$={b:.2g}), d = {d:.1f}")
        if np.isfinite(d):
            ax.plot(x, 4*b*np.exp(cut)*(x*n)**slope, color=c, lw=.9, ls="--")
    _style(ax)
    ax.set(xlabel="n / N", ylabel=r"$\lambda_n = -4\beta\,\ln\mu_n$", title=title)
    ax.legend(fontsize=7, frameon=True, edgecolor="k", fancybox=False, loc="lower right")
    _save(fig, out/"spectrum_weyl")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(np.log10(np.clip(values[:, 1:], FLOOR, None)), aspect="auto", origin="lower",
                   cmap="magma", interpolation="nearest",
                   extent=[.5, values.shape[1]-.5, -.5, len(factors)-.5])
    ax.set_yticks(range(len(factors)), [f"{f:g}" for f in factors])
    ax.set(xlabel="mode k", ylabel=r"$\beta$ factor ($\times$ base $\beta$)", title=title)
    fig.colorbar(im, ax=ax, label=r"$\log_{10}\mu_k$")
    _save(fig, out/"spectrum_heatmap")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for j, (mu, ok, f, c) in enumerate(zip(values, valid, factors, colors)):
        ax.plot(*local_dimension(mu, ok), color=c, lw=.8, marker=MARKERS[j % len(MARKERS)], ms=4,
                mfc="none", label=f"factor {f:g}")
    ax.axvspan(*window, color="0.85", lw=0)
    ax.set(xscale="log", ylim=(0, 40), xlabel="mode i", ylabel="local Weyl d(i) on [i, 2i]", title=title)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(fontsize=7, frameon=True, edgecolor="k", fancybox=False)
    _save(fig, out/"spectrum_local_d")


if __name__ == "__main__":
    if sys.argv[1] == "--plot":
        figures(sys.argv[2])
    else:
        run(json.loads(Path(sys.argv[1]).read_text()))
