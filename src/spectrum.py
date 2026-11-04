"""Eigenvalue spectrum of the frozen DMAP operator for several beta factors.

    python spectrum.py config_spectrum_emnist.json

Same data, anchors (training split) and input standardization as experiment.py, and
beta = factor * base_beta as resolve_beta does, so each curve is the spectrum the encoder of a run
with that beta_factor would truncate.  Writes spectrum.npz and three separate figures (.png/.pdf) to
cfg['out'], see figures().  Weyl's law lambda_i ~ i^(2/d) with lambda_i = -ln mu_i (diffusion time only
shifts the intercept) gives d = 2 / slope of ln lambda_i against ln i over cfg['weyl_window'].
method "dense" is exact; with k "all" it returns every eigenvalue (n/N reaches 1) in float64 on cfg['device']
"cpu" (N <~ 2*10^4); "krylov" is matrix-free, so all_train runs on every training image.
python spectrum.py --sizes <out> <dense_dir>... [--krylov <dir>] overlays runs of several N, see figures_sizes().
"""
import json
from pathlib import Path
import shutil
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
NOISE = 1e-12   # modes at or below this are not drawn nor fitted: float64 eigensolver rounding is ~1e-16 relative, kept 4 decades away


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
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    x, raw_f, truth, split, meta = load_data(cfg)
    u = raw_f[np.concatenate(split[:2]) if cfg.get("all_train") else split[0]].astype(np.float64)
    u = (u-u.mean(0))/max(float(np.sqrt(np.mean((u-u.mean(0))**2))), 1e-12)
    k, factors = cfg.get("k", 1000), [float(f) for f in cfg["factors"]]
    method, window, tol = cfg.get("method", "dense"), tuple(cfg.get("weyl_window", (10, 300))), cfg.get("tol", 1e-4)
    if k == "all":
        if cfg.get("method", "dense") != "dense":
            raise ValueError('k "all" needs method "dense".')
        k = len(u) - 1
    k = int(k)
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
            mu = dmap_spectrum(u, k, f*b0, alpha, backend, device, cfg.get("tile", 4096))
            res = np.zeros_like(mu)
        values.append(mu), residuals.append(res), valid.append((res < tol) & (mu > NOISE))
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
    # y spans whole decades with one labelled tick per decade, so the decades are equally spaced.
    lo, hi = np.log10(ax.get_ylim())
    ax.set_ylim(10**np.floor(lo), 10**np.ceil(hi))
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0e}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=30))
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _save(fig, stem):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
    plt.close(fig)




def _load(out, cfg=None):
    """(npz, config, valid): modes that converged and sit above the noise floor."""
    out = Path(out)
    a = np.load(out/"spectrum.npz")
    cfg = cfg or json.loads((out/"spectrum_meta.json").read_text())["config"]
    res = a["residuals"] if "residuals" in a.files else np.zeros_like(a["eigenvalues"])
    return a, cfg, (res < cfg.get("tol", 1e-4)) & (a["eigenvalues"] > NOISE)


def _curve(ax, mu, b, label, ok, n, window, color, marker, fit=True):
    """Draw lambda_i = -4 b ln mu_i against i/n for one spectrum (label may use {d}); dashed: fitted Weyl
    scaling.  Returns the Weyl dimension on window."""
    i = np.arange(len(mu))
    use = (i > 0) & ok & (mu > 0) & (mu < 1)
    x = i[use]/n
    if len(x) < 2:                                      # no converged mode: nothing to draw
        return np.nan
    d, slope, cut = weyl_dimension(mu, window, ok)
    ax.plot(x, -4*b*np.log(mu[use]), color=color, lw=.8, marker=marker, ms=4, mfc="none",
            markevery=np.unique(np.geomspace(1, len(x), 30).astype(int)-1), label=label.format(d=d))
    if fit and np.isfinite(d):
        ax.plot(x, 4*b*np.exp(cut)*(x*n)**slope, color=color, lw=.9, ls="--")
    return d


def _legend(ax, **kw):
    ax.legend(fontsize=7, frameon=True, edgecolor="k", fancybox=False, **kw)


def figures(out, cfg=None):
    """Separate figures from out/spectrum.npz (also: python spectrum.py --plot <out>):
    spectrum_weyl:    lambda_n = -4 beta ln mu_n against n/N, log-log, one curve per beta, dashed effective
                      Weyl scaling (n/N)^(2/d) fitted on the window.  The Gaussian kernel exp(-beta r^2)
                      with alpha = 1 approximates the heat semigroup at t = 1/(4 beta), so this rescaling
                      makes curves of different beta collapse where the Weyl regime holds.
    spectrum_heatmap: log10 mu_k over beta x k.
    spectrum_local_d: local Weyl dimension on windows [i, 2i].
    Modes that did not converge or are at the noise floor are neither drawn nor fitted."""
    out = Path(out)
    a, cfg, valid = _load(out, cfg)
    values, factors, betas = a["eigenvalues"], a["factors"], a["betas"]
    n = int(a["n_anchors"])
    window = tuple(cfg.get("weyl_window", (10, 300)))
    colors = plt.cm.viridis(np.linspace(0, .9, len(factors)))
    title = f"{cfg['dataset']}{'/'+cfg['split'] if cfg.get('split') else ''}, N = {n}"

    fig, ax = plt.subplots(figsize=(6.4, 5))
    for j, (mu, ok, f, b, c) in enumerate(zip(values, valid, factors, betas, colors)):
        _curve(ax, mu, b, f"factor {f:g} ($\\beta$={b:.2g}), d = {{d:.1f}}", ok, n, window, c, MARKERS[j % len(MARKERS)])
    _style(ax)
    ax.set(xlabel="n / N", ylabel=r"$\lambda_n = -4\beta\,\ln\mu_n$", title=title)
    _legend(ax, loc="lower right")
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
    _legend(ax)
    _save(fig, out/"spectrum_local_d")


def figures_sizes(dirs, out, krylov=None):
    """Sizes overlaid.  dirs: output folders of dense runs at different N; krylov: optional folder of a
    larger Krylov run (leading k modes only).  Writes to out, same style as spectrum_weyl:
    spectrum_sizes_f<factor>: one curve per N against n/N for every beta factor of the dense runs (matched
                              by value), each with its own beta; dashed Weyl fit for the largest dense N only.
    spectrum_full:            all factors of the largest dense N (its spectrum_weyl figure)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(dirs, key=lambda d: int(np.load(Path(d)/"spectrum.npz")["n_anchors"]))
    runs = [_load(d) for d in dirs] + ([_load(krylov)] if krylov else [])
    colors = plt.cm.viridis(np.linspace(0, .9, len(runs)))
    for f in sorted({float(g) for a, _, _ in runs[:len(dirs)] for g in a["factors"]}):
        fig, ax = plt.subplots(figsize=(6.4, 5))
        has = [np.isclose(a["factors"], f).any() for a, _, _ in runs]
        last = max(j for j in range(len(dirs)) if has[j])
        for j, (a, cfg, valid) in enumerate(runs):
            if not has[j]:
                continue
            r, n = int(np.argmax(np.isclose(a["factors"], f))), int(a["n_anchors"])
            label = f"N = {n}, d = {{d:.1f}}" if j < len(dirs) else f"N = {n} (Krylov, k = {int(a['k'])})"
            _curve(ax, a["eigenvalues"][r], a["betas"][r], label, valid[r], n,
                   tuple(cfg.get("weyl_window", (10, 300))), colors[j], MARKERS[j % len(MARKERS)], fit=j == last)
        _style(ax)
        ax.set(xlabel="n / N", ylabel=r"$\lambda_n = -4\beta\,\ln\mu_n$",
               title=f"{runs[0][1]['dataset']}, factor {f:g}")
        _legend(ax, loc="lower right")
        _save(fig, out/f"spectrum_sizes_f{f:g}")
    figures(dirs[-1])
    for ext in ("png", "pdf"):
        shutil.copy(Path(dirs[-1])/f"spectrum_weyl.{ext}", out/f"spectrum_full.{ext}")


if __name__ == "__main__":
    if sys.argv[1] == "--plot":
        figures(sys.argv[2])
    elif sys.argv[1] == "--sizes":
        args = sys.argv[2:]
        krylov = args.pop(args.index("--krylov")+1) if "--krylov" in args else None
        if krylov:
            args.remove("--krylov")
        figures_sizes(args[1:], args[0], krylov)
    else:
        run(json.loads(Path(sys.argv[1]).read_text()))
