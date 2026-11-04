"""Running (scale-dependent) dimension d(E) of the frozen DMAP operator over a log-spaced beta grid.

    python dimension.py config_dimension_emnist.json
    python dimension.py --plot <out>

K_ij = exp(-beta |x_i-x_j|^2), P = D^-1 K (alpha = 0), E_j = 4 beta (1 - lambda_j) from the FULL spectrum
(dense float64, so no E_k/3 truncation rule is needed).  betas are solved in increasing order and the scan
stops at the first fissioned one (gap 1 - lambda_1 <= 1e-8); beta* is the largest beta before it.  Estimators:
  A  kernel sum S(beta) = sum_{i != j} K_ij, d = -2 dlogS/dlogbeta at E = 4 beta (no eigensolve, denser grid);
  B  heat trace Z(tau) = sum_j exp(-tau E_j), d = -2 dlogZ/dlogtau at E = 1/tau;
  C  Weyl counting N(E) = #{j >= 1: E_j <= E}, d = 2 dlogN/dlogE, smoothed.
B and C keep only E <= trust*beta (small betas may have no trusted level: their energies cluster near 4 beta,
so they only contribute through A).  Expected shape: d tapers to 0 at the lowest energies, plateaus with steps
up (the manifold's dimension at each scale), and rises or collapses near beta*.  Curves of different beta must
agree where their trusted ranges overlap.
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
from spectral import dmap_spectrum
from spectrum import _save


def log_slope(x, y, smooth=5):
    s = np.gradient(np.log(y), np.log(x))
    return np.convolve(s, np.ones(smooth)/smooth, mode="same")


def kernel_sums(u, betas, tile=4096):
    """S(beta) = sum_{i != j} exp(-beta |u_i-u_j|^2) for every beta in one tiled float64 pass."""
    u = torch.as_tensor(np.asarray(u, dtype=np.float64))
    betas = np.asarray(betas, dtype=np.float64)
    s, sq = np.zeros(len(betas)), u.square().sum(1)
    for i in range(0, len(u), tile):
        for j in range(0, len(u), tile):
            d2 = (sq[i:i+tile, None] + sq[None, j:j+tile] - 2*u[i:i+tile] @ u[j:j+tile].T).clamp_min_(0)
            s += [torch.exp(-b*d2).sum().item() for b in betas]
    return s - len(u)                       # the diagonal contributes exp(0) = 1 per point


def _trim(a, smooth):
    h = smooth//2                           # np.convolve(mode="same") zero-pads: edge points are biased
    return a[h:len(a)-h]


def estimator_b(E, beta, smooth=5, points=200):
    """(E = 1/tau ascending, d) from the heat trace over taus in [1/(4 beta), 1e3/(4 beta)]."""
    tau = np.geomspace(1/(4*beta), 1e3/(4*beta), points)
    z = np.exp(-tau[:, None]*np.asarray(E)[None]).sum(1)
    k = z > 0
    return _trim(1/tau[k], smooth)[::-1], _trim(-2*log_slope(tau[k], z[k], smooth), smooth)[::-1]


def estimator_c(E, smooth=5, points=200):
    """(E grid, d, N(E)) from the counting function over the positive energies."""
    e = np.sort(E[E > 0])
    g = np.geomspace(e[0], e[-1], points)
    n = np.searchsorted(e, g, side="right")  # >= 1 since g >= e[0]
    return _trim(g, smooth), _trim(2*log_slope(g, n, smooth), smooth), _trim(n, smooth)


def solve(u, betas, tile=4096, tol=1e-8):
    """Yield (beta, E, gap) for increasing betas, stopping after the first fissioned one (gap <= tol)."""
    for b in betas:
        lam = dmap_spectrum(u, len(u)-1, b, alpha=0., backend="torch", device="cpu", tile=tile)
        e = 4*b*(1-lam)
        e[0] = 0.                           # Perron mode: exactly 0, not rounding noise
        yield b, e, 1-lam[1]
        if 1-lam[1] <= tol:
            return


def trusted(E, beta, cfg):
    """B and C as (E, d) and (E, d, n) restricted to E <= trust*beta (arrays may be empty)."""
    smooth, cut = cfg.get("smooth", 5), cfg.get("trust", 1.)*beta
    (eb, db), (ec, dc, nc) = estimator_b(E, beta, smooth), estimator_c(E, smooth)
    kb, kc = eb <= cut, ec <= cut
    return (eb[kb], db[kb]), (ec[kc], dc[kc], nc[kc])


def run(cfg):
    x, raw_f, truth, split, meta = load_data(cfg)
    u = raw_f[np.concatenate(split[:2]) if cfg.get("all_train") else split[0]].astype(np.float64)
    u = (u-u.mean(0))/max(float(np.sqrt(np.mean((u-u.mean(0))**2))), 1e-12)
    b0 = base_beta(u)
    betas = np.geomspace(cfg.get("beta_min_factor", .1)*b0, cfg.get("beta_max", 10.), cfg.get("n_betas", 32))
    print(f"base beta = {b0:.4g}, grid = {np.array2string(betas, precision=3)}", flush=True)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    kept, energies, gaps, table = [], [], [], []
    started = time.perf_counter()
    for b, e, gap in solve(u, betas, cfg.get("tile", 4096)):
        n_trusted = int(np.sum(e[1:] <= cfg.get("trust", 1.)*b))
        print(f"beta={b:.4g} gap={gap:.3e} trusted levels={n_trusted} ({time.perf_counter()-started:.0f} s)", flush=True)
        started = time.perf_counter()
        if gap > 1e-8:
            kept.append(b), energies.append(e), gaps.append(gap)
            dc = trusted(e, b, cfg)[1][1]
            table.append(dict(beta=b, gap=gap, trusted_levels=n_trusted, median_c_d=float(np.median(dc)) if len(dc) else None))
    if not kept:
        raise ValueError("every beta of the grid is fissioned: lower beta_max or beta_min_factor.")
    star = kept[-1]
    print(f"beta* = {star:.4g} ({len(kept)} of {len(betas)} betas below fission)", flush=True)
    # A needs no eigensolve, so it also covers the small betas whose levels are all untrusted (the low-E taper).
    a_betas = np.geomspace(cfg.get("a_min_factor", .01)*b0, betas[-1], cfg.get("a_points", 64))
    a_betas = a_betas[a_betas <= star]
    a_sums = kernel_sums(u, a_betas, cfg.get("tile", 4096))
    np.savez_compressed(out/"dimension.npz", betas=kept, energies=np.array(energies), gaps=gaps, beta_star=star,
                        a_betas=a_betas, a_sums=a_sums, b0=b0, n_anchors=len(u))
    (out/"dimension_meta.json").write_text(json.dumps({"config": cfg, "data": meta, "n_anchors": len(u), "beta_star": star,
                                                       "betas": table}, indent=2, default=str))
    figures(out, cfg)
    return star


def _style(ax, xlabel, ylabel, title):
    ax.set(xscale="log", xlabel=xlabel, ylabel=ylabel, title=title)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def figures(out, cfg=None):
    """Figures from out/dimension.npz (also: python dimension.py --plot <out>):
    dimension_n: d against n = N(E) (levels below E, not normalized), one curve per beta (C solid, B dashed,
                 both on trusted energies), colour by log beta, as in the hand sketch.
    dimension_E: d against E, A (one point per beta, E = 4 beta), B and C (one thin line per beta)."""
    out = Path(out)
    a = np.load(out/"dimension.npz")
    cfg = cfg or json.loads((out/"dimension_meta.json").read_text())["config"]
    betas, smooth = a["betas"], cfg.get("smooth", 5)
    title = f"{cfg['dataset']}{'/'+cfg['split'] if cfg.get('split') else ''}, N = {int(a['n_anchors'])}, $\\beta^*$ = {float(a['beta_star']):.3g}"
    lb = np.log(betas)
    colors = plt.cm.viridis(.9*(lb-lb.min())/max(np.ptp(lb), 1e-12))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.6))
    for b, e, c in zip(betas, a["energies"], colors):
        (eb, db), (ec, dc, nc) = trusted(e, b, cfg)
        e_pos = np.sort(e[e > 0])
        nb = np.searchsorted(e_pos, eb, side="right")
        if len(ec):
            ax.plot(nc, dc, color=c, lw=1, label=f"{b:.2g}")
            ax2.plot(ec, dc, color="tab:green", lw=.5, alpha=.7, label="C" if b == betas[0] else None)
        if len(eb):
            ax.plot(nb[nb > 0], db[nb > 0], color=c, lw=.6, ls="--")
            ax2.plot(eb, db, color="tab:orange", lw=.5, alpha=.7, label="B" if b == betas[0] else None)
    ab = a["a_betas"]
    ok = a["a_sums"] > 0
    ea, da = _trim(4*ab[ok], smooth), _trim(-2*log_slope(ab[ok], a["a_sums"][ok], smooth), smooth)
    ax2.plot(ea, da, "o-", color="k", ms=3, lw=.8, label="A")
    _style(ax, "n = N(E)", "d", title)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(title=r"$\beta$", fontsize=5, ncol=2, frameon=True, edgecolor="k", fancybox=False)
    _style(ax2, "E", "d(E)", title)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=7, frameon=True, edgecolor="k", fancybox=False)
    _save(fig, out/"dimension_n")
    _save(fig2, out/"dimension_E")


if __name__ == "__main__":
    if sys.argv[1] == "--plot":
        figures(sys.argv[2])
    else:
        run(json.loads(Path(sys.argv[1]).read_text()))
