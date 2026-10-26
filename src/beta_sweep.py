"""Bandwidth sweep for the frozen Gaussian DMAP.

For every (data seed, beta factor) it fits the SAME spectral map used by experiment.py
(beta = factor * median heuristic on the standardized training inputs) and records
  label-free:  graph short-circuit mass, beta-flow dimension, spectral gates;
  diagnostic (uses the true chart, NEVER used for selection): true short-circuit mass,
               psi_1 ~ cos(pi s/W), height-mode position, affine-oracle chart MSE
               (validation and test), oracle -> exact roll ambient MSE and its bound.
Selection uses label-free quantities only; the rule is cfg["selection"]:
  "graph"       (Swiss roll)  smallest factor with worst-seed graph short-circuit mass <= threshold
  "degeneracy"  (flat torus)  argmin worst-seed degeneracy defect of the (cos, sin) eigenpairs
  "ambient"     (MNIST, alanine) argmin worst-seed validation MSE of the affine readout of x from
                psi_1..psi_m, averaged over cfg["selection_modes"] (uses x, never labels/charts)

usage:  python beta_sweep.py CONFIG --launch                     (both GPUs, then merge)
        python beta_sweep.py CONFIG --worker 0 --workers 2      (one process per GPU)
        python beta_sweep.py CONFIG --merge                      (table, selection, figure)
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from data import load_data
from geometry import (graph_shortcut_mass, true_shortcut_mass, beta_flow, mode_diagnostics,
                      oracle_exact_swiss, affine_fit, chart_target, select_beta, base_beta,
                      degeneracy_defect, select_beta_argmin, knn_label_accuracy)

CHART_DATASETS = ("swiss", "swissvol", "torus", "alanine")


def standardized_inputs(cfg, data_seed):
    x, raw_f, truth, split, meta = load_data(dict(cfg, data_seed=data_seed))
    train = split[0]
    fm = raw_f[train].mean(0)
    fs = max(float(np.sqrt(np.mean((raw_f[train]-fm)**2))), 1e-12)
    xm = x[train].mean(0)
    xs = max(float(np.sqrt(np.mean((x[train]-xm)**2))), 1e-12)
    return (raw_f-fm)/fs, fs, truth, split, meta, (x-xm)/xs


def run_job(cfg, data_seed, factors, device):
    from spectral import SpectralMap   # torch import only where needed
    inputs, fs, truth, (train, val, test), meta, target = standardized_inputs(cfg, data_seed)
    u = inputs[train].astype(np.float64)
    b0 = base_beta(u)
    betas = [f*b0 for f in factors]
    started = time.perf_counter()
    graph = graph_shortcut_mass(u, betas, k=cfg.get("graph_k", 10), ratio=cfg.get("graph_ratio", 2.))
    flow = beta_flow(u, b0*np.exp(np.linspace(np.log(min(factors)/2), np.log(max(factors)*2), 60)))
    flow["factors"] = (np.asarray(flow["betas"])/b0).tolist()
    swiss = cfg["dataset"] == "swiss"
    true_mass = true_shortcut_mass(u, truth[train], betas, fs) if swiss else [None]*len(betas)
    y = chart_target(cfg["dataset"], truth) if cfg["dataset"] in CHART_DATASETS else None
    mnist = cfg["dataset"] == "mnist"
    rows = []
    for f, beta, gm, tm in zip(factors, betas, graph["graph_shortcut_mass"], true_mass):
        row = dict(data_seed=data_seed, factor=f, beta=beta, base_beta=b0,
                   kernel_length=fs/np.sqrt(beta), graph_shortcut_mass=gm, true_shortcut_mass=tm)
        t0 = time.perf_counter()
        try:
            sm = SpectralMap(max(cfg["modes"]), beta, cfg.get("alpha", 1.), cfg.get("backend", "flash"),
                             device, tolerance=cfg.get("spectral_tolerance", 2e-3)).fit(inputs[train])
            feats = sm.transform(inputs)
            feats[train] = sm.psi
            row.update(ok=True, **{k: v for k, v in sm.diagnostics.items() if k not in ("beta", "backend")})
            row["eigenvalues_head"] = [float(v) for v in sm.values[:16]]
            row["degeneracy_defect"] = degeneracy_defect(sm.values, cfg.get("degeneracy_pairs", 3))
            if swiss:
                row.update(mode_diagnostics(sm.psi, truth[train], meta["width"], meta["height"]))
            for m in cfg["modes"]:
                # label-free: affine readout of the (standardized) data from the prefix
                pa = affine_fit(feats[:, :m], target, train, cfg.get("ridge", 1e-6))
                amb = lambda ids: float(np.mean((pa[ids]-target[ids])**2))
                row[f"ambient_val_m{m}"], row[f"ambient_test_m{m}"] = amb(val), amb(test)
                if mnist:   # diagnostic, uses labels
                    row[f"knn_label_acc_m{m}"] = knn_label_accuracy(feats[train, :m], truth[train],
                                                                    feats[val, :m], truth[val])
                if y is None:
                    continue
                pred = affine_fit(feats[:, :m], y, train, cfg.get("ridge", 1e-6))
                err = lambda ids: float(np.mean(np.sum((pred[ids]-y[ids])**2, 1))/y.shape[1])
                row[f"oracle_val_m{m}"], row[f"oracle_test_m{m}"] = err(val), err(test)
                if swiss:
                    ex = oracle_exact_swiss(pred[test], truth[test], meta["width"], meta["height"])
                    row[f"oracle_exact_x_m{m}"] = ex["oracle_exact_mse_x"]
                    row[f"oracle_lipschitz_m{m}"] = ex["oracle_lipschitz_ratio"]
        except (RuntimeError, ValueError, FloatingPointError) as e:   # gates are data, not crashes
            row.update(ok=False, error=str(e)[:300])
        row["seconds"] = time.perf_counter()-t0
        rows.append(row)
        print(f"seed={data_seed} factor={f}: ok={row['ok']} graph={gm:.2e} "
              f"{'' if not row['ok'] else 'defect=%.3g ambient_val_m%d=%.3g' % (row['degeneracy_defect'], cfg['modes'][-1], row['ambient_val_m%d' % cfg['modes'][-1]])}", flush=True)
    return dict(data_seed=data_seed, rows=rows, flow=flow, graph=dict((k, v) for k, v in graph.items() if k != "graph_shortcut_mass"),
                diagnostic_seconds=time.perf_counter()-started, meta=meta)


def merge(cfg):
    import pandas as pd
    out = Path(cfg["out"])
    parts = [json.loads(p.read_text()) for p in sorted(out.glob("sweep_seed*.json"))]
    if not parts:
        raise FileNotFoundError(f"No sweep results in {out}")
    rows = [r for p in parts for r in p["rows"]]
    df = pd.DataFrame(rows)
    df.drop(columns=[c for c in ("eigenvalues_head",) if c in df]).to_csv(out/"beta_sweep.csv", index=False)
    rule = cfg.get("selection", "graph")
    if rule == "graph":
        selection = select_beta(rows, cfg.get("threshold", 1e-3))
    elif rule == "degeneracy":
        selection = select_beta_argmin(rows, "degeneracy_defect", band=cfg.get("band", 2.),
                                       sanity=cfg.get("sanity", .5))
        selection["note_criterion"] = ("Flat torus: fold criterion vacuous (geodesic/chord <= pi/2 < 2); "
                                       "reported, not used.")
    elif rule == "ambient":
        sel_modes = cfg.get("selection_modes", cfg["modes"])
        for r in rows:
            vals = [r.get(f"ambient_val_m{m}") for m in sel_modes]
            r["ambient_val_selection"] = (float(np.mean(vals)) if r.get("ok") and all(v is not None for v in vals)
                                          else None)
        df["ambient_val_selection"] = [r["ambient_val_selection"] for r in rows]
        df.drop(columns=[c for c in ("eigenvalues_head",) if c in df]).to_csv(out/"beta_sweep.csv", index=False)
        selection = select_beta_argmin(rows, "ambient_val_selection", band=cfg.get("band", 1.05))
        selection["selection_modes"] = list(sel_modes)
    else:
        raise ValueError(f"selection must be graph | degeneracy | ambient, not {rule!r}")
    selection["selection"] = rule
    seeds = sorted(df.data_seed.unique())
    selection.update(data_seeds=[int(s) for s in seeds], dataset=cfg["dataset"], n=cfg.get("n"),
                     note="Label-free selection. Truth-based columns are diagnostics only.")
    if cfg["dataset"] == "swiss" and selection["selected_factor"] is not None:
        sel = df[df.factor == selection["selected_factor"]]
        selection["diagnostics_at_selected"] = {
            k: float(sel[k].mean()) for k in ("psi1_corr_cos_s", "height_mode_position", "true_shortcut_mass",
                                              "graph_shortcut_mass", "kernel_length") if k in sel}
        # For transparency only: the factor a truth-based rule would have chosen per m.
        selection["truth_best_factor_by_validation_oracle"] = {
            str(m): float(df[df.ok == True].groupby("factor")[f"oracle_val_m{m}"].mean().idxmin())
            for m in cfg["modes"] if f"oracle_val_m{m}" in df}
    (out/"beta_selection.json").write_text(json.dumps(selection, indent=2))
    plot(df, parts, cfg, selection, out)
    print(json.dumps(selection, indent=2))
    return df, selection


def plot(df, parts, cfg, selection, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = df[df.ok == True]
    g = lambda col, d=df: d.groupby("factor")[col].agg(["mean", "min", "max"])
    swiss = cfg["dataset"] == "swiss"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    s = g("graph_shortcut_mass")
    ax.fill_between(s.index, s["min"]+1e-12, s["max"]+1e-12, alpha=.25)
    ax.plot(s.index, s["mean"]+1e-12, "o-", label="kNN-graph (label-free)")
    if swiss:
        t = g("true_shortcut_mass")
        ax.plot(t.index, t["mean"]+1e-18, "s--", label="true chart (diagnostic)")
    ax.axhline(cfg.get("threshold", 1e-3), color="k", lw=.8, ls=":", label="threshold")
    ax.set(xscale="log", yscale="log", xlabel=r"$\beta$ factor", ylabel="kernel mass across folds",
           title="Short-circuit mass")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for p in parts:
        ax.plot(p["flow"]["factors"], p["flow"]["dimension_full"], color="C0", alpha=.5)
        ax.plot(p["flow"]["factors"], p["flow"]["dimension_offdiag"], color="C1", alpha=.5)
    ax.plot([], [], color="C0", label="with diagonal")
    ax.plot([], [], color="C1", label="off-diagonal")
    ax.set(xscale="log", xlabel=r"$\beta$ factor", ylabel=r"$2\,d\log S/d\log(1/\beta)$",
           title=r"$\beta$-flow (reported, not used)")
    ax.legend(fontsize=8)
    ax = axes[0, 2]
    if swiss and len(ok):
        c = g("psi1_corr_cos_s", ok)
        ax.plot(c.index, c["mean"], "o-", label=r"|corr($\psi_1$, cos $\pi s/W$)|")
        ax2 = ax.twinx()
        h = g("height_mode_position", ok)
        ax2.plot(h.index, h["mean"], "s--", color="C3", label="height-mode position")
        exp = ok["expected_height_mode_position"].iloc[0]
        ax2.axhline(exp, color="C3", lw=.8, ls=":")
        ax2.set_ylabel("height-mode position (dotted: intrinsic)")
        ax.legend(fontsize=8, loc="lower right")
    elif cfg["dataset"] == "mnist" and len(ok):
        for m in cfg["modes"]:
            col = f"knn_label_acc_m{m}"
            if col in ok:
                s = g(col, ok)
                ax.plot(s.index, s["mean"], "o-", ms=3, label=f"m={m}")
        ax.set_ylabel("kNN digit accuracy of the prefix (validation)")
        ax.legend(fontsize=7, ncol=2)
    ax.set(xscale="log", xlabel=r"$\beta$ factor", title="Diagnostics (use truth; never for selection)")
    ax = axes[1, 0]
    for m in cfg["modes"]:
        col = f"oracle_val_m{m}"
        if col in ok:
            s = g(col, ok)
            ax.errorbar(s.index, s["mean"], yerr=[s["mean"]-s["min"], s["max"]-s["mean"]], marker="o", ms=3, label=f"m={m}")
    ax.set(xscale="log", yscale="log", xlabel=r"$\beta$ factor", ylabel="chart MSE (validation)",
           title="Affine oracle (diagnostic)")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1, 1]
    if swiss:
        for m in cfg["modes"]:
            col = f"oracle_exact_x_m{m}"
            if col in ok:
                s = g(col, ok)
                ax.plot(s.index, s["mean"], "o-", ms=3, label=f"m={m}")
        ax.set(xscale="log", yscale="log", xlabel=r"$\beta$ factor", ylabel="ambient MSE (test)",
               title=r"Oracle $\to$ exact roll $g(\Pi_\Omega\hat z)$")
        ax.legend(fontsize=7, ncol=2)
    elif len(ok):
        rule = selection.get("selection", "graph")
        if rule == "degeneracy":
            s = g("degeneracy_defect", ok)
            ax.fill_between(s.index, s["min"], s["max"], alpha=.25)
            ax.plot(s.index, s["mean"], "o-", ms=4, label="degeneracy defect (label-free)")
            ax.set(yscale="log", ylabel="within-pair split / between-pair gap")
        else:
            for m in cfg["modes"]:
                col = f"ambient_val_m{m}"
                if col in ok:
                    s = g(col, ok)
                    ax.plot(s.index, s["mean"], "o-", ms=3, label=f"m={m}")
            ax.set(yscale="log", ylabel="affine readout of x: validation MSE (standardized)")
        ax.set(xscale="log", xlabel=r"$\beta$ factor", title=f"Label-free criterion ({rule})")
        ax.legend(fontsize=7, ncol=2)
    ax = axes[1, 2]
    if len(ok):
        for col, lab in (("perron_gap", "Perron gap"), ("max_residual", "max residual"),
                         ("nystrom_training_relative_error", "Nyström train rel. error"),
                         ("min_eigenvalue", r"min retained $\lambda$")):
            if col in ok:
                s = g(col, ok)
                ax.plot(s.index, s["mean"], "o-", ms=3, label=lab)
    failed = df[df.ok != True].groupby("factor").size()
    for f in failed.index:
        ax.axvline(f, color="r", alpha=.3)
    ax.set(xscale="log", yscale="log", xlabel=r"$\beta$ factor", title="Spectral gates (red: a seed failed)")
    ax.legend(fontsize=7)
    sel = selection.get("selected_factor")
    for a in axes.ravel():
        if sel is not None:
            a.axvline(sel, color="k", lw=1.2, alpha=.6)
    fig.suptitle(f"{cfg['dataset']}  n={cfg.get('n')}  data seeds={len(parts)}  "
                 f"selected factor={sel} (black line, label-free rule)")
    fig.tight_layout()
    fig.savefig(out/"beta_sweep.png", dpi=150, bbox_inches="tight")
    fig.savefig(out/"beta_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def launch(config, gpus=(0, 1)):
    """Compile the extension once, run one worker per GPU (logs in out/), then merge."""
    import os
    import subprocess
    import sys
    config = Path(config).resolve()
    cfg = json.loads(config.read_text())
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    if cfg.get("backend", "flash") == "flash":
        build = subprocess.run([sys.executable, "-c", "import spectral; spectral.flash_extension()"],
                               cwd=here, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[0])),
                               capture_output=True, text=True)
        if build.returncode:
            raise RuntimeError("FlashDiffusion compilation failed:\n" + build.stderr[-6000:])
    jobs = []
    for w, gpu in enumerate(gpus):
        log = open(out/f"sweep_worker{w}.log", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        jobs.append((w, log, subprocess.Popen([sys.executable, str(here/"beta_sweep.py"), str(config),
                                               "--worker", str(w), "--workers", str(len(gpus))],
                                              cwd=here, env=env, stdout=log, stderr=subprocess.STDOUT)))
    failed = []
    for w, log, job in jobs:
        code = job.wait()
        log.close()
        if code:
            failed.append((w, (out/f"sweep_worker{w}.log").read_text()[-4000:]))
    if failed:
        raise RuntimeError(f"Sweep worker(s) failed: {failed}")
    return merge(cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--launch", action="store_true")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    if a.launch:
        launch(a.config)
    elif a.merge:
        merge(cfg)
    else:
        import torch
        device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu" and cfg.get("backend") == "flash":
            raise RuntimeError("flash backend needs a GPU; use backend='torch' for CPU runs.")
        for seed in cfg["data_seeds"][a.worker::a.workers]:
            result = run_job(cfg, seed, cfg["factors"], device)
            (out/f"sweep_seed{seed}.json").write_text(json.dumps(result, indent=1))
