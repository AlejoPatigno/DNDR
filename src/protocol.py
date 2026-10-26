"""Shared protocol for the notebooks 02b/03b/04b (same rules as the Swiss-roll closing, notebook 05).

  * a PLAN is a list of dicts; each dict overrides the dataset config + COMMON; `tag` names the
    output folder results_<dataset>_<tag>.  Order = priority.  Completed folders are skipped,
    partial folders are reported and never overwritten.
  * chart recovery is reported as distributions (kNN ratio) and as unfolding RATES with
    Clopper-Pearson 95 % intervals; order ablations are tested at m = M (same modes, same class).
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COMMON = dict(epochs=2500, batch=512, min_steps=10_000, tau=1.0, kl=1e-3, resample_data=True,
              knn_k=5, unfold_ratio=10.0, restarts=1, backend="flash")
PLAN_ONLY = ("tag", "eta_min", "note")


# ----------------------------------------------------------------------------- planning
def models_per_seed(p, common=COMMON):
    """Trainings per seed in units of one model at COMMON['epochs'] (the chart decoder counts
    chart_decoder_epochs / epochs; plainvae is trained once, at the first budget)."""
    kinds = list(p["models"])
    per = 1 if p.get("scratch_first_only") and "warm" in p.get("m_protocols", ["scratch"]) else len(p["modes"])
    n = sum(per if k in ("ae", "vae", "direct") else len(p["modes"]) for k in kinds if k not in ("plainvae", "plainae")) \
        + sum(k in kinds for k in ("plainvae", "plainae"))
    if "warm" in p.get("m_protocols", ["scratch"]):
        n += (len(p["modes"]) - 1) * len([k for k in kinds if k in ("ae", "vae", "direct")])
    if p.get("dataset", "swiss") != "mnist" and p.get("oracle_decoder", True):
        n += p.get("chart_decoder_epochs", common["epochs"]) / common["epochs"]
    return n * common.get("restarts", 1)


def show_plan(plan, dataset, seconds_per_model, common=COMMON, gpus=2):
    for p in plan:
        p.setdefault("out", f"/kaggle/working/results_{dataset}_{p['tag']}")
        p["eta_min"] = math.ceil(len(p["seeds"]) / gpus) * models_per_seed(dict(p, dataset=dataset), common) \
            * seconds_per_model / 60
    head = f"{'configuración':22s}{'beta':>7s}{'lat':>5s}{'orden':>10s}{'protocolos':>16s}{'semillas':>9s}{'modelos':>9s}{'ETA min':>9s}"
    print(head)
    for p in plan:
        print(f"{p['tag']:22s}{p['beta_factor']:7g}{p.get('latent', 2):5d}{p.get('mode_order', 'spectral'):>10s}"
              f"{'+'.join(p.get('m_protocols', ['scratch'])):>16s}{len(p['seeds']):9d}"
              f"{models_per_seed(dict(p, dataset=dataset), common):9.0f}{p['eta_min']:9.0f}")
    total = sum(p["eta_min"] for p in plan)
    print(f"\nTOTAL estimado: {total/60:.1f} h (ajusta SECONDS_PER_MODEL con el tiempo real de la primera configuración)")
    return total


def reuse_inputs(plan, roots=("/kaggle/input",)):
    """If a completed output folder with the same name is attached as Input (e.g. the output of an
    earlier version of this notebook), point the plan entry to it so it is not recomputed."""
    for p in plan:
        name = Path(p["out"]).name
        if (Path(p["out"])/"metrics.csv").exists():
            continue
        for r in roots:
            hits = sorted(Path(r).rglob(f"{name}/metrics.csv")) if Path(r).exists() else []
            if hits:
                p["out"] = str(hits[0].parent)
                print(f"{p['tag']:22s} reutilizado de {p['out']}")
                break
    return plan


def run_plan(plan, base, root, common=COMMON, dataset=None):
    """Resumable: complete configurations are skipped; partial ones are reported, not touched."""
    root = Path(root)
    done, problems = [], []
    for p in plan:
        out = Path(p["out"])
        if (out / "metrics.csv").exists():
            print(f"{p['tag']:22s} ya completo, se omite")
            done.append(p["tag"])
            continue
        if any(out.glob("seed_*/metrics.json")) or any(out.glob("seed_*/metrics.partial.json")):
            print(f"{p['tag']:22s} PARCIAL en {out}: bórralo o cambia el tag")
            problems.append((p["tag"], "parcial"))
            continue
        cfg = {**base, **common, **{k: v for k, v in p.items() if k not in PLAN_ONLY}}   # plan overrides COMMON
        path = root / f"active_{dataset or base['dataset']}_{p['tag']}.json"
        path.write_text(json.dumps(cfg, indent=2))
        t0 = time.time()
        r = subprocess.run([sys.executable, str(root / "launch.py"), str(path)] + (["--cpu"] if cfg.get("allow_cpu") else []),
                           capture_output=True, text=True)
        print(f"{p['tag']:22s} {(time.time()-t0)/60:6.1f} min (ETA {p.get('eta_min', float('nan')):.0f})", flush=True)
        if r.returncode:
            print(r.stdout[-2000:])
            print(r.stderr[-3000:])
            problems.append((p["tag"], "falló"))
        else:
            done.append(p["tag"])
    print("\nhechas:", done, "| problemas:", problems)
    return done, problems


# ----------------------------------------------------------------------------- beta
def run_sweep(cfg, root, name):
    """Run beta_sweep.py (both GPUs) unless its selection already exists; returns the selection."""
    out = Path(cfg["out"])
    if not (out / "beta_selection.json").exists():
        active = Path(root) / f"active_{name}.json"
        active.write_text(json.dumps(cfg, indent=2))
        t0 = time.time()
        args = ["--launch"] if cfg.get("backend", "flash") == "flash" else []
        if args:
            r = subprocess.run([sys.executable, str(Path(root) / "beta_sweep.py"), str(active), *args],
                               capture_output=True, text=True)
        else:   # local/CPU test path: one worker then merge
            r = subprocess.run([sys.executable, str(Path(root) / "beta_sweep.py"), str(active), "--device", "cpu"],
                               capture_output=True, text=True)
            if not r.returncode:
                r = subprocess.run([sys.executable, str(Path(root) / "beta_sweep.py"), str(active), "--merge"],
                                   capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:])
            print(r.stderr[-6000:])
            for log in sorted(out.glob("sweep_worker*.log")):
                print("---", log.name)
                print(log.read_text()[-3000:])
            raise RuntimeError(f"Falló el barrido de beta ({name}).")
        print(f"Barrido {name}: {(time.time()-t0)/60:.1f} min")
    return json.loads((out / "beta_selection.json").read_text())


def neighbors(factors, selected, band, k=3):
    """Candidates for the confirmation stage: the selected factor and its nearest listed
    neighbours, preferring the equivalence band.  At most k."""
    factors = sorted(factors)
    i = factors.index(selected)
    cand = [selected] + [f for f in band if f != selected]
    for j in (i-1, i+1, i-2, i+2):
        if 0 <= j < len(factors) and factors[j] not in cand:
            cand.append(factors[j])
    ordered = sorted(cand[:k])
    return ordered


def spread(sweep, k=4):
    """Stage-2 candidates spread over the WHOLE admissible grid (every data seed passed the spectral
    gates), log-evenly, instead of neighbours of the stage-1 pick: the affine readout of stage 1
    favours wide kernels (nearly linear eigenfunctions), so its pick must not restrict stage 2."""
    ok = sweep.groupby("factor").ok.all()
    f = np.array(sorted(float(v) for v in ok.index[ok.values]))
    if len(f) <= k:
        out = f
    else:
        targets = np.exp(np.linspace(np.log(f[0]), np.log(f[-1]), k))
        out = sorted({f[np.argmin(np.abs(np.log(f)-np.log(t)))] for t in targets})
    return [int(v) if float(v).is_integer() else float(v) for v in out]


def confirm_direct(results, modes):
    """Stage 2 (label-free): mean validation MSE of the `direct` decoder over seeds and modes."""
    rows = []
    for f, out in results.items():
        d = load(out)
        if d is None:
            continue
        q = trained(d)
        q = q[(q.model == "direct") & q.m.isin(modes)]
        rows.append(dict(factor=f, direct_validation_mse_x=q.validation_mse_x.mean(),
                         worst_seed=q.groupby("seed").validation_mse_x.mean().max(), n=len(q)))
    t = pd.DataFrame(rows).sort_values("factor")
    if not len(t):
        return t, None
    best = float(t.loc[t.direct_validation_mse_x.idxmin(), "factor"])
    if len(t) > 1 and best in (float(t.factor.min()), float(t.factor.max())):
        print(f"ATENCIÓN: el factor elegido ({best:g}) está en el borde de los candidatos "
              f"[{t.factor.min():g}, {t.factor.max():g}]; el óptimo podría estar fuera: amplía la lista de factores.")
    return t, (int(best) if best.is_integer() else best)


# ----------------------------------------------------------------------------- analysis
def load(out):
    csv = Path(out) / "metrics.csv"
    if not csv.exists():
        return None
    d = pd.read_csv(csv)
    d["restart"] = d["restart"].fillna(0).astype(int) if "restart" in d else 0
    return d


def trained(d):
    q = d[d.validation_mse_x.notna() & d.m.notna()] if "validation_mse_x" in d else d.iloc[:0]
    return q[q.warm_start.isna()] if "warm_start" in q else q


def with_chain_start(d):
    """Trained rows plus the warm chain-start rows (= scratch model at m0): the curve of a
    continuation starts where the path starts."""
    q = d[d.validation_mse_x.notna() & d.m.notna()] if "validation_mse_x" in d else d.iloc[:0]
    return q


def clopper_pearson(k, n, a=.05):
    from scipy.stats import beta as beta_dist
    lo = beta_dist.ppf(a/2, k, n-k+1) if k > 0 else 0.
    hi = beta_dist.ppf(1-a/2, k+1, n-k) if k < n else 1.
    return lo, hi


def unfold_table(d, label, include_chain_start=True):
    rows = []
    q = with_chain_start(d) if include_chain_start else trained(d)
    for (model, m), g in q.groupby(["model", "m"]):
        if "unfolded" not in g or g.unfolded.isna().all():
            continue
        g = g[g.unfolded.notna()]
        k, n = int(g.unfolded.astype(bool).sum()), len(g)
        lo, hi = clopper_pearson(k, n)
        qq = g.mse_x.quantile([.25, .5, .75]).values
        r = g.knn_chart_ratio.quantile([.25, .5, .75]).values
        rows.append(dict(corrida=label, model=model, m=int(m), desplegadas=f"{k}/{n}", tasa=k/n,
                         ic95_inf=lo, ic95_sup=hi, mse_x_mediana=qq[1], mse_x_iqr=f"{qq[0]:.3g}–{qq[2]:.3g}",
                         knn_ratio_mediana=r[1], knn_ratio_iqr=f"{r[0]:.3g}–{r[2]:.3g}"))
    return pd.DataFrame(rows)


def recon_table(d, label):
    """For data without a chart (MNIST): ambient MSE and (if present) kNN label accuracy."""
    rows = []
    for (model, m), g in with_chain_start(d).groupby(["model", "m"]):
        qq = g.mse_x.quantile([.25, .5, .75]).values
        row = dict(corrida=label, model=model, m=int(m), n=len(g), mse_x_mediana=qq[1],
                   mse_x_iqr=f"{qq[0]:.3g}–{qq[2]:.3g}")
        if "knn_label_accuracy" in g and g.knn_label_accuracy.notna().any():
            row["knn_label_acc_mediana"] = g.knn_label_accuracy.median()
        rows.append(row)
    return pd.DataFrame(rows)


def information_table(d, column="chart_mse_z", model="oracle"):
    q = d[(d.model == model) & d.m.notna()]
    return q.groupby("m")[column].quantile([.25, .5, .75]).unstack().rename(
        columns={.25: "q25", .5: "mediana", .75: "q75"})


def order_test(dA, dB, M, models=("vae", "vae_warm"), unfolding=True):
    """At m = M both orders use the SAME modes and the same model class: only the path differs.
    Fisher exact (unfolding) and Mann-Whitney (mse_x), one-sided 'spectral better'."""
    from scipy.stats import fisher_exact, mannwhitneyu
    out = []
    for model in models:
        a = trained(dA)[lambda t: (t.model == model) & (t.m == M)]
        b = trained(dB)[lambda t: (t.model == model) & (t.m == M)]
        if not len(a) or not len(b):
            continue
        row = dict(model=model, m=M, n_espectral=len(a), n_aleatorio=len(b),
                   mse_espectral=a.mse_x.median(), mse_aleatorio=b.mse_x.median(),
                   mannwhitney_p=mannwhitneyu(a.mse_x, b.mse_x, alternative="less").pvalue)
        if unfolding and "unfolded" in a and a.unfolded.notna().any():
            ka, kb = int(a.unfolded.astype(bool).sum()), int(b.unfolded.astype(bool).sum())
            row.update(espectral=f"{ka}/{len(a)}", aleatorio=f"{kb}/{len(b)}",
                       fisher_p=fisher_exact([[ka, len(a)-ka], [kb, len(b)-kb]], alternative="greater")[1])
        out.append(row)
    return pd.DataFrame(out)


def paired_warm_vs_scratch(d, M, column="mse_x"):
    """Same seed = same data, split and scratch initialization: paired Wilcoxon, warm < scratch."""
    from scipy.stats import wilcoxon
    q = trained(d)
    a = q[(q.model == "vae") & (q.m == M)].set_index("seed")[column]
    b = q[(q.model == "vae_warm") & (q.m == M)].set_index("seed")[column]
    s = a.index.intersection(b.index)
    if len(s) < 2:
        return None
    diff = b[s] - a[s]
    p = wilcoxon(b[s], a[s], alternative="less").pvalue if (diff != 0).any() else 1.
    return dict(m=M, n=len(s), scratch_mediana=a[s].median(), warm_mediana=b[s].median(),
                warm_mejor=int((diff < 0).sum()), wilcoxon_p=p)


def errorbar_rates(ax, t, model, dx=1., **kw):
    t = t[t.model == model].sort_values("m")
    if len(t):
        ax.errorbar(t.m*dx, t.tasa, yerr=[t.tasa-t.ic95_inf, t.ic95_sup-t.tasa], capsize=3, **kw)
