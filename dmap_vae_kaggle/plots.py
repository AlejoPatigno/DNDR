"""Figures computed only from completed runs. Error bars = seed SD, not confidence intervals."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import angles, output_angles


def finish(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def ramachandran(arrays, names, path):
    edges = np.linspace(-np.pi, np.pi, 73)
    probs = []
    for a in arrays:
        # Match FlashDiffusion Figure 2 orientation: horizontal psi, vertical phi.
        h = np.histogram2d(a[:, 1], a[:, 0], bins=(edges, edges))[0]
        probs.append(h/max(h.sum(), 1))
    shift = np.log(probs[0].max())
    fig, axes = plt.subplots(1, len(probs), figsize=(4*len(probs), 4), squeeze=False)
    for ax, p, name in zip(axes[0], probs, names):
        energy = np.ma.masked_where(p == 0, -np.log(np.maximum(p, 1e-300))+shift)
        im = ax.pcolormesh(np.degrees(edges), np.degrees(edges), energy.T,
                           vmin=0, vmax=8, cmap="viridis", rasterized=True)
        ax.set(xlabel=r"$\psi$ (deg)", ylabel=r"$\phi$ (deg)",
               title=name if p.sum() else name+" (no defined angles)", aspect="equal")
    fig.colorbar(im, ax=axes.ravel().tolist(), label=r"$-\log p+C_{\mathrm{ref}}$")
    finish(fig, path)
    np.savez_compressed(path.with_suffix(".npz"), probabilities=np.asarray(probs), edges=edges)


MODEL_COLORS = {"ae": "C0", "vae": "C1", "direct": "C2", "plainvae": "C3", "plainae": "C6", "vae_from_ae": "C4",
                "cvae": "C5", "ae_warm": "C0", "vae_warm": "C1", "direct_warm": "C2",
                "oracle": "C7", "oracle_decoder": "C8", "chart_decoder": "k", "affine_x": "gray"}
STOCHASTIC_FAMILIES = ("vae", "vae_warm", "vae_from_ae", "plainvae", "cvae")
NON_NEURAL = ("oracle", "oracle_decoder", "chart_decoder", "reference", "affine_x")


def color(name):
    return MODEL_COLORS.get(name, "C9")


def stem(model, m, restart=0):
    """File stem written by experiment.py (restart 0 keeps the historical name)."""
    return f"{model}_m{int(m)}" + (f"_r{int(restart)}" if restart else "")


def _curve(ax, df, model, column, label=None, **kw):
    g = df[(df.model == model) & df.m.notna()]
    if column not in g or g[column].notna().sum() == 0:
        return
    stats = g.groupby("m")[column].agg(["mean", "std"])
    kw.setdefault("color", color(model))
    kw.setdefault("ls", "--" if model.endswith("_warm") else "-")
    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"].fillna(0), marker="o", ms=3,
                label=label or model, **kw)


def trained(df):
    """Rows of models actually trained (drops the duplicated warm-chain start rows)."""
    d = df[df.validation_mse_x.notna() & df.m.notna()]
    return d[d.warm_start.isna()] if "warm_start" in d else d


def best_by_validation(df, model):
    """One row per (seed, restart): the m with the lowest validation_mse_x for this model."""
    q = trained(df)
    q = q[q.model == model]
    return q.loc[q.groupby(["seed", "restart"]).validation_mse_x.idxmin()] if len(q) else q


def best_restart(df, model):
    """One row per (seed, m): the restart with the lowest validation_mse_x (what a user would keep)."""
    q = trained(df)
    q = q[q.model == model]
    return q.loc[q.groupby(["seed", "m"]).validation_mse_x.idxmin()] if len(q) else q


def summary_table(df):
    """mean ± sd over seeds AND restarts; ambient and chart quantities in separate columns."""
    cols = [c for c in ("mse_x", "mc_mse_x", "validation_mse_x", "chart_mse_z", "oracle_exact_mse_x",
                        "knn_chart_mse", "knn_chart_ratio", "unfolded", "similarity_chart_mse",
                        "affine_chart_mse", "kl_mean", "prior_surface_dist_mean", "prior_chart_coverage",
                        "prior_chart_tv_to_reference", "angular_mae_deg", "knn_label_accuracy",
                        "affine_validation_mse_x") if c in df]
    d = df[df.model != "reference"].copy()
    d["m"] = d.m.fillna(-1).astype(int)
    if "unfolded" in d:
        d["unfolded"] = d.unfolded.astype(float)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(d[c])]
    agg = d.groupby(["model", "m"])[num].agg(["mean", "std"])
    agg[("n", "rows")] = d.groupby(["model", "m"]).size()
    return agg.dropna(axis=1, how="all")


def unfolding_table(df):
    """Chart-recovery success rates.  'all' = every (seed, restart); 'best_restart' = per seed the
    restart a practitioner would keep (lowest validation), the relevant number for a method claim."""
    if "unfolded" not in df:
        return None
    rows = []
    q = trained(df)
    q = q[q.unfolded.notna()]
    for (model, m), g in q.groupby(["model", "m"]):
        b = best_restart(g, model)
        rows.append(dict(model=model, m=int(m), n=len(g), unfolded_all=g.unfolded.astype(float).mean(),
                         median_ratio=g.knn_chart_ratio.median(), n_seeds=g.seed.nunique(),
                         unfolded_best_restart=b.unfolded.astype(float).mean(),
                         mse_x_best_restart=b.mse_x.mean()))
    return pd.DataFrame(rows).sort_values(["model", "m"])


def cfg_modes(df):
    return sorted(int(v) for v in df.m.dropna().unique())


def select_config(df, models=STOCHASTIC_FAMILIES):
    """(model, m) with the lowest mean VALIDATION mse over seeds and restarts, never test; then the
    (seed, restart) row with the median validation of that configuration."""
    cand = trained(df)
    cand = cand[cand.model.isin(models)]
    if cand.empty:
        return None
    model, m = cand.groupby(["model", "m"]).validation_mse_x.mean().idxmin()
    rows = cand[(cand.model == model) & (cand.m == m)].sort_values("validation_mse_x")
    r = rows.iloc[len(rows)//2]
    return dict(model=model, m=int(m), seed=int(r.seed), restart=int(r.restart),
                rule="min mean validation_mse_x over seeds and restarts; median (seed, restart)")


def select_generation_config(df):     # backwards-compatible name
    c = select_config(df)
    return (c["model"], c["m"], c["seed"]) if c else None


def swiss_ensemble_figure(root, df, sel, path):
    folder = root/f"seed_{sel['seed']}"
    data = np.load(folder/"data.npz")
    reference = data["x"][data["test"]]
    p = np.load(folder/f"{stem(sel['model'], sel['m'], sel['restart'])}.npz")
    row = df[(df.model == sel["model"]) & (df.m == sel["m"]) & (df.seed == sel["seed"])
             & (df.restart == sel["restart"])].iloc[0]
    ref = df[(df.model == "reference") & (df.seed == sel["seed"])].iloc[0]
    knn = f", knn/piso {row.knn_chart_ratio:.3g}" if "knn_chart_ratio" in row and pd.notna(row.knn_chart_ratio) else ""
    panels = [(reference, f"Reference test (N={len(reference)})\nsurface dist {ref.test_surface_dist_mean:.3f}"),
              (p["prediction"], f"$D(\\mu(x))$  MSE$_x$={row.mse_x:.3g}\nsurface dist {row.recon_surface_dist_mean:.3f}"),
              (p["prediction_mc"], f"$E_q[D(z)]$  MSE$_x$={row.mc_mse_x:.3g}\nsurface dist {row.mcrecon_surface_dist_mean:.3f}"),
              (p["prior_samples"], f"Prior samples (N={len(p['prior_samples'])})\nsurface dist {row.prior_surface_dist_mean:.3f}, "
               f"coverage {row.prior_chart_coverage:.2f} (ref {ref.test_chart_coverage:.2f})")]
    fig = plt.figure(figsize=(16, 4.4))
    # Axes fixed to the REFERENCE extent (+10%): a few far prior samples must not squash the roll.
    lo, hi = reference.min(0), reference.max(0)
    pad = .1*(hi-lo)
    lo, hi = lo-pad, hi+pad
    for i, (a, title) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 4, i, projection="3d")
        inside = np.all((a >= lo) & (a <= hi), axis=1)
        if (~inside).any():
            title += f"\n{(~inside).sum()} of {len(a)} points outside the axes"
        a = a[inside]
        ax.scatter(a[:, 0], a[:, 1], a[:, 2], s=3, color=color(sel["model"]))
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.set_title(title, fontsize=9)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
        ax.set_box_aspect(np.maximum(hi-lo, 1e-6))
    fig.suptitle(f"{sel['model']}, m={sel['m']}, seed {sel['seed']}, restart {sel['restart']} "
                 f"(selected by validation MSE$_x${knn})", y=1.06)
    fig.tight_layout()
    finish(fig, path)


def plot_results(folder):
    root = Path(folder)
    runs = sorted(root.glob("seed_*/metrics.json"))
    if not runs:
        raise ValueError("No completed runs.")
    df = pd.DataFrame([row for path in runs for row in json.loads(path.read_text())])
    if "restart" not in df:
        df["restart"] = 0
    df["restart"] = df.restart.fillna(0).astype(int)
    df.to_csv(root/"metrics.csv", index=False)
    summary_table(df).to_csv(root/"summary.csv")
    unf = unfolding_table(df)
    if unf is not None:
        unf.to_csv(root/"unfolding.csv", index=False)
    provenance = json.loads((runs[0].parent/"provenance.json").read_text())
    meta, cfg = provenance["data"], provenance["config"]
    neural = [m for m in df.model.unique() if m not in NON_NEURAL]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name in neural:
        if name in ("plainvae", "plainae"):
            v = df[df.model == name].mse_x
            ax.axhline(v.mean(), color=color(name), ls="--", label=f"{name} ({v.mean():.3g}±{v.std():.1g})")
        else:
            _curve(ax, df, name, "mse_x")
    if "chart_decoder" in set(df.model):
        ax.axhline(df[df.model == "chart_decoder"].mse_x.mean(), color="k", ls=":", label="decoder on true chart")
    if "affine_x" in set(df.model):
        _curve(ax, df, "affine_x", "mse_x", r"affine readout of $x$ from $\psi_{1..m}$ (no training)", ls=":")
    ax.set(xlabel="Retained nontrivial modes m", ylabel=r"test ambient MSE$_x$ (per coordinate)",
           xscale="log", yscale="log", title=r"Reconstruction $D(\mu(x))$ (seeds × restarts)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    finish(fig, root/"reconstruction")

    if "oracle" in set(df.model):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        _curve(ax, df, "oracle", "chart_mse_z", r"oracle chart MSE$_z$ (per chart coord.)")
        if "oracle_exact_mse_x" in df:
            _curve(ax, df, "oracle", "oracle_exact_mse_x", r"oracle $\to$ exact $g$: MSE$_x$", color="C6")
            _curve(ax, df, "oracle", "oracle_exact_bound_x", r"bound $\frac{2}{3}$MSE$_z$", color="gray", ls=":")
        _curve(ax, df, "oracle_decoder", "mse_x", r"oracle $\to$ learned decoder: MSE$_x$")
        _curve(ax, df, "direct", "mse_x", r"direct decoder from $m$ modes: MSE$_x$")
        if "chart_decoder" in set(df.model):
            ax.axhline(df[df.model == "chart_decoder"].mse_x.mean(), color="k", ls=":", label="learned decoder on true chart")
        ax.set(xlabel="Retained nontrivial modes m", ylabel="MSE (see legend for units)", xscale="log",
               yscale="log", title="Spectral vs decoder error")
        ax.legend(fontsize=7)
        fig.tight_layout()
        finish(fig, root/"oracle_decomposition")

    if "knn_chart_ratio" in df and df.knn_chart_ratio.notna().any():
        thr = float(cfg.get("unfold_ratio", 10.))
        q = trained(df)
        q = q[q.knn_chart_ratio.notna()]
        fams = [n for n in neural if n in set(q.model)]
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        rng = np.random.default_rng(0)
        for j, name in enumerate(fams):
            g = q[q.model == name]
            jitter = np.exp(rng.uniform(-.08, .08, len(g)) + (j-len(fams)/2)*.05)
            ax.scatter(g.m*jitter, g.knn_chart_ratio, s=12, alpha=.6, color=color(name),
                       marker="s" if name.endswith("_warm") else "o",
                       label=f"{name} ({g.unfolded.astype(float).mean():.0%} desplegadas)")
        ax.axhline(thr, color="k", ls=":", lw=.8, label=f"umbral {thr:g}× piso")
        ax.set(xscale="log", yscale="log", xlabel="m", ylabel="kNN chart MSE / piso (true chart)",
               title="Recuperación no lineal de la carta (cada punto = semilla × reinicio)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        finish(fig, root/"chart_recovery")

    if "similarity_chart_mse" in df:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, col in zip(axes, ("rigid_chart_mse", "similarity_chart_mse", "affine_chart_mse")):
            for name in neural:
                if name in ("plainvae", "plainae"):
                    v = df[df.model == name][col]
                    if v.notna().any():
                        ax.axhline(v.mean(), color=color(name), ls="--", label=name)
                else:
                    _curve(ax, df, name, col)
            if "constant_chart_mse" in df:
                ax.axhline(df[df.model == "oracle"].constant_chart_mse.mean(), color="k", ls=":", label="constant (train mean)")
            ax.set(xscale="log", yscale="log", xlabel="m", title=col.replace("_", " "))
        axes[0].set_ylabel("latent-to-chart MSE (test, per chart coord.)")
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        finish(fig, root/"chart_alignment")

    if "kl_mean" in df and df.kl_mean.notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name in neural:
            _curve(axes[0], df, name, "kl_mean")
            g = df[(df.model == name) & df.posterior_var_mean.notna() & df.m.notna()]
            if len(g):
                pv = g.assign(v=g.posterior_var_mean.apply(np.mean)).groupby("m").v.mean()
                axes[1].plot(pv.index, pv.values, marker="o", ms=3, color=color(name), label=name,
                             ls="--" if name.endswith("_warm") else "-")
        axes[0].set(xscale="log", yscale="log", xlabel="m", ylabel="KL(q(z|x) || N(0,I)), test mean")
        axes[1].set(xscale="log", yscale="log", xlabel="m", ylabel="mean posterior variance, test")
        for a in axes:
            a.legend(fontsize=7)
        fig.tight_layout()
        finish(fig, root/"kl_posterior")

    # Training curves: one panel per model, each at ITS OWN m chosen by validation.
    fams = [n for n in neural if n in set(trained(df).model)]
    if fams:
        fig, axes = plt.subplots(1, len(fams), figsize=(4.2*len(fams), 3.6), squeeze=False)
        for ax, name in zip(axes[0], fams):
            sel = select_config(df, models=(name,))
            path = root/f"seed_{sel['seed']}"/f"{stem(name, sel['m'], sel['restart'])}_history.json"
            if not path.exists():
                ax.set_title(f"{name}: sin historial"); continue
            h = [r for r in json.loads(path.read_text()) if r["epoch"] >= 0]
            ep = [r["epoch"] for r in h]
            ax.plot(ep, [r["train_recon"] for r in h], color=color(name), label="train")
            ax.plot(ep, [r["validation_recon"] for r in h], color=color(name), ls="--", label="validation")
            ax.set(yscale="log", xlabel="epoch", title=f"{name}, m={sel['m']} (s{sel['seed']} r{sel['restart']})")
            ax.legend(fontsize=7)
        axes[0][0].set_ylabel("reconstruction term")
        fig.tight_layout()
        finish(fig, root/"training_curves")

    chosen = select_config(df)
    (root/"figure_selection.json").write_text(json.dumps(chosen or {"model": None}, indent=2))
    if not chosen:
        return
    if cfg["dataset"] == "swiss":
        swiss_ensemble_figure(root, df, chosen, root/"swiss_ensembles")
        for name in [n for n in STOCHASTIC_FAMILIES if n in set(trained(df).model)]:
            swiss_ensemble_figure(root, df, select_config(df, models=(name,)), root/f"swiss_ensembles_{name}")
    elif cfg["dataset"] in ("alanine", "torus"):
        first = root/f"seed_{chosen['seed']}"
        data = np.load(first/"data.npz")
        budgets = sorted(set([min(cfg["modes"], key=lambda n: abs(n-8)), max(cfg["modes"])]))
        for model in ("oracle", "vae", "cvae"):
            aa, names = [data["truth"][data["test"]]], ["Reference"]
            for m in budgets:
                path = first/f"{model}_m{m}.npz"
                if path.exists():
                    p = np.load(path)
                    aa.append(angles(p["prediction"]) if model == "oracle" else output_angles(p["prediction"], cfg["dataset"], meta))
                    names.append(f"{model}, m={m}")
            if len(aa) > 1:
                ramachandran(aa, names, root/f"angles_{model}_reconstructed")
        for name in [n for n in STOCHASTIC_FAMILIES if n in set(trained(df).model)]:
            sel = select_config(df, models=(name,))
            d2 = np.load(root/f"seed_{sel['seed']}"/"data.npz")
            sample = np.load(root/f"seed_{sel['seed']}"/f"{stem(name, sel['m'], sel['restart'])}.npz")["prior_samples"]
            ramachandran([d2["truth"][d2["test"]], output_angles(sample, cfg["dataset"], meta)],
                         ["Reference", f"Prior samples ({name}, m={sel['m']})"], root/f"angles_generated_{name}")
    elif cfg["dataset"] == "mnist":
        first = root/f"seed_{chosen['seed']}"
        data = np.load(first/"data.npz")
        reference = data["x"][data["test"]]
        p = np.load(first/f"{stem(chosen['model'], chosen['m'], chosen['restart'])}.npz")
        fig, axes = plt.subplots(4, 8, figsize=(10, 5.2))
        for row, array in zip(axes, (reference, p["prediction"], p["prediction_mc"], p["prior_samples"])):
            for ax, im in zip(row, array):
                ax.imshow(im.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
        fig.suptitle(f"Reference / D(mu) / E_q[D(z)] / prior samples  ({chosen['model']}, m={chosen['m']})")
        fig.tight_layout()
        finish(fig, root/"mnist_ensembles")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    plot_results(p.parse_args().folder)
