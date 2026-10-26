"""Latent-space grid in the style of 'truth / d=2 / d=4 / ... ' figures.

Every panel shows TEST points, coloured by the true arc-length coordinate s, placed at an AFFINE
readout of the representation onto the true chart (s, h):
  * "oracle" rows: the affine readout of psi_1..psi_m (what DMAP alone gives at budget m);
  * model rows:    the 2-D latent mean mu(x), affinely aligned to the chart with a map fitted on the
                   TRAIN codes.  An affine map cannot unfold anything, so a fold or a collapse in the
                   latent stays visible.
The replicate shown in each row is the one with the MEDIAN kNN chart ratio at the largest budget
(representative, not best); the same replicate is used across the row.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def _affine(code_train, chart_train, code):
    a = np.column_stack([code_train, np.ones(len(code_train))])
    w = np.linalg.lstsq(a, chart_train, rcond=None)[0]
    return np.column_stack([code, np.ones(len(code))]) @ w


def _stem(folder, seed, model, m, first_m):
    """warm chains start from the scratch model at the first budget (no separate file)."""
    base = model[:-5] if model.endswith("_warm") else model
    name = model if not (model.endswith("_warm") and m == first_m) else base
    return Path(folder)/f"seed_{seed}"/f"{name}_m{m}.npz"


def representative_seed(df, model, m):
    col = "chart_mse_z" if model == "oracle" else "knn_chart_ratio"
    q = df[(df.model == model) & (df.m == m) & df[col].notna()]
    if "warm_start" in q:
        q = q[q.warm_start.isna()]
    if not len(q):
        return None, None
    q = q.sort_values(col)
    r = q.iloc[(len(q)-1)//2]
    return int(r.seed), float(r[col])


def panel(folder, seed, model, m, first_m):
    folder = Path(folder)
    data = np.load(folder/f"seed_{seed}"/"data.npz")
    truth, train, test = data["truth"], data["train"], data["test"]
    if model in ("plainae", "plainvae"):
        m = first_m                      # no DMAP: the latent does not depend on m
    path = _stem(folder, seed, model, m, first_m)
    if not path.exists():
        return None, truth[test]
    p = np.load(path)
    if model == "oracle":
        return p["prediction"], truth[test]
    code = p["code"]
    if code.shape[1] != 2:
        raise ValueError(f"{path.name}: latent dimension {code.shape[1]} != 2")
    return _affine(p["code_train"], truth[train], code), truth[test]


def latent_grid(rows, modes, path, title=None, point_size=1.5):
    """rows: list of dicts(label, folder, model).  model = 'truth' for the ground-truth column row,
    'oracle' for the DMAP affine readout, or a trained model name (ae, vae, ae_warm, vae_warm...)."""
    ncol = len(modes) + 1
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.9*ncol, 1.75*len(rows)+.4), squeeze=False)
    table = []
    for i, row in enumerate(rows):
        folder = Path(row["folder"])
        df = pd.read_csv(folder/"metrics.csv")
        first_m = int(df.m.dropna().min())
        seed, score = representative_seed(df, row["model"], first_m if row["model"] in ("plainae", "plainvae") else max(modes))
        if seed is None:
            for ax in axes[i]:
                ax.axis("off")
            axes[i][0].text(0, .5, f"{row['label']}\n(sin datos)", transform=axes[i][0].transAxes)
            continue
        data = np.load(folder/f"seed_{seed}"/"data.npz")
        ref = data["truth"][data["test"]]
        lim = (ref.min(0), ref.max(0))
        pad = .08*(lim[1]-lim[0])
        panels = [("truth", ref)] + [(m, panel(folder, seed, row["model"], m, first_m)[0]) for m in modes]
        for j, (m, z) in enumerate(panels):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            if z is None:
                ax.text(.5, .5, "—", ha="center", va="center", transform=ax.transAxes)
                continue
            inside = np.all((z >= lim[0]-4*pad) & (z <= lim[1]+4*pad), axis=1)
            ax.scatter(z[inside, 0], z[inside, 1], c=ref[inside, 0], s=point_size, cmap="viridis",
                       vmin=lim[0][0], vmax=lim[1][0], rasterized=True, linewidths=0)
            ax.set_xlim(lim[0][0]-pad[0], lim[1][0]+pad[0])
            ax.set_ylim(lim[0][1]-pad[1]*4, lim[1][1]+pad[1]*4)
            if j and m != "truth":
                mq = first_m if row["model"] in ("plainae", "plainvae") else m
                q = df[(df.model == row["model"]) & (df.m == mq) & (df.seed == seed)]
                if "warm_start" in q and row["model"].endswith("_warm"):
                    q = q.sort_values("warm_start", na_position="first")
                col = "chart_mse_z" if row["model"] == "oracle" else "knn_chart_ratio"
                if len(q) and col in q and pd.notna(q[col].iloc[0]):
                    v = float(q[col].iloc[0])
                    lab = f"MSE$_z$ {v:.2g}" if col == "chart_mse_z" else f"kNN/piso {v:.2g}"
                    ax.text(.02, .02, lab, fontsize=6, transform=ax.transAxes, color="k",
                            bbox=dict(facecolor="white", alpha=.7, lw=0, pad=1))
                    table.append(dict(fila=row["label"], m=m, seed=seed, **{col: v}))
                if (~inside).any():
                    ax.text(.98, .02, f"{(~inside).sum()} fuera", fontsize=5, ha="right",
                            transform=ax.transAxes, color="C3")
            if i == 0:
                ax.set_title("truth" if m == "truth" else f"$m = {m}$", fontsize=10)
        axes[i][0].set_ylabel(f"{row['label']}\n(réplica {seed})", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"), dpi=200)
    fig.savefig(Path(path).with_suffix(".pdf"), dpi=200)
    return fig, pd.DataFrame(table)


def chart_r2_table(folder, models=None):
    """Per-coordinate R^2 of the AFFINE readout of each representation onto the true chart (s, h),
    fitted on train, scored on test.  Global shape check: a curved or squashed latent keeps R^2_s
    high but loses R^2_h.  Rows: seed, model, m, r2_s, r2_h (oracle included)."""
    folder = Path(folder)
    df = pd.read_csv(folder/"metrics.csv")
    first_m = int(df.m.dropna().min())
    keys = df[df.m.notna() & df.model.notna()][["seed", "model", "m"]].drop_duplicates()
    if models is not None:
        keys = keys[keys.model.isin(list(models) + ["oracle"])]
    out = []
    cache = {}
    for seed, model, m in keys.itertuples(index=False):
        if model in ("oracle_decoder", "affine_x", "chart_decoder", "reference"):
            continue
        seed, m = int(seed), int(m)
        if seed not in cache:
            d = np.load(folder/f"seed_{seed}"/"data.npz")
            cache[seed] = (d["truth"], d["train"], d["test"])
        truth, train, test = cache[seed]
        path = _stem(folder, seed, model, m, first_m)
        if not path.exists():
            continue
        p = np.load(path)
        if model == "oracle":
            z = p["prediction"]
        elif "code_train" in p.files and p["code"].shape[1] == 2:
            z = _affine(p["code_train"], truth[train], p["code"])
        else:
            continue
        t = truth[test]
        r2 = 1 - ((z - t)**2).mean(0)/t.var(0)
        out.append(dict(seed=seed, model=model, m=m, r2_s=float(r2[0]), r2_h=float(r2[1])))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- latent vs oracle points

def similarity_align(a, b):
    """Best map a -> b in the similarity group (E(2) = translations, rotations, reflections, plus
    ONE isotropic scale): orthogonal Procrustes.  The scale is needed because an AE latent has no
    units (the VAE's is fixed by the prior, the oracle's is the chart's); E(2) alone would compare
    numbers in different units.  Returns aligned a, scale, rotation (2x2, det = +-1)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ma, mb = a.mean(0), b.mean(0)
    ac, bc = a-ma, b-mb
    u, sv, vt = np.linalg.svd(ac.T @ bc)
    rot = u @ vt
    scale = sv.sum()/max((ac**2).sum(), 1e-300)
    return scale*ac @ rot + mb, float(scale), rot


def _code_test(folder, seed, model, m, first_m):
    path = _stem(folder, seed, model, m, first_m)
    if not path.exists():
        return None
    p = np.load(path)
    return p["prediction"] if model == "oracle" else p["code"]


def latent_vs_oracle_table(folder, models, oracle_folder=None):
    """For every (seed, model, m): squared error PER LATENT COORDINATE between the oracle points
    r_oracle (affine readout of psi_1..psi_m onto the chart, test set) and the model's latent mean
    after similarity alignment (fitted on the same test points: 4 parameters for ~600 points), and
    the same error against the TRUE chart.  *_s_norm / *_h_norm: the same errors per chart axis divided
    by that axis' variance (0 = perfect, 1 = no information), so a collapsed height (Var h = H^2/12 is
    ~36x smaller than Var s) is not hidden by the s axis.  The affine column is NaN for m <= 2: there the
    latent and the oracle are both affine images of (psi_1, psi_2), so an affine map matches them by
    construction.  Models without m (plainae / plainvae, no DMAP) are
    compared with the oracle at EVERY budget m of the run.  oracle_folder: where to read the oracle
    from (default: the same run; the oracle depends only on data + beta, not on the model)."""
    folder = Path(folder)
    of = Path(oracle_folder) if oracle_folder else folder
    df = pd.read_csv(folder/"metrics.csv")
    first_m = int(df.m.dropna().min())
    modes = sorted(int(v) for v in pd.read_csv(of/"metrics.csv").query("model == 'oracle'").m.unique())
    out = []
    for model in models:
        rows = df[(df.model == model) & df.m.notna()]
        if "warm_start" in rows:
            rows = rows.drop_duplicates(["seed", "m"])
        for seed, m in rows[["seed", "m"]].itertuples(index=False):
            seed, m = int(seed), int(m)
            z = _code_test(folder, seed, model, m, first_m)
            if z is None or z.shape[1] != 2:
                continue
            truth = np.load(folder/f"seed_{seed}"/"data.npz")
            t = truth["truth"][truth["test"]]
            zt, sc_t, _ = similarity_align(z, t)
            budgets = modes if model in ("plainae", "plainvae") else [m]
            for mo in budgets:
                r = _code_test(of, seed, "oracle", mo, first_m)
                if r is None:
                    continue
                zo, sc_o, _ = similarity_align(z, r)
                za = np.column_stack([z, np.ones(len(z))]) @ np.linalg.lstsq(np.column_stack([z, np.ones(len(z))]), r, rcond=None)[0]
                var = t.var(0)                       # per-axis variance of the TRUE chart (test)
                lo_ax = ((zo-r)**2).mean(0)/var       # normalized per axis: 0 = perfect, 1 = no information
                lt_ax = ((zt-t)**2).mean(0)/var
                or_ax = ((r-t)**2).mean(0)/var
                out.append(dict(seed=seed, model=model, m=mo,
                                latent_vs_oracle_s_norm=float(lo_ax[0]), latent_vs_oracle_h_norm=float(lo_ax[1]),
                                latent_vs_truth_s_norm=float(lt_ax[0]), latent_vs_truth_h_norm=float(lt_ax[1]),
                                oracle_vs_truth_s_norm=float(or_ax[0]), oracle_vs_truth_h_norm=float(or_ax[1]),
                                latent_vs_oracle=float(np.mean((zo-r)**2)),
                                latent_vs_oracle_affine=(float(np.mean((za-r)**2)) if mo > 2 else np.nan),
                                latent_vs_truth=float(np.mean((zt-t)**2)),
                                oracle_vs_truth=float(np.mean((r-t)**2)),
                                scale_to_oracle=sc_o))
    return pd.DataFrame(out)


def rings_figure(rows, modes, path, title=None, max_points=400, seed_pick=None):
    """'Rings': oracle points r_oracle (open circles) and the model's latent mean aligned onto them by a
    similarity map (dots), joined by a thin segment; colour = true s.  One row per model, one column
    per budget m.  Representative replicate = median kNN ratio at the largest m (or seed_pick)."""
    fig, axes = plt.subplots(len(rows), len(modes), figsize=(2.6*len(modes), 1.9*len(rows)+.4), squeeze=False)
    rng = np.random.default_rng(0)
    for i, row in enumerate(rows):
        folder = Path(row["folder"])
        of = Path(row.get("oracle_folder", folder))
        df = pd.read_csv(folder/"metrics.csv")
        first_m = int(df.m.dropna().min())
        plain = row["model"] in ("plainae", "plainvae")
        seed = seed_pick if seed_pick is not None else representative_seed(df, row["model"], first_m if plain else max(modes))[0]
        if seed is None:
            continue
        d = np.load(folder/f"seed_{seed}"/"data.npz")
        s = d["truth"][d["test"]][:, 0]
        pick = rng.choice(len(s), min(max_points, len(s)), replace=False)
        for j, m in enumerate(modes):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            r = _code_test(of, seed, "oracle", m, first_m)
            z = _code_test(folder, seed, row["model"], first_m if plain else m, first_m)
            if r is None or z is None or z.shape[1] != 2:
                ax.text(.5, .5, "—", ha="center", va="center", transform=ax.transAxes)
                continue
            za, _, _ = similarity_align(z, r)
            err = float(np.mean((za-r)**2))
            for k in pick:
                ax.plot([r[k, 0], za[k, 0]], [r[k, 1], za[k, 1]], color="0.75", lw=.3, zorder=1)
            ax.scatter(r[pick, 0], r[pick, 1], s=9, facecolors="none", edgecolors=plt.cm.viridis((s[pick]-s.min())/np.ptp(s)),
                       linewidths=.6, zorder=2)
            ax.scatter(za[pick, 0], za[pick, 1], s=2, c=s[pick], cmap="viridis", zorder=3, linewidths=0)
            ax.text(.02, .02, f"‖r_or − r_lat‖² {err:.3g}", fontsize=6, transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=.7, lw=0, pad=1))
            if i == 0:
                ax.set_title(f"$m = {m}$", fontsize=10)
        axes[i][0].set_ylabel(f"{row['label']}\n(réplica {seed})", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"), dpi=200)
    fig.savefig(Path(path).with_suffix(".pdf"), dpi=200)
    return fig
