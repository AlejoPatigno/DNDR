"""Export the raster layers and data tables used by the LaTeX/pgfplots figures of the paper.

Only DATA MARKS are rasterised (transparent PNG, no axes, no text): frames, titles and labels are
drawn by pgfplots in the document's text colour, so the same figure works with
\\usepackage[night]{dacnotes} and \\usepackage{dacnotes}.

Outputs (in OUT, default /kaggle/working/paper_figs):
  img/fig3_<row>_<col>.png    affine reconstructions of the chart (test points), common extent
  img/rings_<set>_<row>_m<m>.png   oracle points (rings) + latent aligned by a similarity map (dots)
  data/readout_vs_d.dat       in-sample relative chart-readout error vs d for IMAP / UMAP / DMAP
  data/fig3_meta.json         replicate used, extents, per-panel errors (for the caption)

usage (Kaggle, with the outputs of notebook 08 v2 in /kaggle/working or attached as Input):
  python export_figs.py --random <results_swiss_final_AE_l0.1 dir> --out /kaggle/working/paper_figs
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

W, H = 60., 10.
PAD_S, PAD_H = 0.06*W, 0.25*H            # common extent of every chart panel
EXTENT = (-PAD_S, W+PAD_S, -PAD_H, H+PAD_H)
CMAP = LinearSegmentedColormap.from_list("viridis_cut", plt.cm.viridis(np.linspace(.14, 1., 256)))
PANEL_IN = (1.6, 1.15)                    # inches of one panel (aspect of the pgfplots cell)


def _panel(path, pts=None, color=None, rings=None, segs=None, size=4.0, dpi=300):
    fig = plt.figure(figsize=PANEL_IN, dpi=dpi)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.axis("off"); ax.patch.set_alpha(0)
    if segs is not None:
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segs, colors=(.55, .55, .55, .45), linewidths=.35, zorder=1))
    if rings is not None:
        ax.scatter(rings[:, 0], rings[:, 1], s=size*3.2, facecolors="none",
                   edgecolors=CMAP((color-0)/W), linewidths=.45, zorder=2)
    if pts is not None:
        ax.scatter(pts[:, 0], pts[:, 1], s=size, c=color, cmap=CMAP, vmin=0, vmax=W, linewidths=0, zorder=3)
    fig.savefig(path, transparent=True, dpi=dpi)
    plt.close(fig)


def _affine(a_train, y_train, a):
    A = np.column_stack([a_train, np.ones(len(a_train))])
    w = np.linalg.lstsq(A, y_train, rcond=None)[0]
    return np.column_stack([a, np.ones(len(a))]) @ w


def _clip(z):
    """Points outside the common extent are pinned to its border (and counted) so that a
    collapsed or exploded latent stays visible instead of silently leaving the panel."""
    lo, hi = np.array([EXTENT[0], EXTENT[2]]), np.array([EXTENT[1], EXTENT[3]])
    out = ~np.all((z >= lo) & (z <= hi), axis=1)
    return np.clip(z, lo, hi), int(out.sum())


def _pick_seed(folder, model, m):
    df = pd.read_csv(Path(folder)/"metrics.csv")
    q = df[(df.model == model) & (df.m == m) & df.knn_chart_ratio.notna()]
    if "warm_start" in q:
        q = q[q.warm_start.isna()]
    q = q.sort_values("knn_chart_ratio")
    return int(q.iloc[(len(q)-1)//2].seed)


def _code(folder, seed, model, m):
    first = int(pd.read_csv(Path(folder)/"metrics.csv").m.dropna().min())
    name = model
    if model in ("plainae", "plainvae"):
        m = first
    if model.endswith("_warm") and m == first:
        name = model[:-5]
    p = Path(folder)/f"seed_{seed}"/f"{name}_m{m}.npz"
    return np.load(p) if p.exists() else None


def dmap_dense(u, beta, alpha=1., k=1024):
    """Same operator as spectral.SpectralMap (Coifman-Lafon alpha, symmetric form), all points."""
    d2 = np.maximum((u*u).sum(1)[:, None]+(u*u).sum(1)[None]-2*u@u.T, 0)
    K = np.exp(-beta*d2)
    q = K.sum(1)
    w = q**-alpha
    deg = w*(K@w)
    sc = w/np.sqrt(deg)
    S = sc[:, None]*K*sc[None]
    vals, vecs = np.linalg.eigh(S)
    o = np.argsort(vals)[::-1][1:k+1]
    pi = deg/deg.sum()
    return vecs[:, o]/np.sqrt(pi)[:, None], vals[o]


def embeddings(x, ds, method, seed=0):
    out = {}
    if method == "imap":
        # Isomap = classical MDS of kNN-graph geodesics.  Computed once for all d (nested coordinates);
        # negative eigenvalues of the double-centred matrix (non-Euclidean geodesics) are clipped to 0,
        # instead of failing as sklearn does when many components are requested.
        from scipy.sparse.csgraph import shortest_path
        from sklearn.neighbors import kneighbors_graph
        G = shortest_path(kneighbors_graph(x, 10, mode="distance"), directed=False)
        G[~np.isfinite(G)] = G[np.isfinite(G)].max()
        n = len(G)
        J = np.eye(n) - 1/n
        vals, vecs = np.linalg.eigh(-.5*J @ (G**2) @ J)
        o = np.argsort(vals)[::-1]
        emb = vecs[:, o]*np.sqrt(np.maximum(vals[o], 0))
        for d in ds:
            out[d] = emb[:, :d]
    elif method == "umap":
        import umap
        for d in ds:
            out[d] = umap.UMAP(n_components=d, n_neighbors=15, min_dist=.1, random_state=seed).fit_transform(x)
    return out


def main(a):
    from data import load_data
    from geometry import base_beta
    from latent_figure import similarity_align
    out = Path(a.out); (out/"img").mkdir(parents=True, exist_ok=True); (out/"data").mkdir(exist_ok=True)
    main_dir = Path(a.random)
    seed = a.seed if a.seed is not None else _pick_seed(main_dir, "ae_warm", 64)
    meta = dict(seed=seed, extent=EXTENT, panel_inches=PANEL_IN, rows={}, notes=[])
    d = np.load(main_dir/f"seed_{seed}"/"data.npz")
    x, truth, train, test = d["x"], d["truth"], d["train"], d["test"]
    t = truth[test]
    cols = [2, 4, 8, 16, 64]
    _panel(out/"img"/"fig3_truth.png", t, t[:, 0])
    # ---- methods recomputed here on the SAME replicate (in-sample embedding, train-fitted readout)
    xs = (x-x[train].mean(0))/np.sqrt(np.mean((x[train]-x[train].mean(0))**2))
    for meth in a.methods:
        try:
            E = embeddings(xs, cols, meth, seed)
        except ImportError as e:
            meta["notes"].append(f"{meth} skipped: {e}"); continue
        meta["rows"][meth] = {}
        for m in cols:
            z, nout = _clip(_affine(E[m][train], truth[train], E[m][test]))
            _panel(out/"img"/f"fig3_{meth}_{m}.png", z, t[:, 0])
            meta["rows"][meth][m] = dict(outside=nout, mse=float(np.mean((_affine(E[m][train], truth[train], E[m][test])-t)**2)))
    # ---- DMAP oracle and trained models from the experiment folders
    rows = [("dmap", main_dir, "oracle")] + [(r, Path(f), mdl) for r, f, mdl in a.rows]
    for row, folder, model in rows:
        meta["rows"][row] = {}
        for m in cols:
            p = _code(folder, seed, model, m) if model != "oracle" else np.load(folder/f"seed_{seed}"/f"oracle_m{m}.npz")
            if p is None:
                continue
            dd = np.load(folder/f"seed_{seed}"/"data.npz")
            if model == "oracle":
                z = p["prediction"]
            else:
                z = _affine(p["code_train"], dd["truth"][dd["train"]], p["code"])
            z, nout = _clip(z)
            _panel(out/"img"/f"fig3_{row}_{m}.png", z, dd["truth"][dd["test"]][:, 0])
            meta["rows"][row][m] = dict(outside=nout)
    # ---- rings: oracle points + latent aligned by a similarity (E(2) + scale)
    for set_name, set_rows in (("random", a.rings_random), ("grid", a.rings_grid)):
        for row, folder, model in set_rows:
            folder = Path(folder)
            if not (folder/"metrics.csv").exists():
                continue
            s_ = a.seed if a.seed is not None else seed
            if not (folder/f"seed_{s_}").exists():
                s_ = _pick_seed(folder, model, 64) if model not in ("plainae", "plainvae") else 0
            dd = np.load(folder/f"seed_{s_}"/"data.npz")
            tt = dd["truth"][dd["test"]]
            pick = np.random.default_rng(0).choice(len(tt), min(a.ring_points, len(tt)), replace=False)
            for m in a.ring_modes:
                o = folder/f"seed_{s_}"/f"oracle_m{m}.npz"
                p = _code(folder, s_, model, m)
                if p is None or not o.exists():
                    continue
                r = np.load(o)["prediction"]
                za, _, _ = similarity_align(p["code"], r)
                err = float(np.mean((za-r)**2))
                rp, zp = r[pick], za[pick]
                zc, nout = _clip(zp)
                segs = np.stack([rp, zc], 1)
                _panel(out/"img"/f"rings_{set_name}_{row}_m{m}.png", zc, tt[pick, 0], rings=rp, segs=segs, size=3.0)
                meta.setdefault("rings", {}).setdefault(set_name, {}).setdefault(row, {})[m] = dict(seed=s_, err=err, outside=nout)
    # ---- readout error vs d, in-sample (as in the reference figure)
    if a.readout:
        ds = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024]
        Q = truth - truth.mean(0)
        def rel(E):
            Qh = _affine(E, truth, E)
            return float(np.sum((truth-Qh)**2)/np.sum(Q**2))
        res = {"d": ds}
        psi, _ = dmap_dense(xs, a.beta_factor*base_beta(xs[train]), k=max(ds))
        res["dmap"] = [rel(psi[:, :d]) for d in ds]
        for meth, dmax in (("imap", 1024), ("umap", a.umap_dmax)):
            if meth not in a.methods:
                res[meth] = [np.nan]*len(ds); continue
            try:
                E = embeddings(xs, [d for d in ds if d <= dmax], meth, seed)
                res[meth] = [rel(E[d]) if d in E else np.nan for d in ds]
            except ImportError as e:
                res[meth] = [np.nan]*len(ds); meta["notes"].append(f"readout {meth}: {e}")
        pd.DataFrame(res).to_csv(out/"data"/"readout_vs_d.dat", sep=" ", index=False, float_format="%.6g", na_rep="nan")
    (out/"data"/"fig3_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps(meta, indent=1, default=float)[:3000])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--random", required=True, help="results folder with ae_warm (e.g. results_swiss_final_AE_l0.1)")
    p.add_argument("--rows", nargs=3, action="append", default=[], metavar=("ROW", "FOLDER", "MODEL"),
                   help="extra rows for the chart figure, e.g. ae_jac results_swiss_final_AE_l0.1 ae_warm")
    p.add_argument("--rings-random", nargs=3, action="append", default=[], metavar=("ROW", "FOLDER", "MODEL"))
    p.add_argument("--rings-grid", nargs=3, action="append", default=[], metavar=("ROW", "FOLDER", "MODEL"))
    p.add_argument("--ring-modes", nargs="+", type=int, default=[4, 8, 16, 64])
    p.add_argument("--ring-points", type=int, default=250)
    p.add_argument("--methods", nargs="*", default=["imap", "umap"])
    p.add_argument("--readout", action="store_true")
    p.add_argument("--umap-dmax", type=int, default=64)
    p.add_argument("--beta-factor", type=float, default=64.)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="/kaggle/working/paper_figs")
    main(p.parse_args())
