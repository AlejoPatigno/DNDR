"""Swiss roll WITH VOLUME (dataset "swissvol"): theory, oracle readout scan and 3-D latent metrics.

Theory (flat box approximation of the thick roll, Neumann Laplacian on [0,W] x [0,H] x [0,T]):
eigenfunctions cos(k pi s/W) cos(l pi h/H) cos(j pi t'/T), eigenvalues pi^2 [(k/W)^2 + (l/H)^2 + (j/T)^2].
The first mode that depends on h is (0,1,0) and the first that depends on t is (0,0,1); every mode
with a smaller eigenvalue is a function of the other coordinates only, so an affine readout of a
spectral PREFIX that stops before them carries no information about h (resp. t).  0-based psi index
(psi_0 = first non-trivial mode):
    h-mode index = #{k >= 1 : k/W < 1/H}                                    (5 for 60 x 10)
    t-mode index = #{(k,l) != (0,0) : (k/W)^2 + (l/H)^2 < 1/T^2}             (129 for T = 2)
The roll bends the sheet (metric (1 + kappa t)^2 ds^2): this couples t to s, so the thickness mode mixes
with the (k, l, 0) modes of nearby eigenvalue and the t information is spread over a BAND of modes around
the predicted index.  The notebook measures, per axis, the share of variance carried by each mode.
"""
import numpy as np
import pandas as pd
from pathlib import Path

AXES = ("s", "h", "t")
SHOW_D = (8, 64, 128, 160, 256, 512)          # columns of the reconstruction figure (part A)
PROJ = {"sh": (0, 1), "st": (0, 2)}           # projections drawn for a 3-D chart / latent


# ----------------------------------------------------------------------------- theory
def expected_positions(width=60., height=10., thickness=2.):
    ks = np.arange(0, int(4*width/min(height, thickness))+2)
    ls = np.arange(0, int(4*height/thickness)+2)
    K, L = np.meshgrid(ks, ls, indexing="ij")
    lam = (K/width)**2 + (L/height)**2
    return dict(h_mode=int(((ks/width < 1/height) & (ks > 0)).sum()),
                t_mode=int(((lam < 1/thickness**2) & (lam > 0)).sum()))


def box_spectrum(width, height, thickness, n):
    """First n non-trivial Neumann eigenvalues of the box (units 1/length^2) with their (k,l,j)."""
    kmax = int(np.ceil(np.sqrt(n)*3))
    k, l, j = np.meshgrid(np.arange(kmax), np.arange(kmax), np.arange(kmax), indexing="ij")
    lam = np.pi**2*((k/width)**2 + (l/height)**2 + (j/thickness)**2)
    o = np.argsort(lam.ravel(), kind="stable")[1:n+1]
    return lam.ravel()[o], np.stack([k.ravel()[o], l.ravel()[o], j.ravel()[o]], 1)


# ----------------------------------------------------------------------------- embeddings
def _eigh(a, k, largest=True):
    try:
        import torch
        if torch.cuda.is_available():
            t = torch.as_tensor(a, dtype=torch.float64, device="cuda")
            vals, vecs = torch.linalg.eigh(t)
            vals, vecs = vals.cpu().numpy(), vecs.cpu().numpy()
            del t
            torch.cuda.empty_cache()
        else:
            raise ImportError
    except (ImportError, AttributeError, RuntimeError):
        vals, vecs = np.linalg.eigh(a)
    o = np.argsort(vals)[::-1] if largest else np.argsort(vals)
    return vals[o][:k], vecs[:, o][:, :k]


def dmap_dense(u, beta, k, alpha=1.):
    """Dense diffusion map, same operator as spectral.SpectralMap (Coifman-Lafon alpha, symmetric form)."""
    g = (u*u).sum(1)
    K = np.exp(-beta*np.maximum(g[:, None]+g[None]-2*u@u.T, 0))
    w = K.sum(1)**-alpha
    deg = w*(K@w)
    sc = w/np.sqrt(deg)
    vals, vecs = _eigh(sc[:, None]*K*sc[None], k+1)
    pi = deg/deg.sum()
    return vecs[:, 1:]/np.sqrt(pi)[:, None], vals[1:]


def isomap_dense(u, k, nn=10):
    """Classical scaling of kNN-graph geodesics (nested in d); negative eigenvalues clipped to 0."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    from scipy.spatial import cKDTree
    d, j = cKDTree(u).query(u, nn+1)
    n = len(u)
    graph = csr_matrix((d[:, 1:].ravel(), (np.repeat(np.arange(n), nn), j[:, 1:].ravel())), shape=(n, n))
    G = shortest_path(graph, directed=False)
    G[~np.isfinite(G)] = G[np.isfinite(G)].max()
    n = len(G)
    G2 = G**2
    B = -.5*(G2 - G2.mean(0)[None] - G2.mean(1)[:, None] + G2.mean())
    del G, G2
    vals, vecs = _eigh(B, k)
    return vecs*np.sqrt(np.maximum(vals, 0))


def umap_embed(u, d, seed=0):
    import umap
    try:
        return umap.UMAP(n_components=d, n_neighbors=15, min_dist=.1, random_state=seed).fit_transform(u)
    except Exception:
        return umap.UMAP(n_components=d, n_neighbors=15, min_dist=.1, random_state=seed, init="random").fit_transform(u)


def readout(U, Q):
    A = np.column_stack([U, np.ones(len(U))])
    L = np.linalg.lstsq(A, Q, rcond=None)[0]
    return A@L, L[:-1]


def axis_errors(Qh, Q):
    """Relative squared Frobenius error and per-axis error / Var(axis) (0 = perfect, 1 = no information)."""
    out = {"rel": float(np.sum((Q-Qh)**2)/np.sum((Q-Q.mean(0))**2))}
    for i, a in enumerate(AXES):
        out[a] = float(np.mean((Q[:, i]-Qh[:, i])**2)/Q[:, i].var())
    return out


def mode_positions(psi, Q, share=.01):
    """Which modes carry each chart axis.  With A = [1, psi_0, psi_1, ...] and its QR factor q, the share
    of Var(axis) explained by mode n given the earlier modes is c_n^2 = (q_n . y)^2 / |y|^2, and the
    affine readout error of the prefix psi_0..psi_{d-1} is 1 - sum_{n<d} c_n^2.  Reported per axis:
    first mode with share > `share`, the mode with the largest share, and d_half = the smallest prefix
    whose error is below 0.5.  Returns (summary, shares[axis] array)."""
    q, _ = np.linalg.qr(np.column_stack([np.ones(len(psi)), psi]))
    out, shares = {}, {}
    for i, a in enumerate(AXES):
        y = Q[:, i]-Q[:, i].mean()
        c = ((q.T @ y)**2/(y @ y))[1:]
        shares[a] = c
        err = 1-np.cumsum(c)
        hit = np.where(c > share)[0]
        out[f"{a}_first_mode"] = int(hit[0]) if len(hit) else -1
        out[f"{a}_max_mode"] = int(np.argmax(c))
        out[f"{a}_max_share"] = float(c.max())
        out[f"{a}_d_half"] = int(np.argmax(err < .5))+1 if (err < .5).any() else -1
    return out, shares


# ----------------------------------------------------------------------------- oracle scan
def readout_scan(cfg, data_seeds, scan, beta_factor, methods=("dmap", "imap"), umap_dmax=0, log=print):
    """In-sample affine readout of (s, h, t) from d coordinates of each method, per data seed.
    Inputs standardised with training statistics; beta = factor x median heuristic on training pairs,
    exactly as in experiment.py.  Returns (long table, per-seed extras for figures)."""
    from data import load_data
    from geometry import base_beta
    import time
    rows, extras = [], {}
    for ds in data_seeds:
        x, f, Q, (train, val, test), meta = load_data(dict(cfg, data_seed=ds, split_seed=ds))
        xm = x[train].mean(0)
        u = (x-xm)/np.sqrt(np.mean((x[train]-xm)**2))
        beta = beta_factor*base_beta(u[train])
        kmax = max(scan)
        emb, ex = {}, dict(Q=Q, x=x, meta=meta, beta=beta)
        t0 = time.time()
        if "dmap" in methods:
            psi, lam = dmap_dense(u, beta, kmax)
            emb["dmap"] = lambda d, p=psi: p[:, :d]
            ex.update(psi=psi[:, :min(kmax, 512)], lam=lam)
            pos, shares = mode_positions(psi, Q)
            ex.update(positions=pos, shares=shares)
            log(f"data_seed={ds}: DMAP {time.time()-t0:.0f}s | modos medidos {pos}")
        if "imap" in methods:
            t0 = time.time()
            iso = isomap_dense(u, kmax)
            emb["imap"] = lambda d, e=iso: e[:, :d]
            log(f"data_seed={ds}: IMAP {time.time()-t0:.0f}s")
        if umap_dmax:
            try:
                um = {}
                for d in [d for d in scan if d <= umap_dmax]:
                    t0 = time.time(); um[d] = umap_embed(u, d, ds); log(f"data_seed={ds}: UMAP d={d} {time.time()-t0:.0f}s")
                emb["umap"] = lambda d, um=um: um.get(d)
            except ImportError as e:
                log(f"UMAP omitido: {e}")
        for m, E in emb.items():
            for d in scan:
                U = E(d)
                if U is None:
                    continue
                Qh, L = readout(U, Q)
                rows.append(dict(data_seed=ds, method=m, d=d, **axis_errors(Qh, Q)))
                if d in SHOW_D or d == 3:
                    ex[f"recon_{m}_{d}"] = Qh
                if m == "dmap" and d == kmax:
                    ex["L"] = L
        extras[ds] = ex
    return pd.DataFrame(rows), extras


# ----------------------------------------------------------------------------- trained models (3-D latent)
def latent_vs_oracle_table_3d(folder, models):
    """As latent_figure.latent_vs_oracle_table, for a latent of any dimension q equal to the chart's:
    similarity alignment (orthogonal Procrustes + one scale) of the latent onto the oracle points and onto
    the true chart, errors per axis normalised by the variance of that axis."""
    from latent_figure import similarity_align, _code_test
    folder = Path(folder)
    df = pd.read_csv(folder/"metrics.csv")
    first_m = int(df.m.dropna().min())
    modes = sorted(int(v) for v in df.query("model == 'oracle'").m.unique())
    out = []
    for model in models:
        rows = df[(df.model == model) & df.m.notna()]
        if "warm_start" in rows:
            rows = rows.drop_duplicates(["seed", "m"])
        for seed, m in rows[["seed", "m"]].itertuples(index=False):
            seed, m = int(seed), int(m)
            z = _code_test(folder, seed, model, m, first_m)
            if z is None:
                continue
            d = np.load(folder/f"seed_{seed}"/"data.npz")
            t = d["truth"][d["test"]]
            var = t.var(0)
            zt = similarity_align(z, t)[0]
            for mo in (modes if model in ("plainae", "plainvae") else [m]):
                r = _code_test(folder, seed, "oracle", mo, first_m)
                if r is None:
                    continue
                zo = similarity_align(z, r)[0]
                row = dict(seed=seed, model=model, m=mo)
                for i, a in enumerate(AXES):
                    row[f"latent_vs_oracle_{a}_norm"] = float(((zo-r)[:, i]**2).mean()/var[i])
                    row[f"latent_vs_truth_{a}_norm"] = float(((zt-t)[:, i]**2).mean()/var[i])
                    row[f"oracle_vs_truth_{a}_norm"] = float(((r-t)[:, i]**2).mean()/var[i])
                row["latent_vs_oracle"] = float(np.mean((zo-r)**2))
                out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- panels (transparent, data only)
def panel(path, pts, color, extent, size=(1.6, 1.15), s=1.0, vmin=0., vmax=60., cmap=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("viridis_cut", plt.cm.viridis(np.linspace(.14, 1., 256)))
    fig = plt.figure(figsize=size, dpi=300)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.patch.set_alpha(0)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    p = np.column_stack([np.clip(pts[:, 0], extent[0], extent[1]), np.clip(pts[:, 1], extent[2], extent[3])])
    ax.scatter(p[:, 0], p[:, 1], s=s, c=color, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
    fig.savefig(path, transparent=True, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- figures: predicted vs true chart
def _wrap(label, width=18):
    import textwrap
    return "\n".join(textwrap.wrap(label, width))


def _limits(ref, k):
    i, j = PROJ[k]
    lo, hi = ref[:, [i, j]].min(0), ref[:, [i, j]].max(0)
    pad = .08*(hi-lo)
    return (lo[0]-pad[0], hi[0]+pad[0]), (lo[1]-4*pad[1], hi[1]+4*pad[1])


COLOUR = {"sh": (0, "viridis"), "st": (2, "plasma")}   # (s, h): colour = s ; (s, t): colour = t


def _scatter(ax, z, ref, k, size=1.2):
    i, j = PROJ[k]
    ci, cmap = COLOUR[k]
    (x0, x1), (y0, y1) = _limits(ref, k)
    inside = (z[:, i] >= x0) & (z[:, i] <= x1) & (z[:, j] >= y0) & (z[:, j] <= y1)
    ax.scatter(np.clip(z[:, i], x0, x1), np.clip(z[:, j], y0, y1), c=ref[:, ci], s=size, cmap=cmap,
               vmin=ref[:, ci].min(), vmax=ref[:, ci].max(), rasterized=True, linewidths=0)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])
    if (~inside).any():
        ax.text(.98, .02, f"{(~inside).sum()} fuera", fontsize=5, ha="right", transform=ax.transAxes, color="C3")


def _err_label(ax, z, ref):
    e = ((z-ref)**2).mean(0)/ref.var(0)
    ax.text(.02, .02, " ".join(f"{a}:{v:.2g}" for a, v in zip(AXES, e)), fontsize=5.5, transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=.75, lw=0, pad=1))


def readout_grid(ex, methods, path, ds=SHOW_D, title=None):
    """Part A, like Figure 3 of the paper in 3-D: truth and the affine reconstructions Q_hat of each method,
    two rows per method (projections s-h and s-t), one column per d.  Colour = true s.  Corner: error per
    axis / Var(axis)."""
    import matplotlib.pyplot as plt
    Q = ex["Q"]
    methods = [m for m in methods if any(f"recon_{m}_{d}" in ex for d in ds)]
    fig, axes = plt.subplots(2*len(methods), len(ds)+1, figsize=(1.9*(len(ds)+1), 1.25*2*len(methods)+.5), squeeze=False)
    for r, m in enumerate(methods):
        for p, k in enumerate(PROJ):
            row = axes[2*r+p]
            _scatter(row[0], Q, Q, k)
            for c, d in enumerate(ds, 1):
                z = ex.get(f"recon_{m}_{d}")
                if z is None:
                    row[c].axis("off"); continue
                _scatter(row[c], z, Q, k)
                if p == 0:
                    _err_label(row[c], z, Q)
            row[0].set_ylabel(f"{m.upper()}\n({k[0]}, {k[1]})", fontsize=8)
    for c, d in enumerate(("truth",) + tuple(ds)):
        axes[0][c].set_title("truth" if d == "truth" else f"$d = {d}$", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"), dpi=200); fig.savefig(Path(path).with_suffix(".pdf"), dpi=200)
    return fig


def latent_3d(folder, seed, model, m, first_m):
    """Latent of a trained model mapped to the chart by the affine readout fitted on TRAIN (as the 2-D
    latent figure), or the oracle points; returns (z_test_in_chart_units, truth_test)."""
    from latent_figure import _stem
    folder = Path(folder)
    d = np.load(folder/f"seed_{seed}"/"data.npz")
    truth, train, test = d["truth"], d["train"], d["test"]
    if model in ("plainae", "plainvae"):
        m = first_m
    path = _stem(folder, seed, model, m, first_m)
    if not path.exists():
        return None, truth[test]
    p = np.load(path)
    if model == "oracle":
        return p["prediction"], truth[test]
    A = np.column_stack([p["code_train"], np.ones(len(train))])
    W = np.linalg.lstsq(A, truth[train], rcond=None)[0]
    return np.column_stack([p["code"], np.ones(len(p["code"]))]) @ W, truth[test]


def latent_grid_3d(rows, modes, path, title=None, seed=None):
    """'truth / m = ...' figure for a 3-D latent: for each model two rows (projections s-h and s-t) with
    the latent mapped to the chart by an affine map fitted on the training points; colour = true s.
    rows: dicts(label, folder, model) with model 'oracle' for the DMAP affine readout."""
    import matplotlib.pyplot as plt
    from latent_figure import representative_seed
    fig, axes = plt.subplots(2*len(rows), len(modes)+1, figsize=(1.9*(len(modes)+1), 1.25*2*len(rows)+.5), squeeze=False)
    for r, row in enumerate(rows):
        folder = Path(row["folder"])
        df = pd.read_csv(folder/"metrics.csv")
        first_m = int(df.m.dropna().min())
        plain = row["model"] in ("plainae", "plainvae")
        s_ = seed if seed is not None else representative_seed(df, row["model"], first_m if plain else max(modes))[0]
        for p, k in enumerate(PROJ):
            ax_row = axes[2*r+p]
            if s_ is None:
                for ax in ax_row:
                    ax.axis("off")
                continue
            _, ref = latent_3d(folder, s_, "oracle", modes[0], first_m)
            _scatter(ax_row[0], ref, ref, k)
            for c, m in enumerate(modes, 1):
                z, ref = latent_3d(folder, s_, row["model"], m, first_m)
                if z is None:
                    ax_row[c].text(.5, .5, "—", ha="center", va="center", transform=ax_row[c].transAxes)
                    ax_row[c].set_xticks([]); ax_row[c].set_yticks([]); continue
                _scatter(ax_row[c], z, ref, k)
                if p == 0:
                    _err_label(ax_row[c], z, ref)
            ax_row[0].set_ylabel(_wrap(row['label']) + f"\n({k[0]}, {k[1]}) · rép. {s_}", fontsize=6)
    for c, m in enumerate(("truth",) + tuple(modes)):
        axes[0][c].set_title("truth" if m == "truth" else f"$m = {m}$", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"), dpi=200); fig.savefig(Path(path).with_suffix(".pdf"), dpi=200)
    return fig


def rings_3d(rows, modes, path, title=None, seed=None, max_points=300):
    """Rings in 3-D: oracle points (open circles) and the latent aligned onto them by a similarity map
    (rotation/reflection + one scale, dots), joined by grey segments, in the projections s-h and s-t."""
    import matplotlib.pyplot as plt
    from latent_figure import representative_seed, similarity_align, _code_test
    fig, axes = plt.subplots(2*len(rows), len(modes), figsize=(2.4*len(modes), 1.3*2*len(rows)+.5), squeeze=False)
    rng = np.random.default_rng(0)
    for r, row in enumerate(rows):
        folder = Path(row["folder"])
        df = pd.read_csv(folder/"metrics.csv")
        first_m = int(df.m.dropna().min())
        plain = row["model"] in ("plainae", "plainvae")
        s_ = seed if seed is not None else representative_seed(df, row["model"], first_m if plain else max(modes))[0]
        if s_ is None:
            continue
        d = np.load(folder/f"seed_{s_}"/"data.npz")
        t = d["truth"][d["test"]]
        pick = rng.choice(len(t), min(max_points, len(t)), replace=False)
        col = plt.cm.viridis((t[pick, 0]-t[:, 0].min())/np.ptp(t[:, 0]))
        for c, m in enumerate(modes):
            rr = _code_test(folder, s_, "oracle", m, first_m)
            z = _code_test(folder, s_, row["model"], first_m if plain else m, first_m)
            for p, k in enumerate(PROJ):
                ax = axes[2*r+p][c]
                ax.set_xticks([]); ax.set_yticks([])
                if rr is None or z is None:
                    ax.text(.5, .5, "—", ha="center", va="center", transform=ax.transAxes); continue
                za = similarity_align(z, rr)[0]
                i, j = PROJ[k]
                (x0, x1), (y0, y1) = _limits(t, k)
                zc = np.column_stack([np.clip(za[:, i], x0, x1), np.clip(za[:, j], y0, y1)])
                for q in pick:
                    ax.plot([rr[q, i], zc[q, 0]], [rr[q, j], zc[q, 1]], color="0.75", lw=.3, zorder=1)
                ax.scatter(rr[pick, i], rr[pick, j], s=8, facecolors="none", edgecolors=col, linewidths=.5, zorder=2)
                ax.scatter(zc[pick, 0], zc[pick, 1], s=2, c=t[pick, 0], cmap="viridis", zorder=3, linewidths=0)
                ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
                if p == 0:
                    e = ((za-rr)**2).mean(0)/t.var(0)
                    ax.text(.02, .02, " ".join(f"{a}:{v:.2g}" for a, v in zip(AXES, e)), fontsize=5.5,
                            transform=ax.transAxes, bbox=dict(facecolor="white", alpha=.75, lw=0, pad=1))
                if c == 0:
                    ax.set_ylabel(_wrap(row['label']) + f"\n({k[0]}, {k[1]}) · rép. {s_}", fontsize=6)
                if r == 0 and p == 0:
                    ax.set_title(f"$m = {m}$", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"), dpi=200); fig.savefig(Path(path).with_suffix(".pdf"), dpi=200)
    return fig
