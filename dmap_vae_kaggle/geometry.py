"""NumPy-only geometry: exact Swiss-roll chart map, alignment metrics, generation metrics and
bandwidth (beta) diagnostics.  No torch import, so it is testable anywhere.

Conventions
-----------
* Chart errors are per chart coordinate:   chart_mse_z = ||z_hat - z||^2 / 2  (mean over points).
* Ambient errors are per ambient coordinate: mse_x = ||x_hat - x||^2 / 3        (mean over points).
* beta always acts on the SAME standardized inputs the spectral map sees (u = (x - mean)/scale),
  so diagnostics and the fitted operator use one kernel.
"""
from functools import lru_cache
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- exact Swiss roll

THETA0, THETA1, THETA_GRID = 1.5*np.pi, 100., 200000


@lru_cache(None)
def _swiss_table():
    th = np.linspace(THETA0, THETA1, THETA_GRID)
    u = .5 + .6*th
    s = (u*np.sqrt(u*u+.36)+.36*np.arcsinh(u/.6))/1.2   # arclength of r = .5 + .6 theta
    return s - s[0], th


def swiss_embed(z):
    """g: [0,W]x[0,H] -> R^3, isometric up to the interpolation error of theta(s)."""
    z = np.asarray(z, dtype=np.float64)
    s, th = _swiss_table()
    theta = np.interp(z[:, 0], s, th)
    r = .5 + .6*theta
    return np.column_stack([r*np.cos(theta), z[:, 1], r*np.sin(theta)])


def swiss_volume_embed(z):
    """Thick Swiss roll: (s, h, t) -> centre surface g(s, h) + t * n(s), with n the unit outward normal
    of the spiral in the (x, z) plane.  s is arc length on the centre spiral, so the map is isometric
    along the centre (t = 0) and stretches s by (1 + kappa t) off it: the 3-D chart is exact but not
    an isometry.  The spiral advances 2 pi 0.6 = 3.77 per turn, so thicknesses below ~3 do not overlap."""
    z = np.asarray(z, dtype=np.float64)
    s, th = _swiss_table()
    theta = np.interp(z[:, 0], s, th)
    r = .5 + .6*theta
    dx, dz = .6*np.cos(theta) - r*np.sin(theta), .6*np.sin(theta) + r*np.cos(theta)
    norm = np.sqrt(dx*dx + dz*dz)
    nx, nz = dz/norm, -dx/norm                       # tangent rotated by -90 degrees: outward
    t = z[:, 2]
    return np.column_stack([r*np.cos(theta) + t*nx, z[:, 1], r*np.sin(theta) + t*nz])


def project_rectangle(z, width, height):
    """Euclidean projection onto the convex chart domain (non-expansive)."""
    return np.column_stack([np.clip(z[:, 0], 0, width), np.clip(z[:, 1], 0, height)])


def lipschitz_ratio(a, b):
    """max ||g(a)-g(b)|| / ||a-b|| over given pairs; 1 up to interpolation error for an isometry."""
    num = np.linalg.norm(swiss_embed(a)-swiss_embed(b), axis=1)
    den = np.linalg.norm(a-b, axis=1)
    keep = den > 1e-9
    return float(np.max(num[keep]/den[keep])) if keep.any() else 1.


def oracle_exact_swiss(z_hat, z, width, height):
    """Affine oracle -> exact roll.  MSE_X <= (2/3) chart MSE_z (g 1-Lipschitz, projection
    onto a convex set non-expansive).  The empirical Lipschitz ratio certifies the numerics."""
    zp = project_rectangle(z_hat, width, height)
    x_hat, x = swiss_embed(zp), swiss_embed(z)
    chart = float(np.mean(np.sum((z_hat-z)**2, 1))/2)
    return dict(oracle_exact_mse_x=float(np.mean(np.sum((x_hat-x)**2, 1))/3),
                oracle_exact_bound_x=2*chart/3,
                oracle_lipschitz_ratio=lipschitz_ratio(zp, z))


def torus_embed(y):
    """Exact flat torus map from a circular 4-vector (projected to the two unit circles)."""
    a = np.stack([np.arctan2(y[:, 1], y[:, 0]), np.arctan2(y[:, 3], y[:, 2])], 1)
    return np.column_stack([np.cos(a[:, 0]), np.sin(a[:, 0]), .6*np.cos(a[:, 1]), .6*np.sin(a[:, 1])])

# --------------------------------------------------------------------------- chart alignment


def _orthogonal(u, v):
    left, _, right = np.linalg.svd(u.T @ v)
    return left @ right   # O(d): latent orientation (incl. reflection) is not identifiable


def chart_alignment(code_train, chart_train, code_test, chart_test):
    """Fit on TRAIN only, evaluate on TEST.  Returns per-chart-coordinate MSEs.
    rigid: rotation/reflection + translation   (recovers scale and shape)
    similarity: + one global scale             (recovers shape)
    affine: general linear map                 (linear deformation diagnostic, NOT an isometry)"""
    if code_train.shape[1] != chart_train.shape[1]:
        return {}
    cm, zm = code_train.mean(0), chart_train.mean(0)
    u, v = code_train-cm, chart_train-zm
    q = _orthogonal(u, v)
    scale = np.trace((u@q).T @ v)/max(np.sum(u*u), 1e-300)
    a = np.linalg.lstsq(np.column_stack([u, np.ones(len(u))]), chart_train, rcond=None)[0]
    t = code_test-cm
    err = lambda p: float(np.mean(np.sum((p-chart_test)**2, 1))/chart_test.shape[1])
    return dict(rigid_chart_mse=err(t@q+zm), similarity_chart_mse=err(scale*(t@q)+zm),
                similarity_scale=float(scale),
                affine_chart_mse=err(np.column_stack([t, np.ones(len(t))])@a),
                constant_chart_mse=err(np.broadcast_to(zm, chart_test.shape)))

# --------------------------------------------------------------------------- generation metrics


class SwissSurface:
    """Dense sample of the exact surface for distance-to-surface and chart-coverage metrics.
    Resolution h gives distance error <= h/sqrt(2) (reported as `surface_resolution`)."""

    def __init__(self, width=60., height=10., h=.05):
        s = np.linspace(0, width, int(round(width/h))+1)
        t = np.linspace(0, height, int(round(height/h))+1)
        ss, tt = np.meshgrid(s, t, indexing="ij")
        self.chart = np.column_stack([ss.ravel(), tt.ravel()])
        self.tree = cKDTree(swiss_embed(self.chart))
        self.width, self.height, self.h = width, height, h

    def metrics(self, x, reference_chart=None, cells=(30, 5), tv_cells=(12, 2), prefix=""):
        """coverage: fraction of fine chart cells hit (depends on N: compare at equal N).
        TV: total variation of the nearest-chart histogram on a coarse grid, against uniform and
        against a reference sample.  Always report the reference sample's own values."""
        x = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(x).all(1)
        d, i = self.tree.query(x[finite])
        c = self.chart[i]
        grid = lambda n: [np.linspace(0, self.width, n[0]+1), np.linspace(0, self.height, n[1]+1)]
        hist = np.histogram2d(c[:, 0], c[:, 1], bins=grid(cells))[0].ravel()
        coarse = np.histogram2d(c[:, 0], c[:, 1], bins=grid(tv_cells))[0].ravel()
        p = coarse/max(coarse.sum(), 1)
        out = {"surface_dist_mean": float(d.mean()), "surface_dist_median": float(np.median(d)),
               "surface_dist_q95": float(np.quantile(d, .95)),
               "chart_coverage": float((hist > 0).mean()),
               "chart_tv_to_uniform": float(.5*np.abs(p-1/len(p)).sum()),
               "n_points": int(finite.sum()), "surface_resolution": self.h/np.sqrt(2)}
        if reference_chart is not None:
            r = np.histogram2d(reference_chart[:, 0], reference_chart[:, 1], bins=grid(tv_cells))[0].ravel()
            out["chart_tv_to_reference"] = float(.5*np.abs(p-r/max(r.sum(), 1)).sum())
        return {prefix+k: v for k, v in out.items()}

# --------------------------------------------------------------------------- beta diagnostics


def base_beta(u, seed=7):
    """Median heuristic on the standardized training inputs (same as spectral.choose_beta)."""
    rng = np.random.default_rng(seed)
    i, j = rng.integers(len(u), size=4096), rng.integers(len(u), size=4096)
    d2 = np.sum((u[i]-u[j])**2, axis=1)
    return float(1/np.median(d2[d2 > 0]))


def _sqdist(a, b):
    return np.maximum((a*a).sum(1)[:, None]+(b*b).sum(1)[None, :]-2*a@b.T, 0)


def _subsample(u, max_points, seed):
    if len(u) <= max_points:
        return np.arange(len(u))
    return np.sort(np.random.default_rng(seed).choice(len(u), max_points, replace=False))


def graph_shortcut_mass(u, betas, k=10, ratio=2., max_points=4000, seed=7):
    """LABEL-FREE short-circuit diagnostic.
    Geodesics are approximated by shortest paths on a small symmetric kNN graph (Isomap-style,
    trusted only at scales where the kNN graph itself has no shortcuts).  For each beta returns
    the fraction of off-diagonal kernel mass on pairs whose graph distance exceeds `ratio` times
    their ambient distance, i.e. mass the kernel places across folds/sheets."""
    idx = _subsample(np.asarray(u, np.float64), max_points, seed)
    v = np.asarray(u, np.float64)[idx]
    dist, nn = cKDTree(v).query(v, k+1)
    rows = np.repeat(np.arange(len(v)), k)
    graph = csr_matrix((dist[:, 1:].ravel(), (rows, nn[:, 1:].ravel())), shape=(len(v),)*2)
    geo = shortest_path(graph, directed=False)
    d2 = _sqdist(v, v)
    bad = geo > ratio*np.sqrt(d2)+1e-12
    np.fill_diagonal(bad, False)
    out = []
    for b in np.atleast_1d(betas):
        kmat = np.exp(-b*d2)
        np.fill_diagonal(kmat, 0)
        out.append(float((kmat*bad).sum()/max(kmat.sum(), 1e-300)))
    return dict(graph_shortcut_mass=out, graph_disconnected_fraction=float(np.isinf(geo).mean()),
                graph_k=k, graph_ratio=ratio, graph_points=len(v))


def true_shortcut_mass(u, chart, betas, scale, ratio=2., slack=1., max_points=4000, seed=7):
    """DIAGNOSTIC ONLY (uses the true chart; never for selection).  Kernel mass on pairs whose
    chart distance exceeds ratio*ambient distance + slack, both in ORIGINAL units
    (ambient = scale * standardized distance).  u and chart share the index order."""
    idx = _subsample(u, max_points, seed)
    v, c = np.asarray(u, np.float64)[idx], np.asarray(chart, np.float64)[idx]
    d2 = _sqdist(v, v)
    bad = np.sqrt(_sqdist(c, c)) > ratio*scale*np.sqrt(d2)+slack
    out = []
    for b in np.atleast_1d(betas):
        kmat = np.exp(-b*d2)
        np.fill_diagonal(kmat, 0)
        out.append(float((kmat*bad).sum()/max(kmat.sum(), 1e-300)))
    return out


def beta_flow(u, betas, max_points=4000, seed=7):
    """Classical bandwidth-flow curve: local slope of log sum_ij K_ij against log(1/beta),
    reported as an effective dimension 2*dlogS/dlog(1/beta), with and without the diagonal.
    Reported for completeness: on the Swiss roll its peak lies INSIDE the short-circuit regime,
    so it is not used for selection."""
    idx = _subsample(u, max_points, seed)
    v = np.asarray(u, np.float64)[idx]
    d2 = _sqdist(v, v)
    n = len(v)
    betas = np.sort(np.atleast_1d(betas).astype(float))
    full = np.array([np.exp(-b*d2).sum() for b in betas])
    off = full-n
    lb = np.log(betas)
    return dict(betas=betas.tolist(),
                dimension_full=(-2*np.gradient(np.log(full), lb)).tolist(),
                dimension_offdiag=(-2*np.gradient(np.log(np.maximum(off, 1e-300)), lb)).tolist())


def mode_diagnostics(psi, chart, width, height, search=16):
    """DIAGNOSTIC ONLY (uses the true chart).  For an intrinsic Neumann spectrum on a W x H
    rectangle: psi_1 ~ cos(pi s/W) and the first height mode cos(pi h/H) appears at position
    1 + #{k>=1: k/W < 1/H}."""
    s, h = chart[:, 0], chart[:, 1]
    ref_s, ref_h = np.cos(np.pi*s/width), np.cos(np.pi*h/height)
    q = min(search, psi.shape[1])
    corr = lambda a, b: float(abs(np.corrcoef(a, b)[0, 1]))
    ch = [corr(psi[:, k], ref_h) for k in range(q)]
    return dict(psi1_corr_cos_s=corr(psi[:, 0], ref_s), height_mode_position=int(np.argmax(ch))+1,
                height_mode_corr=float(max(ch)),
                expected_height_mode_position=int(1+np.sum(np.arange(1, 10*int(width/height)+2)/width < 1/height)))


def select_beta(rows, threshold=1e-3, key="graph_shortcut_mass"):
    """Label-free rule: the SMALLEST beta factor (widest kernel, best statistics) whose
    worst-case (over data seeds) graph short-circuit mass is <= threshold and whose spectral
    gates passed for every data seed.  rows: dicts with factor, data_seed, ok, and `key`."""
    by = {}
    for r in rows:
        by.setdefault(r["factor"], []).append(r)
    admissible = []
    for f, rs in sorted(by.items()):
        if all(r.get("ok") for r in rs) and max(r[key] for r in rs) <= threshold:
            admissible.append(f)
    return dict(selected_factor=(admissible[0] if admissible else None), admissible=admissible,
                rule=f"min factor with max_seed({key}) <= {threshold} and all spectral gates passed",
                threshold=threshold)


def affine_fit(f, y, train, ridge=1e-6):
    """Ridge-stabilized affine readout fitted on TRAIN rows only; returns predictions for all rows."""
    fm, ym = f[train].mean(0), y[train].mean(0)
    a, b = f[train]-fm, y[train]-ym
    gram = a.T @ a
    strength = ridge*np.trace(gram)/len(gram)
    coef = np.linalg.solve(gram+strength*np.eye(len(gram)), a.T@b)
    return (f-fm)@coef+ym


def chart_target(dataset, truth):
    """Chart coordinates the affine oracle predicts: raw chart for the Swiss roll, circular
    4-vectors for angular charts (torus, alanine)."""
    if dataset in ("swiss", "swissvol"):
        return np.asarray(truth, np.float64)
    a = np.asarray(truth, np.float64)
    return np.stack([np.cos(a[:, 0]), np.sin(a[:, 0]), np.cos(a[:, 1]), np.sin(a[:, 1])], axis=1)


def knn_chart_mse(code_train, chart_train, code_test, chart_test, k=5):
    """Nonlinear chart recovery: kNN regression latent -> chart, fitted on TRAIN, scored on TEST
    (per chart coordinate).  Small <=> the latent is an unfolded chart (injective, continuous)
    up to ANY smooth reparametrization; rigid/affine alignment cannot see this because a
    nonlinear decoder may use any reparametrization.  Codes are standardized with TRAIN stats.
    Compare with knn_chart_floor(): the same estimator applied to the true chart."""
    code_train, code_test = np.asarray(code_train, np.float64), np.asarray(code_test, np.float64)
    mu, sd = code_train.mean(0), np.maximum(code_train.std(0), 1e-12)
    a, b = (code_train-mu)/sd, (code_test-mu)/sd
    _, idx = cKDTree(a).query(b, k)
    pred = np.asarray(chart_train)[idx].mean(1)
    return float(np.mean(np.sum((pred-chart_test)**2, 1))/np.asarray(chart_test).shape[1])


def knn_chart_floor(chart_train, chart_test, k=5):
    """Finite-sample floor of knn_chart_mse: the estimator run on the true chart itself."""
    return knn_chart_mse(chart_train, chart_train, chart_test, chart_test, k)


# --------------------------------------------------------------------------- per-dataset rules


def degeneracy_defect(values, pairs=3):
    """LABEL-FREE, spectrum only.  On a flat product torus every nonzero frequency gives a
    (cos, sin) pair of EXACTLY degenerate eigenvalues.  Defect = largest split WITHIN the first
    `pairs` pairs divided by the smallest gap BETWEEN consecutive pairs (0 = pairs resolved)."""
    lam = np.asarray(values, float)
    if len(lam) < 2*pairs + 1:
        return float("nan")
    within = [abs(lam[2*i] - lam[2*i+1]) for i in range(pairs)]
    between = [abs(lam[2*i+1] - lam[2*i+2]) for i in range(pairs-1)]
    return float(max(within)/max(min(between), 1e-300))


def select_beta_argmin(rows, key, band=2., sanity=None, higher_is_better=False):
    """LABEL-FREE argmin rule: among factors whose spectral gates passed for EVERY data seed,
    the factor minimizing the worst case over seeds of `key`; `band` lists the factors within
    band x the minimum (reported as equivalent).  sanity: warn if even the best exceeds it."""
    by = {}
    for r in rows:
        by.setdefault(r["factor"], []).append(r)
    worst = {}
    for f, rs in sorted(by.items()):
        vals = [r.get(key) for r in rs]
        if all(r.get("ok") for r in rs) and all(v is not None and np.isfinite(v) for v in vals):
            worst[f] = (min(vals) if higher_is_better else max(vals))
    if not worst:
        return dict(selected_factor=None, admissible=[], rule=f"argmin worst-seed {key}: no factor passed")
    sign = -1 if higher_is_better else 1
    best = min(worst, key=lambda f: sign*worst[f])
    b = worst[best]
    equivalent = [f for f, v in worst.items() if (v >= b/band if higher_is_better else v <= band*b)]
    out = dict(selected_factor=best, admissible=equivalent, equivalent_band=equivalent,
               worst_case_by_factor={str(f): v for f, v in worst.items()}, criterion=key,
               rule=(f"factor optimizing the worst case over data seeds of {key} among factors whose "
                     f"spectral gates passed for every seed; band = within {band}x of the optimum"))
    ordered = sorted(worst)
    if len(ordered) > 1 and best in (ordered[0], ordered[-1]):
        out["at_grid_edge"] = True
        out["edge_warning"] = (f"selected factor {best} is at the edge of the admissible grid "
                               f"[{ordered[0]}, {ordered[-1]}]: the optimum may lie outside; extend the grid")
    if sanity is not None and not higher_is_better and b > sanity:
        out["warning"] = f"best worst-case {key} = {b:.3g} > {sanity}: increase n before interpreting"
    return out


def knn_label_accuracy(code_train, label_train, code_test, label_test, k=5):
    """kNN classifier (majority vote) latent -> label, train -> test.  For MNIST only: a probe of
    class structure in the code, NOT a chart-recovery metric (MNIST has no known chart)."""
    code_train, code_test = np.asarray(code_train, np.float64), np.asarray(code_test, np.float64)
    mu, sd = code_train.mean(0), np.maximum(code_train.std(0), 1e-12)
    _, idx = cKDTree((code_train-mu)/sd).query((code_test-mu)/sd, k)
    votes = np.asarray(label_train).ravel()[idx.reshape(len(code_test), -1)].astype(int)
    pred = np.array([np.bincount(v).argmax() for v in votes])
    return float(np.mean(pred == np.asarray(label_test).ravel().astype(int)))


def affine_ambient(f, target, train, ridge=1e-6):
    """LABEL-FREE information of a spectral prefix about the DATA itself: affine readout of the
    standardized ambient target from the first m modes (train fit)."""
    return affine_fit(f, target, train, ridge)


def local_distortion(code, x, k=10):
    """Scale-free local isometry of a code: variance of log(|z_i-z_j| / |x_i-x_j|) over the k ambient
    nearest neighbours (0 = locally isometric up to one global scale).  Same quantity as the "iso"
    regularizer in vae.py, evaluated on held-out points."""
    x = np.asarray(x, np.float64)
    code = np.asarray(code, np.float64)
    d, j = cKDTree(x).query(x, min(k, len(x)-1)+1)
    d, j = d[:, 1:], j[:, 1:]
    dz = np.linalg.norm(code[j]-code[:, None, :], axis=-1)
    ok = d > 1e-8
    lr = np.log(dz[ok]+1e-12)-np.log(d[ok])
    return float(np.var(lr))
