"""Dataset construction and explicit, reproducible train/validation/test partitions."""
from pathlib import Path
import urllib.request
import numpy as np


def dihedral(x, ids):
    p = np.asarray(x)[..., np.asarray(ids), :]
    b0, b1, b2 = p[..., 0, :]-p[..., 1, :], p[..., 2, :]-p[..., 1, :], p[..., 3, :]-p[..., 2, :]
    b1 = b1 / np.maximum(np.linalg.norm(b1, axis=-1, keepdims=True), 1e-12)
    v = b0 - np.sum(b0*b1, axis=-1, keepdims=True)*b1
    w = b2 - np.sum(b2*b1, axis=-1, keepdims=True)*b1
    a = np.arctan2(np.sum(np.cross(b1, v)*w, axis=-1), np.sum(v*w, axis=-1))
    return np.where((np.linalg.norm(v, axis=-1) > 1e-10) & (np.linalg.norm(w, axis=-1) > 1e-10), a, np.nan)


def circular(a):
    return np.stack([np.cos(a[:, 0]), np.sin(a[:, 0]),
                     np.cos(a[:, 1]), np.sin(a[:, 1])], axis=1)


def angles(c):
    a = np.stack([np.arctan2(c[:, 1], c[:, 0]), np.arctan2(c[:, 3], c[:, 2])], axis=1)
    return np.where(np.linalg.norm(c.reshape(-1, 2, 2), axis=2) > 1e-10, a, np.nan)


def output_angles(x, kind, meta):
    if kind == "torus":
        return angles(x)
    xyz = x.reshape(-1, meta["atoms"], 3)
    return np.column_stack([dihedral(xyz, meta["phi_indices"]), dihedral(xyz, meta["psi_indices"])])


def angle_metrics(pred, target):
    err = np.degrees(np.abs(np.angle(np.exp(1j*(pred-target)))))
    valid = np.isfinite(err)
    return {"angular_mae_deg": [float(err[:, j][valid[:, j]].mean()) if valid[:, j].any() else None for j in range(2)],
            "defined_angle_fraction": valid.mean(0).tolist()}


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_KEYS = {"x_train", "y_train", "x_test", "y_test"}


def valid_mnist(path):
    try:
        with np.load(path, allow_pickle=False) as a:
            return MNIST_KEYS <= set(a.files) and a["x_train"].shape[1:] == (28, 28)
    except Exception:
        return False


def ensure_mnist(path):
    """Download ATOMICALLY: each process writes its own temporary file, validates it and renames it
    into place, so concurrent workers never read a half-written archive.  A corrupt file left by an
    earlier interrupted run is replaced."""
    import os
    path = Path(path)
    if valid_mnist(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        urllib.request.urlretrieve(MNIST_URL, tmp)
        if not valid_mnist(tmp):
            raise ValueError(f"downloaded file from {MNIST_URL} is not a valid MNIST npz")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def grid_chart(n, width=60., height=10.):
    """Perfect rectangular grid with ~n nodes and (almost) equal spacing in s and h:
    ny = round(sqrt(n*H/W)), nx = round(n/ny); nodes at cell centres, so no point sits on the border."""
    ny = max(2, int(round(np.sqrt(n*height/width))))
    nx = max(2, int(round(n/ny)))
    s = (np.arange(nx)+.5)*width/nx
    h = (np.arange(ny)+.5)*height/ny
    S, H = np.meshgrid(s, h, indexing="ij")
    return np.column_stack([S.ravel(), H.ravel()])


def square_grid_chart(n, width=60., height=10.):
    """Perfectly separated grid: the SAME spacing delta along s and h (square cells), delta = H/ny with
    ny = round(sqrt(n H/W)) and nx = W/delta nodes along s (W/H*ny must be an integer: 6*22 = 132 for
    the 60 x 10 sheet, N = 2904, delta = 0.4545).  Nodes at cell centres, delta/2 from every border."""
    ny = max(2, int(round(np.sqrt(n*height/width))))
    delta = height/ny
    nx = width/delta
    if abs(nx-round(nx)) > 1e-9:
        raise ValueError(f"W/H*ny = {nx:.6g} is not an integer: no square grid with ny = {ny}.")
    nx = int(round(nx))
    s = (np.arange(nx)+.5)*delta
    h = (np.arange(ny)+.5)*delta
    S, H = np.meshgrid(s, h, indexing="ij")
    return np.column_stack([S.ravel(), H.ravel()])


def swiss(n, width=60., height=10., seed=7, grid=False):
    """grid=False: uniform random sheet; True: near-uniform grid (grid_chart); "square": equal spacing
    in s and h (square_grid_chart).  Grids are deterministic; replicates differ only by the split."""
    from geometry import swiss_embed
    if grid == "square":
        z = square_grid_chart(n, width, height)
    elif grid:
        z = grid_chart(n, width, height)
    else:
        rng = np.random.default_rng(seed)
        z = rng.uniform(size=(n, 2))*[width, height]
    return swiss_embed(z), z


def swiss_volume(n, width=60., height=10., thickness=2., seed=7):
    """Swiss roll WITH VOLUME: uniform (s, h, t) in [0,W] x [0,H] x [-T/2, T/2], rolled by
    geometry.swiss_volume_embed.  Intrinsic dimension 3; chart = (s, h, t)."""
    from geometry import swiss_volume_embed
    z = np.random.default_rng(seed).uniform(size=(n, 3))*[width, height, thickness] - [0, 0, thickness/2]
    return swiss_volume_embed(z), z


def partitions(n, seed=7):
    perm = np.random.default_rng(seed).permutation(n)
    return np.split(perm, [int(.6*n), int(.8*n)])


def load_data(cfg):
    kind, n = cfg["dataset"], cfg.get("n", 3000)
    # Synthetic sets: data and partition seeds are explicit.  Default 7 reproduces earlier runs.
    data_seed = int(cfg.get("data_seed", 7))
    split_seed = int(cfg.get("split_seed", data_seed))
    meta = {"dataset": kind, "data_seed": data_seed, "split_seed": split_seed}
    if kind == "swiss":
        meta.update(width=float(cfg.get("width", 60)), height=float(cfg.get("height", 10)))
        meta["grid"] = cfg.get("grid", False) if cfg.get("grid") == "square" else bool(cfg.get("grid", False))
        x, z = swiss(n, meta["width"], meta["height"], seed=data_seed, grid=meta["grid"])
        f, split = x.copy(), partitions(len(x), split_seed)
    elif kind == "swissvol":
        meta.update(width=float(cfg.get("width", 60)), height=float(cfg.get("height", 10)),
                    thickness=float(cfg.get("thickness", 2.)))
        x, z = swiss_volume(n, meta["width"], meta["height"], meta["thickness"], seed=data_seed)
        f, split = x.copy(), partitions(len(x), split_seed)
    elif kind == "torus":
        z = np.random.default_rng(data_seed).uniform(-np.pi, np.pi, (n, 2))
        x = circular(z)*[1, 1, .6, .6]  # FLAT product torus T2 in R4.
        f, split = x.copy(), partitions(n, split_seed)
    elif kind == "mnist":
        path = ensure_mnist(cfg.get("data", "mnist.npz"))
        a = np.load(path, allow_pickle=False)
        # The subsample follows data_seed (default 7 = earlier runs), so resample_data gives
        # independent replicates (different training images) and not only new initializations.
        rng = np.random.default_rng(data_seed)
        ni, nt = min(n, len(a["x_train"])), min(max(100, n//5), len(a["x_test"]))
        ti = rng.choice(len(a["x_train"]), ni, replace=False)
        ei = rng.choice(len(a["x_test"]), nt, replace=False)
        x = np.concatenate([a["x_train"][ti], a["x_test"][ei]]).reshape(ni+nt, -1)/255.
        z = np.concatenate([a["y_train"][ti], a["y_test"][ei]])[:, None]
        split = [np.arange(int(.8*ni)), np.arange(int(.8*ni), ni), np.arange(ni, ni+nt)]
        f = x.copy()
        meta["source"] = str(path.resolve())
    elif kind == "alanine":
        path = Path(cfg["data"])
        a = np.load(path, allow_pickle=False)
        xyz, group = a["xyz"].astype(float), a["trajectory"]
        ph, ps, bonds = a["phi_indices"].astype(int), a["psi_indices"].astype(int), a["bonds"].astype(int)
        if xyz.ndim != 3 or xyz.shape[2] != 3 or len(group) != len(xyz):
            raise ValueError("Require xyz[N,A,3], trajectory[N], in consistent atom order.")
        if ph.shape != (4,) or ps.shape != (4,) or bonds.ndim != 2 or bonds.shape[1] != 2:
            raise ValueError("Require one phi/psi quadruplet and bonds[B,2].")
        if np.any(np.concatenate([ph, ps, bonds.ravel()]) < 0) or max(ph.max(), ps.max(), bonds.max()) >= xyz.shape[1]:
            raise ValueError("Atom index outside xyz.")
        groups = np.unique(group)
        if len(groups) < 3:
            raise ValueError("Require >=3 independent trajectories: training, validation, test.")
        # Keep temporal order. Downsampling is deterministic, never random frame splitting.
        roles = list(groups[:3])
        phase = 0.
        if cfg.get("trajectory_rotation", False):
            # Replicates for molecular data: data_seed picks which trajectory plays training /
            # validation / test (the 6 permutations) and the phase of the regular subsampling grid.
            # Splits stay whole trajectories, so there is no temporal leakage between them.
            import itertools
            roles = [roles[i] for i in list(itertools.permutations(range(3)))[data_seed % 6]]
            phase = float(np.random.default_rng(data_seed).uniform())
        selected = []
        for g in roles:
            ix = np.flatnonzero(group == g)
            k = min(n//3, len(ix))
            if phase and k < len(ix):
                step = len(ix)/k                      # > 1: floor(.) is strictly increasing
                pos = np.floor(phase*step + np.arange(k)*step).astype(int)
                selected.append(ix[np.minimum(pos, len(ix)-1)])
            else:
                selected.append(ix[np.linspace(0, len(ix)-1, k, dtype=int)])
        meta["trajectory_roles"] = [str(g) for g in roles]
        meta["subsample_phase"] = phase
        xyz = xyz[np.concatenate(selected)]
        split = np.split(np.arange(len(xyz)), np.cumsum([len(i) for i in selected])[:-1])
        z = np.column_stack([dihedral(xyz, ph), dihedral(xyz, ps)])
        # Fix an SO(3) frame using N, CA, C of the phi quadruplet. Preserve handedness.
        xyz = xyz-xyz[:, ph[2], None, :]
        e1 = xyz[:, ph[3]]
        e1 = e1/np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
        e2 = xyz[:, ph[1]]-np.sum(xyz[:, ph[1]]*e1, axis=1, keepdims=True)*e1
        norm = np.linalg.norm(e2, axis=1, keepdims=True)
        if np.min(norm) < 1e-8:
            raise ValueError("Degenerate molecular reference frame.")
        e2 /= norm
        xyz = np.einsum("nai,nij->naj", xyz, np.stack([e1, e2, np.cross(e1, e2)], axis=-1))
        f = np.linalg.norm(xyz[:, :, None]-xyz[:, None, :min(8, xyz.shape[1])], axis=-1).reshape(len(xyz), -1)
        x = xyz.reshape(len(xyz), -1)
        meta.update(phi_indices=ph.tolist(), psi_indices=ps.tolist(), bonds=bonds.tolist(),
                    atoms=xyz.shape[1], units="nm", source=str(path.resolve()),
                    trajectories=[str(g) for g in groups[:3]])
    else:
        raise ValueError(kind)
    if not np.isfinite(x).all() or not np.isfinite(f).all() or not np.isfinite(z).all():
        raise ValueError("Nonfinite input.")
    return x.astype(np.float32), f.astype(np.float32), z, split, meta


def molecular_metrics(x, reference, meta):
    """Empirical bond/chirality diagnostics; NOT an energy/physical-validity certificate."""
    xyz, ref = x.reshape(-1, meta["atoms"], 3), reference.reshape(-1, meta["atoms"], 3)
    bonds = np.asarray(meta["bonds"])
    lengths = lambda a: np.linalg.norm(a[:, bonds[:, 0]]-a[:, bonds[:, 1]], axis=-1)
    base, pred = lengths(ref), lengths(xyz)
    lo, hi = np.quantile(base, [.005, .995], axis=0)
    margin = .1*np.maximum(hi-lo, .005)
    outside = (pred < lo-margin) | (pred > hi+margin)
    result = {"bond_outlier_fraction": float(outside.mean()),
              "bond_rmse_to_train_mean_nm": float(np.sqrt(np.mean((pred-base.mean(0))**2)))}
    adjacent = [set() for _ in range(meta["atoms"])]
    for i, j in bonds:
        adjacent[i].add(j)
        adjacent[j].add(i)
    mismatches = []
    for i, neighbors in enumerate(adjacent):
        if len(neighbors) < 3:
            continue
        ids = sorted(neighbors)[:3]
        volume = lambda a: np.linalg.det(a[:, ids]-a[:, i, None])
        ref_vol, pred_vol = volume(ref), volume(xyz)
        # Only stable, nonplanar centers present in the reference qualify.
        sign = np.sign(np.median(ref_vol))
        if np.median(np.abs(ref_vol)) > 1e-5 and np.mean(np.sign(ref_vol) == sign) > .99:
            mismatches.append(np.mean(pred_vol*sign <= 0))
    result["chirality_sign_error"] = float(np.mean(mismatches)) if mismatches else None
    pairs = [(i, j) for i in range(meta["atoms"]) for j in range(i+1, meta["atoms"])
             if j not in adjacent[i] and not (adjacent[i] & adjacent[j])]
    if pairs:
        ids = np.asarray(pairs)
        distances = lambda a: np.linalg.norm(a[:, ids[:, 0]]-a[:, ids[:, 1]], axis=-1)
        floor = .7*np.quantile(distances(ref), .005, axis=0)
        result["nonbonded_close_pair_fraction"] = float(np.mean(distances(xyz) < floor))
    return result
