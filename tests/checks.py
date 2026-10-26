"""Small correctness gates, not scientific benchmark results."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from data import swiss, partitions, circular, angles, dihedral
from geometry import (swiss_embed, oracle_exact_swiss, chart_alignment, graph_shortcut_mass,
                      select_beta, base_beta, knn_chart_mse, knn_chart_floor,
                      degeneracy_defect, select_beta_argmin, knn_label_accuracy, local_distortion)
from spectral import SpectralMap, choose_beta, backend_check
from vae import (fit_model, predict, predict_mc, posterior_stats, expand_state, build, VAE)


def geometry_checks():
    rng = np.random.default_rng(3)
    z = rng.uniform(size=(500, 2))*[60, 10]
    zh = z + rng.normal(scale=.2, size=z.shape)
    r = oracle_exact_swiss(zh, z, 60, 10)
    assert r["oracle_exact_mse_x"] <= r["oracle_exact_bound_x"]*r["oracle_lipschitz_ratio"]**2 + 1e-12
    assert r["oracle_lipschitz_ratio"] < 1 + 1e-3
    code = (z - z.mean(0)) @ np.array([[0., -1.], [1., 0.]]) * .05
    a = chart_alignment(code[:400], z[:400], code[400:], z[400:])
    assert a["similarity_chart_mse"] < 1e-10 and a["affine_chart_mse"] < 1e-10 and a["rigid_chart_mse"] > 1
    x, _ = swiss(1200)
    u = (x - x.mean(0))/np.sqrt(np.mean((x - x.mean(0))**2))
    b0 = base_beta(u)
    np.testing.assert_allclose(b0, choose_beta(u))
    g = graph_shortcut_mass(u, [8*b0, 64*b0])["graph_shortcut_mass"]
    assert g[0] > 10*g[1], g
    sel = select_beta([dict(factor=8, ok=True, graph_shortcut_mass=g[0]),
                       dict(factor=64, ok=True, graph_shortcut_mass=g[1])], threshold=max(g[1], 1e-3))
    assert sel["selected_factor"] == 64
    return {"geometry": "passed", "graph_shortcut_mass_f8_f64": g}


def knn_checks():
    """Unfolded latent (smooth reparametrization) ~ floor; folded latent >> floor."""
    x, z = swiss(1500)
    tr, _, te = partitions(len(z))
    floor = knn_chart_floor(z[tr], z[te])
    smooth = np.column_stack([np.tanh(z[:, 0]/30), z[:, 1]**2])
    folded = np.column_stack([np.abs(z[:, 0]-30), z[:, 1]])
    a = knn_chart_mse(smooth[tr], z[tr], smooth[te], z[te])
    b = knn_chart_mse(folded[tr], z[tr], folded[te], z[te])
    assert a < 10*floor < b, (floor, a, b)
    return {"knn_chart": "passed", "knn_floor": floor, "knn_smooth_ratio": a/floor, "knn_folded_ratio": b/floor}


def sign_checks(backend, device):
    """Same operator, different number of requested modes -> IDENTICAL leading features
    (ARPACK signs are arbitrary; spectral.fix_signs makes them deterministic)."""
    x, _ = swiss(400)
    u = (x-x.mean(0))/x.std()
    beta = choose_beta(u)*16
    a = SpectralMap(5, beta, backend=backend, device=device).fit(u)
    b = SpectralMap(12, beta, backend=backend, device=device).fit(u)
    overlap = np.sum(a.psi[:, :4]*b.psi[:, :4]*a.pi[:, None], 0)   # L2(pi) inner products
    assert np.all(overlap > 0.999), overlap
    return {"eigenvector_sign_convention": "passed", "sign_overlap_min": float(overlap.min())}


def protocol_checks():
    """v5 additions: torus degeneracy rule, argmin rule, MNIST label probe, alanine replicates."""
    lam = 1 - .01*np.array([1, 1, 2.78, 2.78, 4, 4, 3.78, 3.78])      # exact (cos, sin) pairs
    assert degeneracy_defect(lam) < 1e-9
    assert degeneracy_defect(lam + np.array([0, 1e-3, 0, 0, 0, 0, 0, 0])) > .05
    rows = [dict(factor=f, data_seed=s, ok=True, d=v*(1+.1*s)) for f, v in ((8, 1.), (16, .2), (32, .3)) for s in (0, 1)]
    sel = select_beta_argmin(rows, "d", band=2.)
    assert sel["selected_factor"] == 16 and sel["equivalent_band"] == [16, 32], sel
    rows.append(dict(factor=16, data_seed=2, ok=False, d=.1))
    assert select_beta_argmin(rows, "d")["selected_factor"] == 32
    rng = np.random.default_rng(0)
    lab = rng.integers(3, size=400)
    code = lab[:, None] + .05*rng.normal(size=(400, 2))
    assert knn_label_accuracy(code[:300], lab[:300], code[300:], lab[300:]) == 1.
    # Alanine replicates: trajectory roles rotate with data_seed; splits stay whole trajectories.
    from data import load_data
    n_atoms, frames = 10, 90
    xyz = rng.normal(size=(3*frames, n_atoms, 3))
    group = np.repeat([0, 1, 2], frames)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/"ala.npz"
        np.savez(path, xyz=xyz, trajectory=group, phi_indices=[0, 1, 2, 3], psi_indices=[1, 2, 3, 4],
                 bonds=np.array([[i, i+1] for i in range(n_atoms-1)]))
        roles = set()
        for seed in range(6):
            _, _, _, split, meta = load_data(dict(dataset="alanine", data=str(path), n=60, data_seed=seed,
                                                  trajectory_rotation=True))
            roles.add(tuple(meta["trajectory_roles"]))
            assert [len(s) for s in split] == [20, 20, 20]
        assert len(roles) == 6
        _, _, _, _, meta0 = load_data(dict(dataset="alanine", data=str(path), n=60))
        assert meta0["trajectory_roles"] == ["0", "1", "2"] and meta0["subsample_phase"] == 0.
    # perfect-grid Swiss roll: regular spacing, whole grid split into train/validation/test
    xg, _, zg, sg, mg = load_data(dict(dataset="swiss", n=600, grid=True, data_seed=1001))
    assert mg["grid"] and len(xg) == sum(len(a) for a in sg) and len(np.unique(np.round(np.diff(np.unique(zg[:, 0])), 9))) == 1
    # perfectly separated (square) grid: one spacing along s AND h
    xq, _, zq, sq, mq = load_data(dict(dataset="swiss", n=600, grid="square", data_seed=1001))
    ds_, dh_ = [np.unique(np.round(np.diff(np.unique(zq[:, k])), 9)) for k in (0, 1)]
    assert mq["grid"] == "square" and len(ds_) == 1 and len(dh_) == 1 and abs(ds_[0]-dh_[0]) < 1e-9
    assert len(xq) == sum(len(a) for a in sq)
    # Swiss roll with volume: centre surface = the sheet's roll, offset distance = |t|, turns do not touch
    from geometry import swiss_volume_embed
    xv, _, zv, sv, mv = load_data(dict(dataset="swissvol", n=4000, thickness=2., data_seed=1001))
    zc = zv.copy(); zc[:, 2] = 0
    assert np.abs(swiss_volume_embed(zc)-swiss_embed(zc[:, :2])).max() < 1e-9
    assert np.abs(np.linalg.norm(xv-swiss_volume_embed(zc), axis=1)-np.abs(zv[:, 2])).max() < 1e-5
    from scipy.spatial import cKDTree
    pr = cKDTree(xv).query_pairs(1.2, output_type="ndarray")
    assert (np.abs(zv[pr[:, 0], 0]-zv[pr[:, 1], 0]) > 5).sum() == 0, "turns of the thick roll touch"
    return {"protocol_v5": "passed"}


def distortion_checks():
    """local_distortion: 0 for an isometry up to scale, large for a folded code."""
    x, z = swiss(1500)
    iso = local_distortion(3.*z, x)                  # the true chart, rescaled
    fold = local_distortion(np.column_stack([np.cos(z[:, 0]), z[:, 1]]), x)
    assert iso < .05 and fold > 10*iso, (iso, fold)
    return {"local_distortion": "passed", "distortion_true_chart": iso, "distortion_folded": fold}


def reg_checks(device):
    """Latent regularizers train, are finite and are recorded; `direct` ignores them."""
    x, _ = swiss(300)
    split = partitions(len(x))
    u = ((x-x[split[0]].mean(0))/x[split[0]].std()).astype(np.float32)
    f = np.column_stack([u, np.random.default_rng(0).normal(size=(len(u), 5))]).astype(np.float32)
    for reg in ("iso", "jac"):
        cfg = {"seed": 3, "epochs": 3, "width_nn": 16, "reg": reg, "reg_weight": 1.0, "reg_k": 5, "batch": 64}
        for name in ("ae", "vae", "vae_warm", "plainae", "plainvae"):
            model, mc, hist = fit_model(f, u, split, cfg, name, device)
            h = hist[-1]
            assert mc["reg"] == reg and np.isfinite(h["train_reg"]) and h["train_reg"] > 0, (reg, name, h)
            assert np.isfinite(h["validation_loss"]) and h["validation_reg"] > 0
    _, mc, _ = fit_model(f, u, split, dict(cfg, reg="iso"), "direct", device)
    assert mc["reg"] is None
    return {"latent_regularizers": "passed"}


def oracle_head_checks(device):
    """AE + oracle: the frozen head reproduces the given affine readout exactly and does not move;
    the free head starts on it and is trained."""
    x, z = swiss(300)
    split = partitions(len(x))
    u = ((x-x[split[0]].mean(0))/x[split[0]].std()).astype(np.float32)
    f = np.column_stack([u, np.random.default_rng(0).normal(size=(len(u), 5))]).astype(np.float32)
    rng = np.random.default_rng(1)
    W, b = rng.normal(size=(2, f.shape[1])), rng.normal(size=2)
    cfg = {"seed": 3, "epochs": 3, "width_nn": 16, "batch": 64}
    frozen, mc, _ = fit_model(f, u, split, cfg, "oracle_ae", device, head_init=(W, b), freeze_head=True)
    _, code = predict(frozen, f, device)
    np.testing.assert_allclose(code, f @ W.T + b, atol=1e-4)
    assert mc["frozen_head"] and mc["initialized"] and mc["kind"] == "ae"
    free, mc2, _ = fit_model(f, u, split, cfg, "oracle_ae_free", device, head_init=(W, b))
    moved = float(np.abs(free.mu.weight.detach().cpu().numpy()-W).max())
    assert not mc2["frozen_head"] and moved > 0
    return {"oracle_head": "passed", "oracle_free_head_moved": moved}


def continuation_checks(device):
    """L_{m+1} = (L_m; 0): the expanded model computes the same function on nested features."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    f = rng.normal(size=(64, 12)).astype(np.float32)
    out = {}
    for kind in ("ae", "vae", "direct"):
        small = VAE(8, 3, kind=kind, width=16).to(device).eval()
        big = expand_state(small.state_dict(), VAE(12, 3, kind=kind, width=16).to(device)).eval()
        a = predict(small, f[:, :8], device)[0]
        b = predict(big, f, device)[0]
        np.testing.assert_allclose(a, b, atol=1e-5)
        out[kind] = float(np.abs(a-b).max())
    ae = VAE(8, 3, kind="ae", width=16).to(device).eval()
    vae = expand_state(ae.state_dict(), VAE(8, 3, kind="vae", width=16).to(device),
                       allow_missing_logvar=True).eval()
    np.testing.assert_allclose(predict(ae, f[:, :8], device)[0], predict(vae, f[:, :8], device)[0], atol=1e-5)
    assert float(vae.logvar.bias.mean()) == -2.
    return {"continuation_max_abs_diff": out, "ae_to_vae_mean_path": "passed"}


def check(backend="torch", device="cpu"):
    torch.set_num_threads(2)
    result = backend_check(backend, device)
    result.update(geometry_checks())
    result.update(protocol_checks())
    result.update(distortion_checks())
    result.update(reg_checks(device))
    result.update(knn_checks())
    result.update(sign_checks(backend, device))
    result.update(continuation_checks(device))
    result.update(oracle_head_checks(device))
    x, truth = swiss(160)
    split = partitions(len(x))
    assert not set(split[0]) & set(split[1]) and not set(split[0]) & set(split[2])
    u = (x-x[split[0]].mean(0))/x[split[0]].std()
    dm = SpectralMap(8, choose_beta(u[split[0]])*8, backend=backend, device=device).fit(u[split[0]])
    gram = dm.psi.T @ (dm.pi[:, None]*dm.psi)
    np.testing.assert_allclose(gram, np.eye(8), atol=1e-5)
    result.update(dm.diagnostics)
    f = dm.transform(u).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        dm.save(Path(tmp)/"encoder.npz")
        loaded = SpectralMap.load(Path(tmp)/"encoder.npz", device)
        np.testing.assert_allclose(loaded.transform(u[split[2]]), f[split[2]], atol=3e-3)
    torch.manual_seed(7)
    cfg = {"seed": 7, "epochs": 2, "width_nn": 16, "min_steps": 5}
    model, model_cfg, history = fit_model(f, u.astype(np.float32), split, cfg, "vae", device)
    assert not any(p.requires_grad for p in model.parameters())
    assert model_cfg["optimizer_steps"] >= 5 and {"train_recon", "train_kl", "validation_loss"} <= set(history[-1])
    y, z = predict(model, f[split[2]], device)
    assert y.shape == (len(split[2]), 3) and z.shape[1] == 2 and np.isfinite(y).all()
    mc = predict_mc(model, f[split[2]], device, samples=8)
    assert mc.shape == y.shape and np.isfinite(mc).all()
    stats = posterior_stats(model, f[split[2]], device)
    assert np.isfinite(stats["kl_mean"]) and len(stats["posterior_var_mean"]) == 2
    rebuilt = build(model_cfg).to(device)
    rebuilt.load_state_dict(model.state_dict())
    # Warm start: the initialization is a selection candidate, so validation cannot get worse.
    warm, warm_cfg, warm_hist = fit_model(f, u.astype(np.float32), split, cfg, "vae_warm", device,
                                          init_state=model.state_dict())
    assert warm_hist[0]["epoch"] == -1
    assert min(h["validation_loss"] for h in warm_hist) <= warm_hist[0]["validation_loss"]
    staged, _, staged_hist = fit_model(f, u.astype(np.float32), split, dict(cfg, kl_warmup_staged=2), "vae_from_ae",
                                       device, init_state=fit_model(f, u.astype(np.float32), split, cfg, "ae", device)[0].state_dict())
    assert staged.kind == "vae" and staged_hist[1]["train_kl_weight"] < staged_hist[-1]["train_kl_weight"]
    a = np.array([[3.14, -3.14], [.5, -.7]])
    np.testing.assert_allclose(angles(circular(a)), a, atol=1e-12)
    # Known signed dihedral, then invariance under a proper rotation and translation.
    pts = np.array([[[0., 1., 0.], [0., 0., 0.], [1., 0., 0.], [1., 0., 1.]]])
    np.testing.assert_allclose(dihedral(pts, [0, 1, 2, 3]), np.pi/2)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    np.testing.assert_allclose(dihedral(pts@rotation+5, [0, 1, 2, 3]), np.pi/2)
    result["vae_freeze_shapes_mc_posterior"] = "passed"
    result["warm_start_and_staged_vae"] = "passed"
    result["dihedral_orientation"] = "passed"
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["torch", "flash"], default="torch")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="checks.json")
    a = p.parse_args()
    result = check(a.backend, a.device)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
