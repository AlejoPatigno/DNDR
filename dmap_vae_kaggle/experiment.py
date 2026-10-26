"""One seed per process/GPU; all mode budgets reuse one frozen dictionary.

Metric names (never mixed):
  mse_x              ambient reconstruction MSE per ambient coordinate, test, from D(mu(x))
  mc_mse_x           same with E_q[D(z)] (Monte Carlo), stochastic models
  validation_mse_x   ambient MSE on validation (use this, not test, to choose configurations)
  chart_mse_z        affine-oracle chart MSE per chart coordinate (DIFFERENT quantity)
  oracle_exact_mse_x ambient MSE of g(Pi_Omega z_hat), exact roll; bound (2/3) chart_mse_z
  *_chart_mse        latent-to-chart errors after rigid / similarity / affine alignment
  knn_chart_mse      NONLINEAR chart recovery (kNN latent->chart, train->test); knn_chart_ratio =
                     knn_chart_mse / knn_chart_floor (estimator on the true chart); unfolded =
                     ratio <= unfold_ratio.  Invariant to smooth reparametrizations of the latent.
  restart            index of the initialization restart (data and split unchanged)
  affine_x rows      label-free prefix information: affine readout of x from psi_1..psi_m (mse_x,
                     affine_validation_mse_x); MNIST adds knn_label_accuracy of the prefix.
  local_distortion   Var log(|z_i-z_j|/|x_i-x_j|) over ambient kNN of the test set (0 = locally
                     isometric up to scale); for oracle rows it is the affine chart readout.
  knn_label_accuracy MNIST only: kNN (latent -> digit) accuracy, a class-structure probe, not a chart.
Config: mode_order = 'spectral' | 'shuffled' (ablation, permutation in provenance.json);
        scratch_first_only = train scratch models only at the first budget (warm chain start);
        chart_decoder_epochs = longer budget for the true-chart reference decoder.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
import torch
from data import load_data, circular, angles, molecular_metrics, angle_metrics, output_angles
from geometry import (affine_fit, chart_target, chart_alignment, oracle_exact_swiss, torus_embed,
                      SwissSurface, base_beta, knn_chart_mse, knn_chart_floor, knn_label_accuracy,
                      affine_ambient, local_distortion)
from spectral import SpectralMap, backend_check
from vae import fit_model, predict, predict_mc, posterior_stats, generate, base_kind, STOCHASTIC, ORACLE_KINDS

ORDER = ["ae", "vae", "direct", "plainvae", "plainae", "vae_from_ae", "cvae", "oracle_ae", "oracle_ae_free"]
PLAIN = ("plainvae", "plainae")   # MLP encoder on x itself: no DMAP features (control)
INIT_STRIDE = 7919   # init seed of restart r = seed + INIT_STRIDE*r (r=0 reproduces earlier runs)
WARMABLE = ("ae", "vae", "direct")


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def clean(obj):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def resolve_beta(cfg, inputs_train):
    """beta (absolute, standardized units) > beta_factor (number, or path to beta_selection.json)."""
    b0 = base_beta(inputs_train)
    if cfg.get("beta") is not None:
        return float(cfg["beta"]), b0, None
    factor = cfg.get("beta_factor", 16.)
    if isinstance(factor, str):
        selection = json.loads(Path(factor).read_text())
        if selection.get("selected_factor") is None:
            raise ValueError(f"{factor}: no admissible beta factor; widen the sweep.")
        factor = selection["selected_factor"]
    return b0*float(factor), b0, float(factor)


def model_names(cfg):
    kinds = [k for k in ORDER if k in cfg.get("models", ["ae", "vae", "direct", "plainvae"])]
    unknown = set(cfg.get("models", [])) - set(ORDER)
    if unknown:
        raise ValueError(f"Unknown models {unknown}")
    if "vae_from_ae" in kinds and "ae" not in kinds:
        raise ValueError("vae_from_ae requires ae in the same run.")
    names = list(kinds)
    if "warm" in cfg.get("m_protocols", ["scratch"]):
        names += [f"{k}_warm" for k in kinds if k in WARMABLE]
    return names


def run(cfg):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not cfg.get("allow_cpu", False):
        raise RuntimeError("GPU not visible. CPU allowed only by explicit allow_cpu=true.")
    if cfg.get("resample_data", False):
        # Independent replicates: data sample AND partition change with the seed.
        cfg = dict(cfg, data_seed=cfg.get("data_seed_base", 1000)+cfg["seed"])
        cfg["split_seed"] = cfg["data_seed"]
    out = Path(cfg["out"])/f"seed_{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    if (out/"metrics.json").exists():
        raise FileExistsError(f"Completed run exists: {out}. Choose a new output directory.")
    x, raw_f, truth, split, meta = load_data(cfg)
    train, val, test = split
    dataset = cfg["dataset"]
    fm = raw_f[train].mean(0)
    fs = max(float(np.sqrt(np.mean((raw_f[train]-fm)**2))), 1e-12)
    inputs = (raw_f-fm)/fs
    xm = x[train].mean(0)
    xs = max(float(np.sqrt(np.mean((x[train]-xm)**2))), 1e-12)
    target = (x-xm)/xs
    anchors = np.arange(len(x)) if cfg.get("protocol", "inductive") == "transductive" else train
    beta, beta0, factor = resolve_beta(cfg, inputs[train].astype(np.float64))
    backend = cfg.get("backend", "flash")
    check = backend_check(backend, device)
    started = time.perf_counter()
    M = max(cfg["modes"])
    spectral = SpectralMap(M, beta, cfg.get("alpha", 1.), backend, device,
                           tolerance=cfg.get("spectral_tolerance", 2e-3)).fit(inputs[anchors])
    features = spectral.transform(inputs)
    features[anchors] = spectral.psi
    # Mode-order ablation: 'spectral' keeps psi_1, psi_2, ... ; 'shuffled' applies ONE fixed random
    # permutation of all M modes, so every budget m is still a nested prefix (warm continuation and
    # the nested-class argument stay valid) but the prefix no longer follows the spectral ordering.
    mode_order = cfg.get("mode_order", "spectral")
    if mode_order == "spectral":
        permutation = np.arange(features.shape[1])
    elif mode_order == "shuffled":
        permutation = np.random.default_rng(int(cfg.get("mode_order_seed", 0)) + int(cfg["seed"])).permutation(features.shape[1])
        features = features[:, permutation]
    else:
        raise ValueError(f"mode_order must be 'spectral' or 'shuffled', not {mode_order!r}")
    spectral.save(out/"encoder.npz")
    n_anchor, dim = len(anchors), inputs.shape[1]
    encoder_floats = lambda m: int(n_anchor*dim + n_anchor*m + 2*n_anchor + m + dim)  # anchors, psi, q, degree, lambda, center
    c0 = features[train].mean(0)
    cs = np.maximum(features[train].std(0), 1e-8)
    features = (features-c0)/cs  # train-only per-coordinate gauge; keeps the dictionary nested
    meta.update(protocol=cfg.get("protocol", "inductive"), n_train=len(train), n_val=len(val),
                n_test=len(test), spectral=spectral.diagnostics, kernel_check=check,
                beta=beta, base_beta=beta0, beta_factor=factor, kernel_length=fs/np.sqrt(beta),
                spectral_seconds=time.perf_counter()-started, torch=torch.__version__,
                python=platform.python_version(), device=device,
                gpu=torch.cuda.get_device_name() if device != "cpu" else None,
                encoder_stored_floats=encoder_floats(M),
                mode_order=mode_order, mode_permutation=permutation.tolist(),
                mode_permutation_head=[int(k)+1 for k in permutation[:16]],   # 1-based psi indices
                input_sha256=hashlib.sha256(x.tobytes()+raw_f.tobytes()).hexdigest())
    save_json(out/"provenance.json", clean({"config": cfg, "data": meta}))
    np.savez_compressed(out/"data.npz", x=x, truth=truth, train=train, validation=val, test=test)

    swiss = dataset == "swiss"
    chart = chart_target(dataset, truth) if dataset in ("swiss", "swissvol", "torus", "alanine") else None
    surface = SwissSurface(meta["width"], meta["height"]) if swiss else None
    metrics = []
    seed = cfg["seed"]

    def ambient(pred, ids):
        return float(np.mean((pred-x[ids])**2))

    if swiss:   # reference sample at the SAME N as generated/reconstructed sets
        metrics.append(dict(seed=seed, model="reference", m=None,
                            **surface.metrics(x[test], reference_chart=truth[train], prefix="test_")))

    # Chart decoder: learns x from the TRUE chart.  Separates decoder error from spectral error.
    chart_decoder = None
    if chart is not None and cfg.get("oracle_decoder", True):
        zc, zs = chart[train].mean(0), np.maximum(chart[train].std(0), 1e-12)
        # The reference decoder is cheap (one per seed) and converges slowly: optional longer budget.
        cd_train = dict(cfg, epochs=int(cfg.get("chart_decoder_epochs", cfg.get("epochs", 100))))
        chart_decoder, cd_cfg, cd_hist = fit_model((chart-zc)/zs, target, split, cd_train, "direct", device)
        pred = predict(chart_decoder, ((chart-zc)/zs)[test], device)[0]*xs+xm
        vpred = predict(chart_decoder, ((chart-zc)/zs)[val], device)[0]*xs+xm
        metrics.append(dict(seed=seed, model="chart_decoder", m=None, mse_x=ambient(pred, test),
                            validation_mse_x=ambient(vpred, val), selected_epoch=cd_cfg["selected_epoch"],
                            optimizer_steps=cd_cfg["optimizer_steps"]))
        save_json(out/"chart_decoder_history.json", clean(cd_hist))

    names = model_names(cfg)
    previous = {}      # warm-start chain: (name, restart) -> state_dict at the previous m
    ae_states = {}     # (m, restart) -> same-m AE state for vae_from_ae
    restarts = int(cfg.get("restarts", 1))
    if restarts < 1:
        raise ValueError("restarts must be >= 1")
    knn_k, unfold_ratio = int(cfg.get("knn_k", 5)), float(cfg.get("unfold_ratio", 10.))
    knn_floor = knn_chart_floor(chart[train], chart[test], knn_k) if chart is not None else None
    for m in cfg["modes"]:
        f = features[:, :m].astype(np.float32)
        head = None
        if chart is not None:
            oracle = affine_fit(f, chart, train, cfg.get("ridge", 1e-6))
            # Oracle readout as an encoder head (AE + oracle): same ridge solution as affine_fit, written as
            # mu(f) = f W^T + b in latent units = chart units / sigma, with sigma ONE isotropic scale (RMS of
            # the training chart), so the latent is the oracle chart up to a similarity (no anisotropic rescaling).
            fm_, ym_ = f[train].astype(np.float64).mean(0), chart[train].mean(0)
            a_, b_ = f[train].astype(np.float64)-fm_, chart[train]-ym_
            g_ = a_.T @ a_
            coef = np.linalg.solve(g_+cfg.get("ridge", 1e-6)*np.trace(g_)/len(g_)*np.eye(len(g_)), a_.T @ b_)
            sigma = float(np.sqrt(np.mean(b_**2)))
            head = (coef.T/sigma, (ym_-fm_@coef-ym_)/sigma)      # latent centred on the training chart mean
            err = lambda ids: float(np.mean(np.sum((oracle[ids]-chart[ids])**2, 1))/chart.shape[1])
            row = dict(seed=seed, model="oracle", m=m, chart_mse_z=err(test),
                       chart_mse_z_validation=err(val),
                       local_distortion=local_distortion(oracle[test], target[test], int(cfg.get("reg_k", 10))),
                       local_distortion_truth=local_distortion(chart[test], target[test], int(cfg.get("reg_k", 10))))
            if swiss:
                row.update(oracle_exact_swiss(oracle[test], truth[test], meta["width"], meta["height"]))
                row["constant_chart_mse"] = float(np.mean(np.sum((truth[train].mean(0)-truth[test])**2, 1))/2)
            elif dataset == "torus":
                row["oracle_exact_mse_x"] = ambient(torus_embed(oracle[test]), test)
            if dataset in ("torus", "alanine"):
                row.update(angle_metrics(angles(oracle[test]), truth[test]))
            metrics.append(row)
            np.savez_compressed(out/f"oracle_m{m}.npz", prediction=oracle[test], truth=chart[test])
            if chart_decoder is not None:
                pred = predict(chart_decoder, ((oracle-zc)/zs)[test], device)[0]*xs+xm
                metrics.append(dict(seed=seed, model="oracle_decoder", m=m, mse_x=ambient(pred, test)))
        # Label-free information of the prefix about the DATA (all datasets, incl. MNIST):
        # affine readout of x from psi_1..psi_m.  Its own column names, so it is never mistaken
        # for a trained model (no validation_mse_x).
        ax_pred = affine_ambient(f.astype(np.float64), target.astype(np.float64), train, cfg.get("ridge", 1e-6))*xs+xm
        row = dict(seed=seed, model="affine_x", m=m, mse_x=ambient(ax_pred[test], test),
                   affine_validation_mse_x=ambient(ax_pred[val], val))
        if dataset == "mnist":
            row["knn_label_accuracy"] = knn_label_accuracy(f[train], truth[train], f[test], truth[test], knn_k)
        metrics.append(row)
        for name in names:
            kind = base_kind(name)
            if kind in PLAIN and m != cfg["modes"][0]:
                continue  # conventional VAE does not depend on the mode budget
            if (cfg.get("scratch_first_only") and name in WARMABLE and m != cfg["modes"][0]
                    and "warm" in cfg.get("m_protocols", ["scratch"])):
                continue  # only the chain start is needed when the question is about continuation
            conditional = kind == "cvae"
            if conditional and dataset != "alanine":
                raise ValueError("cvae reserved for explicit molecular conditioning.")
            for r in range(restarts):
                # Restarts change ONLY the initialization/minibatch seed; data and split are fixed.
                cfg_r = dict(cfg, seed=cfg["seed"] + INIT_STRIDE*r)
                init = None
                if name.endswith("_warm"):
                    init = previous.get((name, r))
                    if init is None:      # chain starts from the scratch model at the first budget
                        init = previous[(name, r)] = previous[(name[:-5]+"@scratch", r)]
                        src = [q for q in metrics if q.get("model") == name[:-5] and q.get("m") == m
                               and q.get("restart", 0) == r][0]
                        metrics.append(dict(src, model=name, warm_start="chain start = scratch model"))
                        continue
                elif name == "vae_from_ae":
                    init = ae_states[(m, r)]
                fi = inputs if kind in PLAIN else f
                c = circular(truth).astype(np.float32) if conditional else None
                sub = lambda ids: None if c is None else c[ids]
                extra = {}
                if name in ORACLE_KINDS:
                    if head is None:
                        raise ValueError(f"{name} needs a chart (oracle readout); dataset {dataset} has none.")
                    extra = dict(head_init=head, freeze_head=name == "oracle_ae")
                model, model_cfg, history = fit_model(fi, target, split, cfg_r, name, device, c, init_state=init, **extra)
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if name in WARMABLE:
                    previous[(name+"@scratch", r)] = state
                if name.endswith("_warm"):
                    previous[(name, r)] = state
                if name == "ae":
                    ae_states[(m, r)] = state
                pred, code = predict(model, fi[test], device, sub(test))
                pred = pred*xs+xm
                vpred, _ = predict(model, fi[val], device, sub(val))
                row = dict(seed=seed, model=name, m=m, restart=r, init_seed=cfg_r["seed"],
                           mse_x=ambient(pred, test), validation_mse_x=ambient(vpred*xs+xm, val),
                           relative_rmse=float(np.linalg.norm(pred-x[test])/np.linalg.norm(x[test]-xm)),
                           neural_parameters=sum(p.numel() for p in model.parameters()),
                           encoder_stored_floats=0 if kind in PLAIN else encoder_floats(m),
                           selected_epoch=model_cfg["selected_epoch"], optimizer_steps=model_cfg["optimizer_steps"])
                packet = dict(prediction=pred, code=code, truth=truth[test])
                if kind in STOCHASTIC:
                    mc = predict_mc(model, fi[test], device, sub(test))*xs+xm
                    row["mc_mse_x"] = ambient(mc, test)
                    packet["prediction_mc"] = mc
                stats = posterior_stats(model, fi[test], device, sub(test))
                row.update({k: v for k, v in stats.items()})
                if chart is not None and kind != "direct":
                    _, train_code = predict(model, fi[train], device, sub(train))
                    packet["code_train"] = train_code
                    knn = knn_chart_mse(train_code, chart[train], code, chart[test], knn_k)
                    row["local_distortion"] = local_distortion(code, target[test], int(cfg.get("reg_k", 10)))
                    row.update(knn_chart_mse=knn, knn_chart_floor=knn_floor,
                               knn_chart_ratio=knn/max(knn_floor, 1e-300),
                               unfolded=bool(knn <= unfold_ratio*knn_floor))
                    if swiss:
                        row.update(chart_alignment(train_code, truth[train], code, truth[test]))
                if dataset == "mnist" and kind != "direct":
                    _, train_code = predict(model, fi[train], device, sub(train))
                    packet["code_train"] = train_code
                    row["knn_label_accuracy"] = knn_label_accuracy(train_code, truth[train], code, truth[test], knn_k)
                if dataset in ("torus", "alanine"):
                    row.update(angle_metrics(output_angles(pred, dataset, meta), truth[test]))
                if dataset == "alanine":
                    row.update(molecular_metrics(pred, x[train], meta))
                if swiss:
                    row.update(surface.metrics(pred, reference_chart=truth[train], prefix="recon_"))
                    if "prediction_mc" in packet:
                        row.update(surface.metrics(packet["prediction_mc"], reference_chart=truth[train], prefix="mcrecon_"))
                if kind in STOCHASTIC:
                    cond = c[test] if conditional else None
                    samples = generate(model, len(test), device, cond)*xs+xm   # same N as the reference
                    packet["prior_samples"] = samples
                    if conditional:
                        packet["conditions"] = cond
                        row["conditional_prior"] = angle_metrics(output_angles(samples, "alanine", meta), angles(cond))
                    if dataset == "alanine":
                        row["generation_diagnostics"] = molecular_metrics(samples, x[train], meta)
                    if swiss:
                        row.update(surface.metrics(samples, reference_chart=truth[train], prefix="prior_"))
                stem = f"{name}_m{m}" + (f"_r{r}" if r else "")
                np.savez_compressed(out/f"{stem}.npz", **packet)
                torch.save(dict(state=model.state_dict(), model_cfg=model_cfg, cfg=cfg_r, meta=meta,
                                feature_center=c0[:m], feature_scale=cs[:m], input_center=fm,
                                input_scale=fs, target_center=xm, target_scale=xs, m=m), out/f"{stem}.pt")
                save_json(out/f"{stem}_history.json", clean(history))
                metrics.append(row)
                save_json(out/"metrics.partial.json", clean(metrics))
                extra = f", knn/piso {row['knn_chart_ratio']:.3g}" if "knn_chart_ratio" in row else ""
                print(f"{dataset} seed={seed} {stem}: test mse_x={row['mse_x']:.5g} "
                      f"(validation {row['validation_mse_x']:.5g}, epoch {row['selected_epoch']}{extra})", flush=True)
                del model
    save_json(out/"metrics.json", clean(metrics))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cfg["seed"] = a.seed
    run(cfg)
