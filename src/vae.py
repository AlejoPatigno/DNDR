"""Two affine posterior heads, nonlinear decoder; frozen spectral features supplied as data.

Model names used by experiment.py and the base architecture they map to:
  ae, vae, direct, plainvae, cvae      trained from scratch
  plainae                              deterministic AE with the same MLP encoder as plainvae (no DMAP)
  <kind>_warm                          continuation in m: previous-m weights, new input columns = 0
  vae_from_ae                          VAE initialized from the same-m AE (mean head + decoder)
  oracle_ae                            AE whose affine encoder IS the oracle readout (frozen); only the
                                       decoder is trained: what a learned decoder makes of the oracle chart
  oracle_ae_free                       same initialization, encoder trainable: does reconstruction keep
                                       the chart when it starts exactly on it?

Optional latent regularizer (cfg "reg", weight "reg_weight"), applied to the MEAN code mu(x):
  "iso"  local isometry up to scale: variance of log(|mu_i-mu_j| / |x_i-x_j|) over the reg_k
         ambient nearest neighbours of each point (neighbours searched inside the same split).
  "jac"  decoder isometry up to scale (relaxed distortion, Lee et al. 2021): Var/Mean^2 of
         |J_D(mu) v|^2 over the batch, v ~ N(0, I), J v by central finite differences.
Both are scale-free, so they do not fight the KL term.  The validation objective used for epoch
selection includes the same weighted term.
"""
import copy
import math
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

STOCHASTIC = ("vae", "plainvae", "cvae")


ORACLE_KINDS = ("oracle_ae", "oracle_ae_free")


def base_kind(name):
    name = name[:-5] if name.endswith("_warm") else name
    if name in ORACLE_KINDS:
        return "ae"
    return "vae" if name == "vae_from_ae" else name


def mlp(a, b, width):
    return nn.Sequential(nn.Linear(a, width), nn.SiLU(), nn.Linear(width, width),
                         nn.SiLU(), nn.Linear(width, b))


class VAE(nn.Module):
    def __init__(self, inputs, outputs, latent=2, width=128, kind="vae", condition=0):
        super().__init__()
        self.kind, self.latent, self.condition = kind, latent, condition
        plain = kind in ("plainvae", "plainae")
        self.trunk = mlp(inputs+condition, width, width) if plain else nn.Identity()
        h = width if plain else inputs+condition
        self.mu = nn.Linear(h, latent) if kind != "direct" else None
        self.logvar = nn.Linear(h, latent) if kind not in ("direct", "ae", "plainae") else None
        if self.logvar is not None:
            nn.init.zeros_(self.logvar.weight)
            nn.init.constant_(self.logvar.bias, -2.)
        self.decoder = mlp((inputs if kind == "direct" else latent)+condition, outputs, width)

    def encode(self, f, c):
        h = self.trunk(torch.cat([f, c], dim=1))
        mu = self.mu(h)
        return mu, self.logvar(h) if self.logvar is not None else torch.zeros_like(mu)

    def forward(self, f, c, sample=True):
        if self.kind == "direct":
            return self.decoder(torch.cat([f, c], 1)), f, torch.zeros_like(f)
        mu, lv = self.encode(f, c)
        z = mu
        if sample and self.kind not in ("ae", "plainae"):
            z = mu + torch.exp(.5*lv)*torch.randn_like(mu)
        return self.decoder(torch.cat([z, c], 1)), mu, lv


ARCH_KEYS = ("inputs", "outputs", "latent", "width", "kind", "condition")


def build(model_cfg):
    """Rebuild a model from a saved model_cfg (ignores training bookkeeping keys)."""
    return VAE(**{k: model_cfg[k] for k in ARCH_KEYS})


def expand_state(state, model, allow_missing_logvar=False):
    """Embed a smaller model's weights into `model`.  Input columns that did not exist
    (new modes psi_{m+1..m'}) are ZERO, so the expanded model computes exactly the same
    function on nested features: L_{m'} = (L_m ; 0).  Any other shape change is an error."""
    target = model.state_dict()
    out = {}
    for key, new in target.items():
        if key not in state:
            if key.startswith("logvar.") and allow_missing_logvar:
                out[key] = new   # AE -> VAE: keep the VAE's own log-variance initialization
                continue
            raise KeyError(f"Missing {key} in initialization state.")
        old = state[key].to(new.device, new.dtype)
        if old.shape == new.shape:
            out[key] = old.clone()
        elif old.ndim == 2 and old.shape[0] == new.shape[0] and old.shape[1] < new.shape[1]:
            pad = torch.zeros_like(new)
            pad[:, :old.shape[1]] = old
            out[key] = pad
        else:
            raise ValueError(f"Cannot expand {key}: {tuple(old.shape)} -> {tuple(new.shape)}")
    model.load_state_dict(out)
    return model


REGULARIZERS = (None, "iso", "jac")


def _neighbours(x, k):
    """kNN (excluding self) inside one split; zero distances (duplicates) are masked later."""
    from scipy.spatial import cKDTree
    k = min(k, len(x)-1)
    d, j = cKDTree(np.asarray(x, np.float64)).query(np.asarray(x, np.float64), k+1)
    return j[:, 1:], d[:, 1:]


def _reg_term(model, reg, fb, cb, ib, mu, table):
    if reg == "iso":
        J, D, F, C = table
        nb = J[ib]                                        # (B, k) indices inside the split
        mun, _ = model.encode(F[nb.reshape(-1)], C[nb.reshape(-1)])
        mun = mun.reshape(nb.shape[0], nb.shape[1], -1)
        dz = (mun - mu[:, None, :]).norm(dim=-1)
        dx = D[ib]
        ok = dx > 1e-8
        lr = torch.log(dz[ok] + 1e-8) - torch.log(dx[ok])
        return ((lr - lr.mean())**2).mean()
    if reg == "jac":
        v = torch.randn_like(mu)
        eps = 1e-2*mu.detach().std(0).mean().clamp_min(1e-6)
        dec = lambda z: model.decoder(torch.cat([z, cb], 1))
        jv = (dec(mu + eps*v) - dec(mu - eps*v))/(2*eps)
        n2 = (jv**2).sum(1)
        return n2.var()/(n2.mean()**2 + 1e-12)
    return torch.zeros((), device=fb.device)


def _loss_terms(model, fb, cb, xb, tau, kind):
    pred, mu, lv = model(fb, cb, sample=True)
    recon = ((pred-xb)**2).sum(1).mean()/(2*tau*tau)
    if kind in STOCHASTIC:
        kl = .5*(mu.square()+lv.exp()-1-lv).sum(1).mean()
        var = lv.exp().mean()
    else:
        kl = torch.zeros((), device=fb.device)
        var = torch.zeros((), device=fb.device)
    return recon, kl, var, mu


def fit_model(f, x, split, cfg, name, device, c=None, init_state=None, head_init=None, freeze_head=False):
    """Select the epoch by the FULL-weight validation objective (ELBO-type loss for stochastic
    models, reconstruction for deterministic ones).  Likelihood variance fixed (tau); the
    reconstruction term sums over output dimensions.  With init_state the initialization itself
    is a candidate (epoch -1), so continuation cannot worsen the selected validation objective.
    KL warm-up (cfg kl_warmup / kl_warmup_staged epochs) only changes the TRAINING weight.
    head_init=(W, b): initial mean head mu(f) = f W^T + b (e.g. the oracle readout); freeze_head keeps it
    fixed, so only the decoder (and the log-variance head, if any) is trained."""
    kind = base_kind(name)
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    c = np.zeros((len(f), 0), np.float32) if c is None else np.asarray(c, np.float32)
    f, x = np.asarray(f, np.float32), np.asarray(x, np.float32)
    model_cfg = dict(inputs=f.shape[1], outputs=x.shape[1], latent=cfg.get("latent", 2),
                     width=cfg.get("width_nn", 128), kind=kind, condition=c.shape[1])
    model = build(model_cfg).to(device)
    if init_state is not None:
        expand_state(init_state, model, allow_missing_logvar=name == "vae_from_ae")
    if head_init is not None:
        W, b = head_init
        with torch.no_grad():
            model.mu.weight.copy_(torch.as_tensor(W, dtype=torch.float32))
            model.mu.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    if freeze_head:
        model.mu.requires_grad_(False)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=cfg.get("lr", 1e-3))
    beta, tau = cfg.get("kl", 1e-3), cfg.get("tau", 1.)
    if tau <= 0 or beta < 0:
        raise ValueError("Require tau>0 and kl>=0.")
    warmup = cfg.get("kl_warmup_staged", 20) if name == "vae_from_ae" else cfg.get("kl_warmup", 0)
    reg, reg_w = cfg.get("reg"), float(cfg.get("reg_weight", 1.))
    if reg not in REGULARIZERS:
        raise ValueError(f"reg must be one of {REGULARIZERS}, not {reg!r}")
    if kind == "direct" or reg_w == 0:
        reg = None                       # no latent: nothing to regularize
    tables = {}
    if reg == "iso":
        for stage, ids in (("train", split[0]), ("validation", split[1])):
            j, d = _neighbours(x[ids], int(cfg.get("reg_k", 10)))
            tables[stage] = (torch.as_tensor(j, dtype=torch.long, device=device),
                             torch.as_tensor(d, dtype=torch.float32, device=device),
                             torch.from_numpy(f[ids]).to(device), torch.from_numpy(c[ids]).to(device))
    loaders = []
    batch = cfg.get("batch", 256)
    for i, ids in enumerate(split[:2]):
        ds = TensorDataset(*[torch.from_numpy(a[ids]) for a in (f, x, c)], torch.arange(len(ids)))
        loaders.append(DataLoader(ds, batch_size=batch, shuffle=i == 0,
                                  num_workers=0, pin_memory=device != "cpu"))
    per_epoch = math.ceil(len(split[0])/batch)
    epochs = max(cfg.get("epochs", 100), math.ceil(cfg.get("min_steps", 0)/per_epoch))

    def run_epoch(stage, kl_weight):
        loader = loaders[0] if stage == "train" else loaders[1]
        model.train(stage == "train")
        sums = dict(loss=0., recon=0., kl=0., posterior_var=0., reg=0.)
        count = 0
        # Fixed validation noise: checkpoint choice must not depend on fresh MC draws.
        rng_context = (torch.random.fork_rng(devices=[torch.cuda.current_device()] if device != "cpu" else [])
                       if stage != "train" else nullcontext())
        with rng_context:
            if stage != "train":
                torch.manual_seed(1729)
            for fb, xb, cb, ib in loader:
                fb, xb, cb, ib = [a.to(device) for a in (fb, xb, cb, ib)]
                with torch.set_grad_enabled(stage == "train"):
                    recon, kl, var, mu = _loss_terms(model, fb, cb, xb, tau, kind)
                    r = _reg_term(model, reg, fb, cb, ib, mu, tables.get(stage)) if reg else torch.zeros((), device=fb.device)
                    loss = recon + kl_weight*kl + reg_w*r
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite VAE loss; reduce learning rate. No silent clipping of affine logvariance.")
                    if stage == "train":
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 10.)
                        opt.step()
                n = len(xb)
                for k, v in (("loss", loss), ("recon", recon), ("kl", kl), ("posterior_var", var), ("reg", r)):
                    sums[k] = sums[k] + v.detach()*n   # accumulate on device: no extra host syncs
                count += n
        return {k: float(v)/count for k, v in sums.items()}

    history, best, state, best_epoch = [], float("inf"), None, None
    if init_state is not None:
        v = run_epoch("validation", beta)
        history.append({"epoch": -1, **{f"validation_{k}": val for k, val in v.items()}})
        best, state, best_epoch = v["loss"], copy.deepcopy(model.state_dict()), -1
    for epoch in range(epochs):
        weight = beta*min(1., (epoch+1)/warmup) if warmup else beta
        t = run_epoch("train", weight)
        v = run_epoch("validation", beta)
        history.append({"epoch": epoch, "train_kl_weight": weight,
                        **{f"train_{k}": val for k, val in t.items()},
                        **{f"validation_{k}": val for k, val in v.items()}})
        if v["loss"] < best:
            best, state, best_epoch = v["loss"], copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    model_cfg.update(selected_epoch=best_epoch, epochs=epochs, optimizer_steps=epochs*per_epoch,
                     kl_warmup=warmup, initialized=init_state is not None or head_init is not None, frozen_head=bool(freeze_head), reg=reg, reg_weight=reg_w if reg else 0.)
    return model, model_cfg, history


def _batches(f, c, device, size=512):
    c = np.zeros((len(f), 0), np.float32) if c is None else np.asarray(c, np.float32)
    for start in range(0, len(f), size):
        yield [torch.as_tensor(a[start:start+size], device=device, dtype=torch.float32) for a in (f, c)]


@torch.no_grad()
def predict(model, f, device, c=None, sample=False):
    """D(mu(x)) and mu(x).  NOTE: D(mu) != E_q[D(z)] for a nonlinear decoder; see predict_mc."""
    predictions, codes = [], []
    for ft, ct in _batches(f, c, device):
        y, mu, _ = model(ft, ct, sample)
        predictions.append(y.cpu().numpy())
        codes.append(mu.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(codes)


@torch.no_grad()
def predict_mc(model, f, device, c=None, samples=64, seed=2024):
    """Monte Carlo E_{q(z|x)}[D(z)] with a fixed generator (equals D(mu) for ae/direct)."""
    if model.kind in ("ae", "direct", "plainae"):
        return predict(model, f, device, c)[0]
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for ft, ct in _batches(f, c, device):
        mu, lv = model.encode(ft, ct)
        acc = 0.
        for _ in range(samples):
            z = mu + torch.exp(.5*lv)*torch.randn(mu.shape, generator=g, device=device)
            acc = acc + model.decoder(torch.cat([z, ct], 1))
        out.append((acc/samples).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def posterior_stats(model, f, device, c=None, active_threshold=1e-2):
    """KL to N(0,I), posterior variances and active units (Var_x E[z|x] > threshold)."""
    if model.kind == "direct":
        return {}
    mus, lvs = [], []
    for ft, ct in _batches(f, c, device):
        mu, lv = model.encode(ft, ct)
        mus.append(mu.cpu().numpy())
        lvs.append(lv.cpu().numpy())
    mu, lv = np.concatenate(mus), np.concatenate(lvs)
    out = {"code_variance": mu.var(0).tolist(),
           "active_units": int((mu.var(0) > active_threshold).sum())}
    if model.kind in STOCHASTIC:
        var = np.exp(lv)
        out.update(kl_mean=float((.5*(mu**2+var-1-lv)).sum(1).mean()),
                   posterior_var_mean=var.mean(0).tolist())
    return out


@torch.no_grad()
def generate(model, n, device, c=None, seed=123):
    if model.kind in ("ae", "direct", "plainae"):
        raise ValueError("No learned prior for deterministic controls.")
    generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(n, model.latent, generator=generator, device=device)
    if model.condition:
        if c is None or len(c) != n:
            raise ValueError("Conditional decoder requires one circular condition per generated sample.")
        z = torch.cat([z, torch.as_tensor(c, dtype=torch.float32, device=device)], 1)
    return model.decoder(z).cpu().numpy()
