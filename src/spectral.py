"""Frozen Gaussian DMAP. Diagonal retained; right modes normalized in L2(pi).

flash: unmodified upstream WMMA source, compiled for the actual device (T4=7.5).
torch: explicit tiled reference/alternative, NEVER reported as FlashDiffusion.
CPU ARPACK controls the iteration; kernel applications execute on the selected GPU.
"""
from functools import lru_cache
from pathlib import Path
import os
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

FLASH_COMMIT = "f5c2d1abb7e418829651f7e2fedbb8b323fd56f9"


@lru_cache(None)
def flash_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("FlashDiffusion requires CUDA; use backend='torch' explicitly for CPU tests.")
    from torch.utils.cpp_extension import load
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    os.environ.setdefault("MAX_JOBS", "2")
    here = Path(__file__).resolve().parent
    candidates = [Path(os.environ["FLASH_DIFFUSION_CU"])] if os.environ.get("FLASH_DIFFUSION_CU") else []
    candidates += [here / "vendor" / "flash_diffusion.cu", here / "flash_diffusion.cu"]
    source = next((c for c in candidates if c.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            "flash_diffusion.cu not found in: " + ", ".join(map(str, candidates)) +
            ". Upload vendor/ (or the full ZIP) to the Kaggle Dataset and rerun the setup cell.")
    return load(name=f"fd_vae_sm{major}{minor}",
                sources=[str(source)],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"])


class Gaussian:
    def __init__(self, x, beta, backend="torch", device="cpu", tile=512):
        self.backend, self.device, self.beta, self.tile = backend, device, beta, tile
        self.dtype = torch.float64 if device == "cpu" else torch.float32
        self.x = torch.as_tensor(np.asarray(x), dtype=self.dtype, device=device)
        self.ext = None
        if backend == "flash":
            self.ext = flash_extension()
            self.x = self.x.half().contiguous()  # upstream coordinate precision
            self.sq = self.ext.compute_sq_norms(self.x)
        elif backend != "torch":
            raise ValueError(backend)

    @torch.no_grad()
    def apply(self, v, query=None):
        v = torch.as_tensor(np.asarray(v), device=self.device, dtype=self.dtype)
        if v.ndim == 1:
            v = v[:, None]
        if self.ext is not None and query is None:
            return torch.stack([self.ext.prepass(self.x, col.contiguous().float(),
                                self.sq, self.beta) for col in v.T], 1).cpu().double().numpy()
        # Nyström uses the SAME quantized anchors and query precision as the backend.
        q = self.x if query is None else torch.as_tensor(query, device=self.device,
                                                       dtype=self.x.dtype)
        x, q = self.x.to(self.dtype), q.to(self.dtype)
        out = torch.zeros((len(q), v.shape[1]), device=self.device, dtype=self.dtype)
        for i in range(0, len(q), self.tile):
            qi = q[i:i+self.tile]
            for j in range(0, len(x), self.tile):
                xj = x[j:j+self.tile]
                d2 = (qi.square().sum(1)[:, None] + xj.square().sum(1)[None, :]
                      - 2 * qi @ xj.T).clamp_min_(0)
                out[i:i+len(qi)] += torch.exp(-self.beta*d2) @ v[j:j+len(xj)]
        return out.cpu().double().numpy()

    def symmetric(self, v, scale):
        if self.ext is not None:
            vt = torch.as_tensor(v, device=self.device, dtype=torch.float32).contiguous()
            st = torch.as_tensor(scale, device=self.device, dtype=torch.float32).contiguous()
            return self.ext.matvec(self.x, vt, st, self.sq, self.beta).cpu().double().numpy()
        return scale * self.apply(scale * v).ravel()


def choose_beta(x, seed=7):
    """Training-only median heuristic. Not the paper's beta-flow calibration."""
    rng = np.random.default_rng(seed)
    i = rng.integers(len(x), size=4096)
    j = rng.integers(len(x), size=4096)
    d2 = np.sum((x[i] - x[j])**2, axis=1)
    positive = d2[d2 > 0]
    if not len(positive):
        raise ValueError("Constant training features.")
    return float(1 / np.median(positive))


def fix_signs(vecs):
    """Deterministic sign convention: the entry of largest magnitude of every eigenvector is
    positive.  ARPACK returns each eigenvector up to an arbitrary sign that depends on how many
    modes are requested, so without this the SAME operator gives different features when only
    max(modes) changes.  (Rotations inside exactly degenerate eigenspaces are NOT fixed.)"""
    vecs = np.array(vecs, copy=True)
    idx = np.argmax(np.abs(vecs), axis=0)
    signs = np.sign(vecs[idx, np.arange(vecs.shape[1])])
    signs[signs == 0] = 1
    return vecs * signs


class SpectralMap:
    def __init__(self, m=32, beta=None, alpha=1., backend="torch", device="cpu",
                 tolerance=2e-3, tile=512):
        self.m, self.beta, self.alpha = m, beta, alpha
        self.backend, self.device, self.tolerance, self.tile = backend, device, tolerance, tile

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.m + 2 >= len(x):
            raise ValueError("Need more anchors than modes + 2.")
        self.center = x.mean(0)
        x = x - self.center
        self.beta = choose_beta(x) if self.beta is None else float(self.beta)
        if self.beta <= 0:
            raise ValueError("beta must be positive.")
        self.kernel = Gaussian(x, self.beta, self.backend, self.device, self.tile)
        self.q = self.kernel.apply(np.ones(len(x))).ravel()
        self.w = self.q**(-self.alpha)
        self.degree = self.w * self.kernel.apply(self.w).ravel()
        self.pi = self.degree / self.degree.sum()
        scale = self.w / np.sqrt(self.degree)
        op = LinearOperator((len(x), len(x)), dtype=np.float64,
                            matvec=lambda v: self.kernel.symmetric(v, scale))
        vals, vecs = eigsh(op, k=self.m+1, which="LA", tol=1e-6,
                           maxiter=10000, v0=np.random.default_rng(7).normal(size=len(x)))
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        vecs = fix_signs(vecs)
        residual = np.linalg.norm(np.column_stack([op @ v for v in vecs.T]) - vecs*vals, axis=0)
        if max(residual) > self.tolerance or abs(vals[0]-1) > self.tolerance:
            raise RuntimeError(f"Spectral residual/Perron gate failed: {residual.max()}, {vals[0]}")
        if 1-vals[1] < 5*max(residual.max(), np.finfo(np.float32).eps):
            raise RuntimeError("Perron gap unresolved. Reduce beta or change precision/backend.")
        self.values = vals[1:]
        if self.values.min() <= 1e-7:
            raise RuntimeError("Retained eigenvalue too small for Nyström; reduce mode budget.")
        self.psi = vecs[:, 1:] / np.sqrt(self.pi)[:, None]
        self.diagnostics = dict(sign_convention="max_abs_positive",
                                max_residual=float(residual.max()), perron_gap=float(1-vals[1]),
                                min_eigenvalue=float(self.values.min()), beta=self.beta,
                                alpha=self.alpha, backend=self.backend, flash_commit=FLASH_COMMIT)
        count = min(64, len(x))
        observed = self.transform(x[:count] + self.center)
        error = np.linalg.norm(observed-self.psi[:count])/max(np.linalg.norm(self.psi[:count]), 1e-15)
        self.diagnostics["nystrom_training_relative_error"] = float(error)
        if error > max(.01, 5*self.tolerance):
            raise RuntimeError(f"Nyström consistency gate failed: {error}; lower m or use torch backend.")
        return self

    def transform(self, x):
        # P(x,j) = K(x,j) q_j^-alpha / sum_l K(x,l) q_l^-alpha.
        query = np.asarray(x, dtype=np.float64) - self.center
        rhs = np.column_stack([self.w, self.w[:, None]*self.psi])
        sums = self.kernel.apply(rhs, query=query)
        if np.any(sums[:, 0] <= 1e-25):
            raise RuntimeError("Nyström query outside resolved support; kernel underflow.")
        return sums[:, 1:] / sums[:, :1] / self.values

    def save(self, path):
        np.savez_compressed(path, anchors=self.kernel.x.cpu().numpy(), center=self.center,
                            q=self.q, degree=self.degree, psi=self.psi, values=self.values,
                            beta=self.beta, alpha=self.alpha, backend=self.backend)

    @classmethod
    def load(cls, path, device="cpu"):
        a = np.load(path, allow_pickle=False)
        model = cls(len(a["values"]), float(a["beta"]), float(a["alpha"]),
                    str(a["backend"]), device)
        model.center, model.q, model.degree = a["center"], a["q"], a["degree"]
        model.psi, model.values = a["psi"], a["values"]
        model.w, model.pi = model.q**(-model.alpha), model.degree/model.degree.sum()
        # Out-of-sample evaluation needs no compiled extension.
        model.kernel = Gaussian(a["anchors"], model.beta, "torch", device)
        if model.backend == "flash":
            model.kernel.x = model.kernel.x.half()
        return model


def backend_check(backend="torch", device="cpu"):
    """Non-multiple tile sizes and feature widths; compare the ACTUAL quantized operator."""
    errors = []
    for dim in (3, 80, 784):
        x = np.random.default_rng(dim).normal(size=(73, dim)) / np.sqrt(dim)
        k = Gaussian(x, .7, backend, device, tile=32)
        exact_x = k.x.cpu().double().numpy()
        d2 = np.sum((exact_x[:, None]-exact_x[None, :])**2, axis=2)
        dense = np.exp(-.7*d2)
        v = np.random.default_rng(0).normal(size=(73, 3))
        err = np.linalg.norm(k.apply(v)-dense@v)/np.linalg.norm(dense@v)
        scale = 1/np.sqrt(dense.sum(1))
        err2 = np.linalg.norm(k.symmetric(v[:, 0], scale)-scale*(dense@(scale*v[:, 0])))
        errors.append(max(float(err), float(err2)))
    if max(errors) > 2e-3:
        raise RuntimeError(f"Kernel/dense comparison failed: {errors}")
    return {"kernel_dense_errors": errors, "backend": backend, "device": device}
