"""Generate from a trained conditional checkpoint; outputs remain UNVALIDATED candidates."""
import argparse
import json
import numpy as np
import torch
from data import circular, angles, angle_metrics
from data import output_angles
from vae import build, generate


def sample(checkpoint, phi, psi, n, out):
    # Load only your own checkpoint; torch checkpoints are pickle-based.
    a = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if a["model_cfg"]["kind"] != "cvae":
        raise ValueError("An unconditional VAE has no identified angular control. Use a cvae checkpoint.")
    model = build(a["model_cfg"])
    model.load_state_dict(a["state"])
    model.eval().requires_grad_(False)
    c = circular(np.tile(np.radians([phi, psi]), (n, 1)))
    xyz = generate(model, n, "cpu", c)*a["target_scale"]+a["target_center"]
    recovered = output_angles(xyz, "alanine", a["meta"])
    err = np.degrees(np.abs(np.angle(np.exp(1j*(recovered-angles(c))))))
    np.savez_compressed(out, xyz_nm=xyz.reshape(n, a["meta"]["atoms"], 3),
                        requested_deg=np.tile([phi, psi], (n, 1)), measured_deg=np.degrees(recovered),
                        angular_error_deg=err)
    print(json.dumps({"status": "unvalidated_generated_candidates", "n": n,
                      **angle_metrics(recovered, angles(c))}, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--phi", type=float, required=True)
    p.add_argument("--psi", type=float, required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", default="generated_candidates.npz")
    sample(**vars(p.parse_args()))
