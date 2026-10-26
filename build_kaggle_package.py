"""Build the flat dmap_vae_kaggle/ folder (and .zip) that the Kaggle notebooks expect.

The notebooks look in /kaggle/input for ONE folder holding launch.py, spectral.py, checks.py,
SOURCE.json, the config_*.json files and vendor/. The repository keeps those apart
(src/, tests/, experiments/configs/), so this script flattens them.

    python build_kaggle_package.py            # -> build/dmap_vae_kaggle/ and build/dmap_vae_kaggle.zip
    python build_kaggle_package.py OUT_DIR
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
NAME = "dmap_vae_kaggle"


def build(out_dir):
    pkg = Path(out_dir).resolve() / NAME
    if pkg.exists():
        shutil.rmtree(pkg)
    shutil.copytree(ROOT / "src", pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    extra = [ROOT / "tests" / "checks.py", ROOT / "tests" / "config_smoke.json",
             *sorted((ROOT / "experiments" / "configs").glob("*.json"))]
    for f in extra:
        if (pkg / f.name).exists():
            raise SystemExit(f"Name clash while flattening: {f.name}")
        shutil.copy2(f, pkg / f.name)
    shutil.make_archive(str(pkg), "zip", pkg.parent, NAME)
    return pkg


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build"))
