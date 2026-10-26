# DNDR — DMAP congelado + VAE/AE

Experimentos de la sección de aprendizaje del artículo *"DMAP is not NDR"*: un encoder espectral
de Diffusion Maps **congelado**, cabezas afines y un decoder MLP, evaluados en Kaggle con GPU T4 x2.
Qué se evalúa y por qué: [`context/README.md`](context/README.md).

## Estructura

| Carpeta | Contenido |
|---|---|
| [`context/`](context/) | Contexto del proyecto (`README.md`), detalle técnico del paquete (`PACKAGE.md`) y estado de validación (`VALIDATION.md`) |
| [`src/`](src/) | Código Python, `SOURCE.json` y `vendor/` (kernel CUDA de FlashDiffusion) |
| [`experiments/`](experiments/) | Notebooks de experimentos (01–12) y sus configuraciones en `configs/` |
| [`tests/`](tests/) | Notebook de controles `00_checks_kaggle`, `checks.py`, configuración smoke y resultado de validación en CPU |
| [`build_kaggle_package.py`](build_kaggle_package.py) | Junta `src/`, `tests/` y `experiments/configs/` en la carpeta plana que esperan los notebooks |
| [`.github/workflows/kaggle-gpu.yml`](.github/workflows/kaggle-gpu.yml) | CI: sube el paquete a Kaggle y ejecuta los notebooks con GPU T4 x2 |

## Uso rápido

```bash
python build_kaggle_package.py   # -> build/dmap_vae_kaggle/ y build/dmap_vae_kaggle.zip
```

- **Desde GitHub:** *Actions → Kaggle GPU notebooks → Run workflow* (requiere los secretos
  `KAGGLE_USERNAME` y `KAGGLE_API_TOKEN` o `KAGGLE_KEY`). Cada push a `main` corre los controles de `tests/`.
- **A mano en Kaggle:** crea un Dataset con `build/dmap_vae_kaggle.zip`, importa el notebook y
  selecciona **GPU T4 x2**.

Detalles, límites y prueba local sin GPU en [`context/README.md`](context/README.md).
