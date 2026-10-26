# DNDR — DMAP congelado + VAE/AE

Código y notebooks de los experimentos de la sección de aprendizaje del artículo
*"DMAP is not NDR"*, pensados para correr en Kaggle con **GPU T4 x2**. El detalle técnico
(contrato matemático, cambios por versión, alanina) está en [`PACKAGE.md`](PACKAGE.md);
lo que se validó y lo que no, en [`VALIDATION.md`](VALIDATION.md).

## Qué busca evaluar el proyecto

**Tesis.** Los modos de Diffusion Maps (DMAP) son una *base espectral* de la geometría intrínseca,
no una *carta* de baja dimensión. La carta correcta está en el span de los modos, pero elegirla
requiere un principio externo (lectura afín con oráculo, isometría, etc.).

**Teorema del prefijo.** Ninguna lectura afín de un prefijo corto de modos contiene un eje
intrínseco hasta que entra su primer modo. Ejemplos: en el Swiss roll 60×10 la altura `h` aparece
recién en ψ₅; en el Swiss roll con volumen, el grosor `t` aparece en ψ₁₂₉.

**Pregunta experimental.** Si el encoder es el DMAP **congelado** (extensión de Nyström de los
primeros `m` modos) seguido de cabezas afines y un decoder MLP, ¿basta con minimizar la
reconstrucción para que el latente recupere la carta?

Lo que se contrasta:

- **Reconstrucción vs. carta.** Con continuación en `m` el rollo se despliega y la reconstrucción
  es la mejor, pero el latente puede quedar doblado: la reconstrucción sola no selecciona la carta.
- **Regularización isométrica** (`jac`, isometría del decoder; `iso`, isometría local). Se mide si
  el latente coincide con el oráculo DMAP y cuánto cuesta en reconstrucción.
- **Controles sin DMAP** (`plainae`, `plainvae`), **orden de modos aleatorio** (`mode_order='shuffled'`)
  y **AE con oráculo** (`oracle_ae`, `oracle_ae_free`).
- **Otras geometrías:** toro plano T² en R⁴ (obstrucción topológica de un latente euclídeo 2-D),
  MNIST (sin carta conocida), dipéptido de alanina (diedros φ/ψ) y Swiss roll con volumen (3-D).

Ningún experimento demuestra por sí solo tasas minimax, isometría global ni novedad matemática;
los criterios de éxito se fijan **antes** de correr cada protocolo.

## Arquitectura del código

```
x ──► DMAP congelado (ψ₁..ψₘ, Nyström) ──► cabezas afines μ, log σ² ──► z ──► decoder MLP ──► x̂
```

Todo está en [`src/`](../src/), salvo `checks.py`, que está en [`tests/`](../tests/).

| Archivo | Responsabilidad |
|---|---|
| `spectral.py` | Operador DMAP (kernel FlashDiffusion compilado para SM75 o referencia PyTorch), Nyström, persistencia |
| `vae.py` | Cabezas afines, decoder, entrenamiento/congelación, regularizadores `iso`/`jac` |
| `data.py` | Swiss roll (aleatorio, cuadrícula, volumen), toro, MNIST, alanina |
| `experiment.py` | Una corrida (una semilla): oráculo, AE/VAE espectral, `direct`, `plainvae`, CVAE y métricas |
| `launch.py` | Reparte semillas entre las dos T4 (un proceso por GPU) |
| `beta_sweep.py` | Selección de β sin etiquetas |
| `protocol.py` | Planes reanudables, réplicas, pruebas estadísticas (Fisher, Mann-Whitney, Wilcoxon, Clopper-Pearson) |
| `geometry.py`, `volume.py`, `latent_figure.py` | Geometría exacta, alineamientos y análisis del latente |
| `plots.py`, `export_figs.py` | CSV y figuras |
| `checks.py` | Controles numéricos de implementación (kernel vs. referencia densa, ortogonalidad L²(π), Nyström…) |
| `vendor/flash_diffusion.cu` | Copia sin cambios del kernel de [FlashDiffusion](https://github.com/sparsetrace/FlashDiffusion) (MIT), verificada por SHA-256 contra `SOURCE.json` |

## Métricas y criterios principales

- **`unfolded`**: `knn_chart_mse ≤ 10 × knn_chart_floor` (regresión kNN latente→carta frente al mismo
  estimador sobre la carta verdadera; invariante a reparametrizaciones suaves).
- **Tasas de despliegue** con intervalos Clopper-Pearson 95 % sobre réplicas independientes.
- **Orden espectral**: Fisher exacto, Mann-Whitney o Wilcoxon pareado en `m = M`.
- **Rectángulo v2** (notebooks 07 y 08): desplegado, distorsión local ≤ 2× la del oráculo,
  R²_s ≥ 0.90 y R²_h ≥ 0.80.
- **Costo**: `MSE_x` relativo al modelo sin regularizar; admisible si ≤ 2×.
- **Error de latente**: Procrustes con escala frente a la lectura del oráculo (0 perfecto, 1 sin información).

## Notebooks

`00_checks_kaggle` está en [`tests/`](../tests/); el resto, en [`experiments/`](../experiments/).

| Notebook | Tipo | Qué hace | Duración aprox. (2×T4) |
|---|---|---|---|
| `00_checks_kaggle` | Validación | Compila el kernel y corre `checks.py` en ambas GPU | minutos |
| `01a_swiss_beta_sweep_kaggle` | Experimento | Barrido de β sin etiquetas (Swiss roll) | 10–55 min |
| `01_swiss_kaggle` | Experimento | Swiss roll completo con β seleccionado | 15–25 min |
| `02_torus_kaggle` / `02b_…protocolo` | Experimento | Toro básico / protocolo de cierre T1–T5 | — / ~5.5 h |
| `03_mnist_kaggle` / `03b_…protocolo` | Experimento | MNIST básico / protocolo M1–M4 (descarga MNIST) | — / ~3.5 h |
| `04_alanine_kaggle` / `04b_…protocolo` | Experimento | Alanina básico (requiere NPZ preparado) / protocolo L1–L5 (descarga mdshare) | — / ~4 h |
| `05_swiss_cierre_kaggle` | Experimento | Experimentos A, B y D del Swiss roll | ~7 h |
| `06_swiss_latente_kaggle` | Experimento | ¿El latente es el rectángulo? (sin reg., `iso`, `jac`) | ~4.5 h |
| `07_swiss_lambda_kaggle` | Experimento | Barrido de λ de `jac` | 3–4.5 h |
| `08_swiss_isometria_final_kaggle` / `_v2` | Experimento | Cierre de isometría; v2 añade control sin DMAP y cuadrícula | ~3 h / 2–5 h |
| `09_exportar_figuras_kaggle` | Figuras | Exporta `paper_figs.zip` (**requiere la salida de 08 v2**) | 5–10 min |
| `11_swiss_oraculo_cuadricula_kaggle` | Experimento | AE + oráculo y cuadrícula separada 132×22 | ~6 h |
| `12_swiss_volumen_kaggle` | Experimento | Swiss roll con volumen (dimensión intrínseca 3) | ~3.5 h |

Varios notebooks reutilizan resultados de otros desde `/kaggle/working` (p. ej. `beta_selection.json`
de 01a). En casi todos es **opcional**: si no está, recalculan o usan β por defecto. Las excepciones
son `09` (necesita la salida de 08 v2 en la misma sesión) y `04` (necesita
`/kaggle/input/alanine-prepared/alanine.npz`).

## Ejecución en Kaggle desde GitHub Actions

Los notebooks esperan **una sola carpeta** con todo el código, las configuraciones y `vendor/`.
[`build_kaggle_package.py`](../build_kaggle_package.py) la arma a partir de `src/`, `tests/checks.py`,
`tests/config_smoke.json` y `experiments/configs/` (deja `build/dmap_vae_kaggle/` y
`build/dmap_vae_kaggle.zip`; el ZIP sirve para crear el Dataset a mano).

El workflow [`.github/workflows/kaggle-gpu.yml`](../.github/workflows/kaggle-gpu.yml):

1. Arma ese paquete y lo sube como nueva versión del Dataset privado
   `<KAGGLE_USERNAME>/dndr-dmap-vae-kaggle` (los notebooks lo buscan en `/kaggle/input`).
2. Publica cada notebook elegido como kernel privado `dndr-<nombre>` con **GPU T4 x2** e Internet,
   uno a la vez.
3. Espera a que termine, descarga la salida (logs, CSV, figuras) y la guarda como artefacto del run.

**Disparadores**

- `push` a `main` que toque `src/`, `tests/`, `experiments/` o el script de empaquetado: corre solo
  `00_checks_kaggle`.
- Manual (*Actions → Kaggle GPU notebooks → Run workflow*):
  - Una casilla por notebook (con su duración aproximada); se ejecutan los marcados, en orden
    numérico. Por defecto solo está marcado `00_checks_kaggle`.
  - `action`: `run` (sube el código, ejecuta y espera) o `collect` (solo consulta el estado y descarga
    la salida de la última ejecución).

**Secretos del repositorio** (*Settings → Secrets and variables → Actions*):

| Secreto | Valor |
|---|---|
| `KAGGLE_USERNAME` | Usuario de Kaggle |
| `KAGGLE_API_TOKEN` | Token de API nuevo (*kaggle.com/settings → API*), **o bien** |
| `KAGGLE_KEY` | Clave del `kaggle.json` legado |

**Límites a tener en cuenta**

- Un job de GitHub se corta a las 6 h; Kaggle sigue ejecutando (hasta 12 h por sesión GPU). Para
  notebooks largos (05, 11, 02b…) lanza con `run` y, cuando termine en Kaggle, recupera la salida con
  `collect`.
- Cada notebook corre en su **propia** sesión de Kaggle: `/kaggle/working` no se comparte entre
  notebooks. `09` no funciona si se lanza por separado de 08 v2.
- La cuota semanal de GPU de Kaggle es limitada; el `push` solo lanza los controles (00) a propósito.

## Prueba local sin GPU

Se corre sobre el paquete armado, igual que en Kaggle:

```bash
python build_kaggle_package.py
cd build/dmap_vae_kaggle
python checks.py --backend torch --device cpu --out checks_cpu.json
python launch.py config_smoke.json --cpu
```
