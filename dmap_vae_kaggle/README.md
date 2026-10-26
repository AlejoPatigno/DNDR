# DMAP congelado + VAE: experimentos Kaggle T4 x2

Implementación compacta: nueve scripts compartidos, cinco notebooks y cuatro configuraciones piloto.
No contiene resultados científicos precalculados ni garantiza calidad generativa.

## Ejecución

1. Crea un Kaggle Dataset con `dmap_vae_kaggle.zip` o con su contenido completo, incluido `vendor/`.
2. Importa `00_checks_kaggle.ipynb`; selecciona **GPU T4 x2**. Activa Internet si debes instalar dependencias.
3. Ejecuta los controles en ambas tarjetas. Si fallan, no interpretes el entrenamiento como una corrida validada.
4. Importa y ejecuta, en orden, `01_swiss_kaggle.ipynb`, `02_torus_kaggle.ipynb`,
   `03_mnist_kaggle.ipynb`, `04_alanine_kaggle.ipynb`.
5. Las configuraciones por defecto son **pilotos** (m=2,8,32). `PAPER_BUDGET=True` amplía a m=256,
   pero sigue sin reproducir N=750000, el beta-flow ni el protocolo completo de FlashDiffusion.
6. Los resultados quedan en `/kaggle/working/results_<dataset>`. Cambia `out` para cada nueva configuración.
   Los logs se actualizan por semilla; los resultados parciales se guardan después de cada modelo.

Dos GPU significan dos procesos independientes, uno por semilla; no DDP de un único VAE ni un
operador repartido entre tarjetas. Cada eigensolve usa ARPACK en CPU y aplicaciones de kernel en su GPU.
Esto aprovecha ambas T4 sin duplicar el código del entrenador. No hay límite artificial de 120 segundos.
El kernel evita almacenar N×N, pero sigue teniendo coste cuadrático: no se promete N=10^6 en una sesión T4.


## Cambios de la versión 5 (protocolo de cierre para toro, MNIST y alanina)

- Notebooks nuevos `02b_torus_protocolo_kaggle.ipynb`, `03b_mnist_protocolo_kaggle.ipynb`, `04b_alanine_protocolo_kaggle.ipynb`: el mismo protocolo del cierre del Swiss roll (05). Tienen hipótesis y criterios fijados antes de correr, `vae` desde cero y por continuación espectral, réplicas independientes, kNN carta/piso como distribución, tasas con Clopper-Pearson y planes que se pueden retomar con ETA.
- `protocol.py`: plan, ETA, ejecución que se retoma, barrido de β, confirmación con `direct`, tablas de despliegue, prueba de orden en m = M y Wilcoxon pareado.
- `beta_sweep.py`: regla por conjunto de datos (`selection`): `graph` (Swiss), `degeneracy` (toro: defecto de degeneración de los pares cos/sin) y `ambient` (MNIST, alanina: lectura afín de x en validación). Los diagnósticos de carta solo se calculan donde existe una carta.
- `experiment.py`: filas `affine_x` (información sin etiquetas del prefijo sobre x, en todos los conjuntos de datos) y `knn_label_accuracy` (solo MNIST, sonda de clases, no carta).
- `data.py`: la submuestra de MNIST sigue a `data_seed` (el valor por defecto 7 reproduce v4). Alanina: `trajectory_rotation` rota qué trayectoria es entrenamiento, validación y test, y la fase del submuestreo.
- `prepare_alanine.py --mdshare`: construye `alanine.npz` desde los NPZ de mdshare (3 × 250 ns, átomos pesados, nm). Usa los archivos si ya están en `/kaggle/input` y, si no, los descarga. φ y ψ se identifican como las únicas cuádruplas enlazadas que reproducen el archivo publicado de diedros en todos los fotogramas, y los enlaces salen de la geometría. El notebook 04b lo ejecuta solo cuando no encuentra un NPZ preparado.
- Regularizadores del latente (`vae.py`, `cfg['reg']`): `iso` (isometría local respecto de los vecinos ambientales) y `jac` (isometría del decoder, distorsión relajada). Ambos son invariantes a escala y actúan sobre μ(x). Nueva métrica `local_distortion` en `experiment.py`. `latent_figure.py` dibuja la figura *truth / m = 2 / 4 / …*. Notebook `06_swiss_latente_kaggle.ipynb`. `07_swiss_lambda_kaggle.ipynb` barre λ de `jac` con el criterio de rectángulo v2 (desplegado, isometría local y forma global R²_s ≥ 0.9, R²_h ≥ 0.8 por `latent_figure.chart_r2_table`). `08_swiss_isometria_final_kaggle.ipynb`: cierre (AE con continuación y `jac`, 12 réplicas; costo frente al piso oráculo → roll exacto; control KL del VAE). `08_swiss_isometria_final_v2_kaggle.ipynb` añade el control sin DMAP (`plainae`, `plainvae`), el error en el latente frente a los puntos del oráculo con alineación de semejanza (`latent_figure.latent_vs_oracle_table`, `rings_figure`) y el Swiss roll sobre cuadrícula perfecta (`grid=True`). Opción `scratch_first_only`: con continuación, entrena desde cero solo en la primera m.
- **AE + oráculo** (`oracle_ae`, `oracle_ae_free`): el encoder afín se inicializa con la lectura del oráculo (misma solución ridge de `affine_fit`, dividida por una escala isótropa); `oracle_ae` lo deja fijo y solo entrena el decodificador, `oracle_ae_free` lo deja libre. **Cuadrícula perfectamente separada** (`grid='square'`, `data.square_grid_chart`): el mismo paso en s y h (132 × 22, Δ = 0.4545, N = 2904). Notebook `11_swiss_oraculo_cuadricula_kaggle.ipynb` (≈ 6 h con 2 × T4; exporta `paper_figs_v3.zip`). Controles nuevos en `checks.py`: `oracle_head_checks` y la cuadrícula cuadrada en `protocol_checks`.
- `checks.py`: `protocol_checks()` cubre estas reglas; `distortion_checks()` y `reg_checks()` (en GPU) cubren los regularizadores.
- `05`: la figura resumen pasa a ser 2×2. El panel (c) muestra el oráculo en ambas geometrías con los pisos del teorema y el panel (d) el despliegue desde cero y por continuación en ambas geometrías.

## Cambios de la versión 4 (cierre del Swiss roll)

- `mode_order` ('spectral' | 'shuffled', con `mode_order_seed`): ablación del orden de los modos. Una
  permutación fija de los M modos por réplica; cada m sigue siendo un prefijo anidado, así que la continuación
  y el argumento de clases anidadas siguen valiendo. La permutación queda en `provenance.json`. La prueba
  limpia es en m = M, donde ambos órdenes tienen exactamente los mismos modos.
- `chart_decoder_epochs`: presupuesto más largo para el decoder de referencia sobre la carta verdadera.
- `swiss_ensembles*`: ejes fijados a la extensión de la referencia; se informa cuántos puntos quedan fuera.
- Notebook `05_swiss_cierre_kaggle.ipynb`: experimentos A (16 réplicas de continuación espectral),
  B (orden aleatorio) y D (geometría 30×10, con su propio barrido de β), con criterios fijados de antemano,
  intervalos Clopper-Pearson y pruebas de Fisher / Mann-Whitney.

## Cambios de la versión 3 (reinicios, signo de autovectores, recuperación de carta)

- `spectral.fix_signs`: convención de signo determinista (la entrada de mayor |valor| de cada autovector es
  positiva). Antes el mismo operador daba features distintas si solo cambiaba `max(modes)`, porque ARPACK fija
  el signo de forma arbitraria; eso hacía variar el AE hasta ×20 entre corridas idénticas. No resuelve
  rotaciones dentro de espacios propios exactamente degenerados (p. ej. los pares cos/sin del toro).
- `restarts` (config): cada modelo se entrena R veces con semilla de inicialización `seed + 7919·r`; datos y
  partición no cambian. r=0 reproduce las corridas anteriores. Columna `restart`; archivos `*_r{r}` para r>0.
- `knn_chart_mse`: regresión kNN latente→carta (ajustada en train, evaluada en test). Es invariante a
  reparametrizaciones suaves del latente, a diferencia del alineamiento rígido/afín. `knn_chart_floor` = mismo
  estimador sobre la carta verdadera; `knn_chart_ratio` = cociente; `unfolded` = cociente ≤ `unfold_ratio` (10).
  Funciona también con la carta circular del toro y la alanina.
- `plots.py`: `unfolding.csv` (tasa de despliegue, sobre todos los reinicios y quedándose con el mejor por
  validación), `chart_recovery.png`, figuras de ensambles y curvas de entrenamiento POR MODELO, cada una en su
  propia m elegida por validación, y colores fijos por modelo.

## Cambios de la versión 2 (barrido de β y revisión del tutor)

**Hallazgo.** Con `beta_factor=16` (heurística anterior), la longitud del kernel (≈2.4) es comparable a la
separación entre capas del Swiss roll (2π·0.6≈3.77): ~6 % de la masa del kernel conecta capas distintas.
ψ₁ no se parece a cos(πs/W) (|corr|≈0.3) y el primer modo de altura aparece en la posición 4, no en la 6
esperada para un rectángulo 60×10. El diccionario describía la geometría ambiental, no la intrínseca.

**Selección de β sin etiquetas** (`beta_sweep.py`, notebook `01a`): masa del kernel en pares cuya distancia
geodésica en un grafo kNN (k=10) supera 2× la distancia ambiental; se elige el menor factor con masa ≤10⁻³ en
todas las semillas de datos y controles espectrales superados. El β-flow clásico se reporta pero no se usa:
su máximo cae dentro del régimen de cortocircuito. Los diagnósticos con la carta verdadera (masa real entre
capas, ψ₁, modo de altura, oráculos) se registran solo como verificación.

**Experimento** (`experiment.py`, `vae.py`, `geometry.py`, `plots.py`):
- `mse_x` (ambiental, por coordenada) y `chart_mse_z` (carta, por coordenada) con nombres distintos.
- Oráculo → enrollado exacto g(Π_Ω ẑ) con la cota MSE_X ≤ ⅔ MSE_z y el cociente de Lipschitz empírico.
- Decoder entrenado sobre la carta verdadera y evaluado con ẑ (`chart_decoder`, `oracle_decoder`).
- Continuación en m (`*_warm`: pesos anteriores, columnas nuevas en cero; la inicialización es candidata en la
  selección por validación) frente a entrenamiento desde cero.
- `vae_from_ae`: VAE inicializado con la cabeza de media y el decoder del AE del mismo m, con KL warm-up.
- Reconstrucción, KL, varianzas posteriores y unidades activas por separado; curvas de entrenamiento.
- Alineamientos rígido, por similitud y afín (ajustados en train) y el predictor constante como referencia.
- Generación: configuración elegida por validación, D(μ) y E_q[D(z)], N igual al de test, distancia a la
  superficie exacta, cobertura y TV de la carta, con los mismos valores para la muestra de referencia.
- `resample_data`: cada semilla cambia datos y partición (réplicas independientes también del oráculo).
- `encoder_stored_floats` declara el almacenamiento del encoder Nyström; `neural_parameters` solo la red.

## Código y fuentes

- `spectral.py`: operador, adaptador FlashDiffusion, referencia PyTorch, Nyström y persistencia.
- `vae.py`: cabezas afines de media/logvarianza, decoder nonlinear y entrenamiento/congelación.
- `data.py`: Swiss roll isométrico, toro plano T² en R⁴, MNIST y entradas moleculares.
- `experiment.py`: readout oracle, AE espectral, VAE espectral, decoder directo, VAE convencional y CVAE.
- `launch.py`: colas de semillas en las dos GPU, logs y control de fallos.
- `plots.py`: CSV, PNG y PDF; separa reconstrucción de generación.
- `prepare_alanine.py`: conversión de trayectorias emparejadas con su topología.
- `generate_angles.py`: muestreo condicional desde un checkpoint entrenado y congelado.
- `checks.py`: controles pequeños de implementación.
- `geometry.py`: g exacta, alineamientos, métricas de generación y diagnósticos de β (solo NumPy).
- `beta_sweep.py`: barrido de β en dos GPU, regla de selección sin etiquetas y figura.

Se intentó leer https://github.com/sparsetrace/FlashDiffusionTests/tree/main mediante GitHub:
el acceso devolvió **404**. Su contenido actual NO fue inspeccionado ni se certifica compatibilidad con
sus tests. La implementación utiliza el repositorio público
https://github.com/sparsetrace/FlashDiffusion/tree/f5c2d1abb7e418829651f7e2fedbb8b323fd56f9 .
`vendor/flash_diffusion.cu` es una copia **sin cambios** de
`src/flashdiffusion/csrc/flash_diffusion_sm80.cu` de ese commit, con su licencia MIT.
El adaptador compila esa fuente WMMA para la arquitectura detectada (T4: SM75), sin invocar el
autodespacho upstream que no declara SM75. Es una adaptación de compilación, no soporte T4 certificado
por los autores. El notebook 00 verifica compilación y comparación numérica en el hardware real.

Se evita `DiffusionMap.transform()` upstream: la conversión aquí se obtiene de
S=D^(-1/2) K_alpha D^(-1/2), psi=D^(-1/2)v, con normalización L2(pi).
No se utiliza la fórmula general incorrecta psi=v/rscale.

## Contrato matemático y correcciones

- K_ij=exp(-beta ||x_i-x_j||²), diagonal retenida; mismo tratamiento en fit y Nyström.
- q=K1, K_alpha=diag(q^-alpha) K diag(q^-alpha), D=diag(K_alpha 1), P=D^-1 K_alpha.
- Los modos no triviales se normalizan con pi_i=D_ii/sum D_ii; psi.T diag(pi) psi=I.
- La salida congelada son **funciones propias psi**, sin multiplicación lambda^t (t=0).
- Nyström usa P(x,j)=K(x,x_j) q_j^-alpha / sum_l K(x,x_l) q_l^-alpha y divide por lambda.
- Los controles numéricos detienen residuos, separación de Perron o evaluación Nyström insuficientes.
  Las tolerancias son controles de ingeniería explícitos, no constantes de un teorema.
- La implementación Flash almacena coordenadas FP16 y acumula FP32. PyTorch usa FP32 en GPU y FP64
  en CPU. La comparación densa usa las coordenadas **efectivamente cuantizadas** del kernel.
  No se afirma identidad numérica con el experimento FP32 del artículo.
- No se llama FlashDiffusion al backend `torch`. Debe elegirse explícitamente, nunca como fallback silencioso.
- alpha=1 en los ejemplos geométricos; alpha=0.5 en alanina. Con muestreo no uniforme no representan
  necesariamente el mismo generador límite. No se equiparan los resultados sin justificarlo.
- beta=None: mediana de distancias de entrenamiento multiplicada por `beta_factor`; **no beta-flow**.
  Beta y el presupuesto se deben validar sin usar test. Si el control falla, ajustar la configuración,
  cambiar el presupuesto o estudiar precisión; no relajar umbrales solo para conseguir una figura.
- Escalado de entradas/salidas: centrado por componente y UN escalar RMS de entrenamiento, conserva
  razones de aspecto. Escalado de modos: media/desviación por modo aprendidas solo con entrenamiento,
  guardadas en el checkpoint. Esto cambia el gauge y debe deshacerse para analizar coeficientes espectrales.
- Los espacios propios degenerados se interpretan como bloques; no se asigna significado invariante
  a columnas individuales ni se afirma equivalencia del truncamiento si corta una multiplicidad.
- Las coordenadas verdaderas se usan solo para evaluación y el control oracle. CVAE usa ángulos
  explícitamente: es una variante supervisada, no evidencia de identificación espontánea.
- El oracle usa ridge real, escalado por tr(Gram)/m; el intercepto se ajusta por centrado y no se penaliza.
- El VAE usa media y logvarianza AFINES, sin clipping de logvarianza. Se entrenan cabezas y decoder;
  las funciones espectrales permanecen fijas. Se congela todo al seleccionar el checkpoint.
- Objetivo: E||D(z)-x||²/(2 tau²) + beta_KL KL. tau fijo; sumas sobre componentes antes de promediar
  ejemplos. KL por posterior condicional, no sustitución por covarianza agregada.
- Época elegida por ELBO de validación estimada con ruido fijo; reporta aparte reconstrucción por media.
  El entrenamiento aplica clipping del gradiente, no de las funciones afines.
- `plainvae` tiene encoder nonlinear; `ae` es determinista; `direct` recibe todos los m modos sin
  cuello de botella. Se registra número de parámetros: no se afirma igualdad de capacidad total.
- Semillas cambian entrenamiento; las particiones y el diccionario objetivo son fijos. La desviación
  entre semillas mide variabilidad de optimización, no incertidumbre estadística de la población.
- El toro tiene latente euclídeo 2D como control de la obstrucción topológica. No se afirma una carta
  global inyectiva ni una realización de un VAE toroidal. `latent=4` permite una ablación dimensional.
- Ningún experimento demuestra por sí solo tasas minimax, isometría global, reach, ni novedad matemática.

## Alanina: datos y alcance

Se requiere un NPZ con:

| Clave | Forma | Significado |
|---|---|---|
| xyz | N×A×3 | Átomos pesados en nm, orden consistente |
| trajectory | N | ID de trayectoria independiente |
| phi_indices | 4 | Cuádrupla atómica de phi, índices desde cero |
| psi_indices | 4 | Cuádrupla atómica de psi |
| bonds | B×2 | Enlaces entre los átomos guardados |

Conversión desde datos propios compatibles con MDTraj:

```bash
pip install mdtraj
python prepare_alanine.py --topology ala.pdb --trajectories traj0.xtc traj1.xtc traj2.xtc \
    --stride 10 --out alanine.npz
```

Verifica explícitamente las cuádruplas reportadas y el orden de átomos. No se descarga automáticamente
un archivo de diedros ajeno a las coordenadas. No se suministra un dataset molecular inventado.
El convertidor presupone una topología de la molécula de interés; elimina hidrógenos. Para más de una
pareja backbone hay que elegir `--phi-column` y `--psi-column` correctamente.

Partición por defecto: trayectoria 0 entrenamiento, 1 validación, 2 prueba. Se conserva orden temporal
al submuestrear. Solo se usan las primeras tres trayectorias, registradas por sus IDs.
El encoder se ajusta solo en entrenamiento. `protocol="transductive"` incluye todos los puntos en el
operador y se etiqueta como tal, pero mantiene la misma partición del decoder: tampoco es una réplica
exacta de las dos trayectorias de entrenamiento y 5000 frames aleatorios del artículo.

Los puntos moleculares se orientan en un marco SO(3) N-CA-C que conserva lateralidad; los inputs
espectrales son distancias de cada átomo a los primeros ocho átomos pesados. La salida del VAE son
coordenadas, no solo los cuatro senos/cosenos. No se atribuye identificación de quiralidad a esas distancias.
Los diedros predichos se vuelven a calcular desde la estructura. Los diedros indefinidos se reportan
con su fracción válida y no se sustituyen por errores cero.

El control oracle reconstruye cuatro coordenadas circulares. El VAE reconstruye estructuras. Las figuras
`angles_*_reconstructed` y `angles_generated` son análisis distintos; no prueban el mismo resultado.
La superficie -log p+C_ref usa bins compartidos, máscara para bins vacíos y la constante de referencia.
Valores por encima de 8 están saturados solo visualmente; las probabilidades completas se guardan en NPZ.
No se etiqueta como energía libre de equilibrio la distribución producida por un modelo sin validación física.

Para generar después del entrenamiento:

```bash
python generate_angles.py results_alanine/seed_0/cvae_m32.pt --phi -60 --psi -40 --n 64 \
    --out conditioned_candidates.npz
```

CVAE concatena la condición circular a las entradas de las cabezas y del decoder; el muestreo usa un prior
residual N(0,I). No impone exactamente los ángulos. Su error debe medirse; si es alto, el experimento falla
como control angular. No es un generador físico certificado. Los controles empíricos de enlaces,
vecinos no enlazados y signos quirales no sustituyen un campo de fuerza ni una validación de equilibrio.
No se generan proteínas completas ni se promete control físico de todos sus grados de libertad.

## Validación y dependencias

Los notebooks no llevan resultados falsos de GPU. Consultar VALIDATION.md para las pruebas realizadas.
Kaggle proporciona PyTorch/CUDA; no se reemplaza su instalación. Dependencias adicionales:
NumPy, SciPy, pandas, Matplotlib y ninja. MDTraj solo para preparar trayectorias.
`environment.txt` y `provenance.json` registran versiones efectivas; no se promete reproducibilidad bit a bit
entre dispositivos o versiones de CUDA.

Prueba local pequeña, sin CUDA:

```bash
python checks.py --backend torch --device cpu --out checks_cpu.json
python launch.py config_smoke.json --cpu
```

La configuración smoke usa `/tmp/dmap_smoke_swiss`; cambiar la ruta si ya existe una corrida completa.
No cargar checkpoints `.pt` de terceros no confiables; el cargador está destinado a los archivos propios.
- **Swiss roll con volumen** (`dataset='swissvol'`, `data.swiss_volume`, `geometry.swiss_volume_embed`): la lámina 60 × 10 desplazada t ∈ [−T/2, T/2] a lo largo de la normal de la espiral (T = 2 por defecto; las vueltas quedan a 1.77). Carta (s, h, t), dimensión intrínseca 3. `volume.py`: predicción del primer modo de h (ψ₅) y de t (ψ₁₂₉ para T = 2), barrido de la lectura afín por eje (DMAP, IMAP, UMAP), fracción de varianza de cada eje por modo y tabla latente 3-D frente al oráculo. Notebook `12_swiss_volumen_kaggle.ipynb` (≈ 3.5 h con 2 × T4; exporta `paper_figs_vol.zip`).
