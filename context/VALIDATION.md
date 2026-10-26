# Estado de validación

## Versión 4: qué se ejecutó

- Notebook 05 completo en CPU con dobles NumPy de `torch`, `vae` y `SpectralMap` (este último con DMAP denso
  real): barrido 30×10 lanzado en dos procesos, las cuatro configuraciones lanzadas por `launch.py`, análisis
  y figura resumen. Se omitieron solo la preparación de Kaggle y los controles en GPU.
- `mode_order='shuffled'`: permutación distinta por réplica, registrada, y el prefijo m = M contiene
  exactamente los mismos modos que el orden espectral.
- Con el doble NumPy, el decoder directo en 30×10 cae por debajo de piso/10 en m = 3 (predicción: 3 o 4).
- NO ejecutado aquí: el código PyTorch. El notebook 05 corre `checks.py` en GPU antes de todo lo demás.


## Versión 3: qué se ejecutó

- `fix_signs`: probado con autovectores ortonormales aleatorios y signos volteados (resultado idéntico, entrada
  máxima positiva). En Kaggle, `checks.py` verifica que el operador real con 5 y con 12 modos dé las mismas
  features (producto interno L2(π) > 0.999, sin permitir cambio de signo).
- `knn_chart_mse`: reparametrización suave del Swiss roll = 2.4–2.6× el piso; carta plegada |s−30| ≈ 800–1500×.
- `experiment.py` + `plots.py` de extremo a extremo con dobles NumPy: Swiss roll con 3 reinicios, `warm`,
  `vae_from_ae` y `plainvae`; toro con 2 reinicios. Celdas de la etapa 3 del notebook probadas sobre esas salidas.
- NO ejecutado aquí: el código PyTorch (sin acceso a PyPI). Vuelve a correr el notebook 00 antes de la etapa 3.


## Versión 2 (barrido de β): qué se ejecutó y qué no

Ejecutado en CPU en este entorno (sin PyTorch disponible):
- `geometry.py` completo: cota del oráculo (cociente de Lipschitz 1.00003), alineamientos (similitud recupera
  una carta rotada y escalada con error ~1e-28), métricas de superficie, masa de atajos por grafo frente a la
  masa verdadera (coinciden hasta ~1e-3), regla de selección.
- Reproducción NumPy del DMAP de la corrida anterior (factor 16): oráculo 131 / 14.1 / 0.85 para m=2/8/16,
  frente a 133 / 14.1 / 0.88 reportados.
- `beta_sweep.py` de extremo a extremo (dos procesos, combinación, figura) con un doble NumPy de `SpectralMap`
  que aplica los mismos controles.
- `experiment.py` y `plots.py` de extremo a extremo (Swiss roll y toro) con dobles NumPy de `torch` y `vae`:
  nombres de métricas, cadena de continuación, AE→VAE, semillas de datos, β desde `beta_selection.json`, figuras.

NO ejecutado aquí: el código PyTorch nuevo de `vae.py` (expansión con ceros, candidato de época −1, KL warm-up,
E_q[D(z)], estadísticas posteriores). Está cubierto por `checks.py`, que ejecuta el notebook 00 en Kaggle
antes de cualquier corrida: **vuelve a correr 00 con este paquete**.

## Versión 1

## Ejecutado en CPU

- Análisis sintáctico de los nueve `.py` y de todas las celdas de código.
- Validación nbformat de los cinco notebooks.
- Comparación del operador por bloques PyTorch con una matriz gaussiana densa: N=73 y D=3,80,784.
- Ortogonalidad L2(pi), residuos propios, separación de Perron y consistencia Nyström en Swiss roll pequeño.
- Guardado/carga del encoder y reproducción de sus coordenadas fuera de muestra.
- Entrenamiento breve del VAE, formas de los tensores y congelación de todos los parámetros al terminar.
- Sentido del diedro, invariancia ante SO(3) y traducción y periodicidad de la representación circular.
- Lanzador CLI con dos semillas de Swiss roll, modos 2 y 8, dos épocas y backend CPU explícito.
- Recorridos de entrenamiento, evaluación, checkpoints y figuras para toro (N=180).
- MNIST real: descarga del NPZ, 200 ejemplos del conjunto oficial de entrenamiento (160 train/40 val)
  y 100 ejemplos del test oficial; modos 2 y 8, dos épocas. Se ejecutaron AE, VAE, decoder directo
  y VAE convencional. Esto NO mide rendimiento científico y no constituye una comparación concluyente.
- Rama molecular sobre fixture sintético de 180 configuraciones de 10 puntos, tres grupos, incluyendo
  CVAE, diagnóstico estructural y generación condicionada desde checkpoint.
- Convertidor MDTraj sobre una topología sintética ACE-ALA-NME y tres archivos XTC pequeños:
  salida 18×80 características; las cuádruplas detectadas fueron phi=[1,3,4,6], psi=[3,4,6,8].
  Diferencia angular máxima respecto a MDTraj: menor que 5e-7 rad.
- Inspección visual de ejemplos renderizados de las figuras; se corrigieron la máscara de bins vacíos
  y el uso de escalas distintas entre paneles Swiss roll.
- Copia CUDA vendorizada verificada byte a byte contra el commit upstream registrado en SOURCE.json.

Los resultados numéricos de los controles del operador están en `validation_cpu.json`.
Las fixtures y figuras de depuración no se presentan como datos científicos y no se incluyen en el paquete.

## No ejecutado / pendiente

- No hay GPU CUDA en el entorno de preparación. NO se compiló ni se ejecutó el kernel SM75 aquí.
- NO se ejecutaron los notebooks Kaggle de principio a fin en T4 x2. El notebook 00 contiene los
  controles necesarios en cada tarjeta, y los workers repiten el control del backend antes de entrenar.
- No se certifica aceleración, uso de memoria ni convergencia a grandes N en dos T4.
- No se auditó el repositorio FlashDiffusionTests solicitado: el acceso devolvió 404. Se utilizó la
  fuente pública FlashDiffusion fijada al commit declarado. Se requiere acceso a aquel repositorio
  para certificar equivalencia con sus experimentos actuales.
- Falta ejecutar con las trayectorias reales de alanina y su topología. Ningún resultado físico,
  energía libre de equilibrio, control angular preciso o generación de proteínas queda establecido.
- No se validó el método de desembalado periódico en una trayectoria física con cruces de celda;
  usa la operación oficial make_molecules_whole de MDTraj antes de calcular diedros no periódicos.
- Los pilotos no replican beta-flow, N=750000 ni el protocolo exacto de la figura 2.

## Comandos de validación en destino

1. Activar GPU T4 x2 y ejecutar `00_checks_kaggle.ipynb`.
2. Si ambos controles pasan, ejecutar `01_swiss_kaggle.ipynb` con el presupuesto piloto.
3. Inspeccionar los JSON de procedencia y los logs. Un fallo de residuo, Perron o Nyström invalida
   esa corrida; no se relajan automáticamente las tolerancias.
4. Continuar con toro, MNIST y alanina. Evaluar calidad fuera de muestra y, en generación condicional,
   los ángulos medidos desde la salida, no únicamente los ángulos suministrados como condición.

Referencias de las interfaces moleculares:
https://mdtraj.readthedocs.io/en/latest/api/generated/mdtraj.iterload.html
https://mdtraj.readthedocs.io/en/latest/api/generated/mdtraj.compute_phi.html
https://mdtraj.readthedocs.io/en/latest/api/generated/mdtraj.Trajectory.html


## v5 (sin GPU en este entorno)

Los tres notebooks de protocolo y `protocol_checks()` se ejecutaron de principio a fin con dobles numpy de `torch`, `spectral` y `vae`, sobre datos sintéticos (MNIST de manchas y un NPZ de alanina con tres trayectorias). Esto valida la plomería, no los números. `checks.py` en GPU (celda 2 de cada notebook) es el control real.
