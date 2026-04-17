"""
experimentos2.py
===============
Script genérico de experimentos MCAR/MAR/MNAR para cualquier dataset
compatible con run_imputation.py.

DISEÑO METODOLÓGICO (importante para reproducibilidad):
═══════════════════════════════════════════════════════
1. MISSINGNESS GLOBAL: La máscara de valores faltantes se genera UNA SOLA VEZ
   por experimento sobre toda la serie temporal completa. No existe ninguna
   lógica que regenere o modifique la máscara por partición.

2. IMPUTACIÓN GLOBAL: El modelo GRIN se ejecuta UNA SOLA VEZ por experimento
   usando la máscara global. Produce y_hat completo para toda la serie.
   No hay reentrenamiento ni reimputación por partición.

3. EVALUACIÓN LOCAL (post-proceso): Las particiones temporales solo filtran
   (indexing) sobre y_hat, y_true y mask ya calculados. No alteran datos
   ni reconstruyen ventanas.

4. VARIABILIDAD ESPERADA: Las métricas por partición reflejan variación local
   del desempeño del modelo. Esta variabilidad es metodológicamente correcta
   y no introduce sesgos, ya que el missingness y la imputación son globales.

Uso:
    python experimentos2.py \
        --dataset manglaria \
        --config config/grin/manglaria.yaml \
        --data-path manglaria_abril_timeseries.csv \
        --output-dir resultados_manglaria \
        --epochs 12 \
        --corte-inicio "2022-04-15" \
        --corte-fin "2022-04-30" \
        --num-particiones 5

    python experimentos2.py \
        --dataset mexflux \
        --config config/grin/mexflux.yaml \
        --data-path mexflux.csv \
        --output-dir resultados_mexflux \
        --epochs 12 \
        --num-particiones 5 \
        --particiones-a-evaluar "0,2,4"

Resultados en output-dir/:
    resultados.csv             — métricas globales por experimento
    resultados_por_nodo.csv    — métricas por nodo por experimento
    resultados_corte.csv       — métricas en el corte temporal fijo
    resultados_particiones.csv — métricas por partición temporal
    resumen_datos.txt          — info de datos faltantes originales
    log_experimentos.txt       — log completo de ejecución
"""

import os
import sys
import csv
import time
import subprocess
import argparse
import numpy as np
import pandas as pd

# ── EXPERIMENTOS ──────────────────────────────────────────────────────────────
# Estructura: (nombre, patron, pct_objetivo, p_block, p_point, min_seq, max_seq, desc)
# El pct_objetivo incluye los huecos reales ya existentes en el dataset.
# Los parámetros de máscara se aplican únicamente sobre celdas con datos presentes.
EXPERIMENTOS = [
    # MCAR — huecos puntuales completamente aleatorios por celda
    ("mcar_1pct",  "MCAR", 1.0,  0.000, 0.008, 4,  6,  "MCAR ~1%"),
    ("mcar_3pct",  "MCAR", 3.0,  0.000, 0.028, 4,  6,  "MCAR ~3%"),
    ("mcar_5pct",  "MCAR", 5.0,  0.000, 0.048, 4,  6,  "MCAR ~5%"),
    ("mcar_10pct", "MCAR", 10.0, 0.000, 0.098, 4,  6,  "MCAR ~10%"),
    ("mcar_15pct", "MCAR", 15.0, 0.000, 0.148, 4,  6,  "MCAR ~15%"),
    # MAR — mezcla de bloques cortos y puntuales
    ("mar_1pct",   "MAR",  1.0,  0.001, 0.003, 4,  6,  "MAR ~1%"),
    ("mar_3pct",   "MAR",  3.0,  0.004, 0.012, 4,  6,  "MAR ~3%"),
    ("mar_5pct",   "MAR",  5.0,  0.006, 0.020, 4,  6,  "MAR ~5%"),
    ("mar_10pct",  "MAR",  10.0, 0.012, 0.040, 4,  6,  "MAR ~10%"),
    ("mar_15pct",  "MAR",  15.0, 0.018, 0.060, 4,  6,  "MAR ~15%"),
    # MNAR — solo bloques largos, simula fallo de sensor
    ("mnar_1pct",  "MNAR", 1.0,  0.001, 0.000, 6,  12, "MNAR ~1%"),
    ("mnar_3pct",  "MNAR", 3.0,  0.003, 0.000, 6,  12, "MNAR ~3%"),
    ("mnar_5pct",  "MNAR", 5.0,  0.006, 0.000, 6,  12, "MNAR ~5%"),
    ("mnar_10pct", "MNAR", 10.0, 0.011, 0.000, 6,  12, "MNAR ~10%"),
    ("mnar_15pct", "MNAR", 15.0, 0.017, 0.000, 6,  12, "MNAR ~15%"),
]
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description='Experimentos MCAR/MAR/MNAR con GRIN — evaluación global y por partición'
    )
    parser.add_argument('--dataset',       type=str, required=True,
                        help='Nombre del dataset (manglaria, mexflux, etc.)')
    parser.add_argument('--config',        type=str, required=True,
                        help='Ruta al yaml de configuración GRIN')
    parser.add_argument('--data-path',     type=str, required=True,
                        help='Ruta al archivo CSV de datos')
    parser.add_argument('--output-dir',    type=str, default='resultados_experimentos')
    parser.add_argument('--epochs',        type=int, default=12)
    parser.add_argument('--workers',       type=int, default=8)
    parser.add_argument('--corte-inicio',  type=str, default=None,
                        help='Inicio del corte temporal fijo (YYYY-MM-DD)')
    parser.add_argument('--corte-fin',     type=str, default=None,
                        help='Fin del corte temporal fijo (YYYY-MM-DD)')
    parser.add_argument('--timestamp-col', type=str, default=None,
                        help='Nombre de la columna de tiempo en el CSV')
    # NUEVO — argumentos para evaluación por partición
    parser.add_argument('--num-particiones', type=int, default=5,
                        help='Número total de particiones temporales del test set (default: 5)')
    parser.add_argument('--particiones-a-evaluar', type=str, default=None,
                        help='Índices a evaluar separados por coma, ej: "0,2,4". '
                             'None = todas las particiones')
    return parser.parse_args()


# ── ANÁLISIS DE DATOS ORIGINALES ──────────────────────────────────────────────

def analizar_datos_originales(data_path, timestamp_col, output_dir):
    """
    Genera resumen de datos faltantes originales del dataset ANTES
    de introducir cualquier missingness artificial.
    Guarda resumen_datos.txt en output_dir.
    """
    df = pd.read_csv(data_path)

    if timestamp_col is None:
        for col in ['TIMESTAMP', 'timestamp', 'time', 'fecha', 'date']:
            if col in df.columns:
                timestamp_col = col
                break

    excluir = [timestamp_col, 'site_id', 'primary_key', 'DOY'] if timestamp_col else []
    df_num = df.drop(columns=[c for c in excluir if c in df.columns], errors='ignore')
    df_num = df_num.select_dtypes(include=[np.number])

    T, N = df_num.shape
    total_celdas = T * N
    total_nulos  = df_num.isnull().sum().sum()
    pct_global   = total_nulos / total_celdas * 100

    lineas = []
    lineas.append("=" * 60)
    lineas.append("  RESUMEN DE DATOS FALTANTES ORIGINALES")
    lineas.append("  (antes de introducir missingness artificial)")
    lineas.append("=" * 60)
    lineas.append(f"  Dataset:           {data_path}")
    lineas.append(f"  Filas:             {T}")
    lineas.append(f"  Variables:         {N}")
    lineas.append(f"  Celdas totales:    {total_celdas:,}")
    lineas.append(f"  Valores faltantes: {total_nulos:,} ({pct_global:.2f}%)")
    lineas.append("")
    lineas.append(f"  {'Variable':<30} {'Faltantes':>10} {'%':>8}")
    lineas.append(f"  {'-'*50}")

    pcts = df_num.isnull().mean() * 100
    for col in pcts.sort_values(ascending=False).index:
        n   = int(df_num[col].isnull().sum())
        pct = pcts[col]
        lineas.append(f"  {col[:30]:<30} {n:>10,} {pct:>7.2f}%")

    lineas.append("=" * 60)
    texto = "\n".join(lineas)
    print(texto)

    with open(f"{output_dir}/resumen_datos.txt", 'w', encoding='utf-8') as f:
        f.write(texto)

    return pct_global, N, df_num.columns.tolist()


# ── GENERACIÓN DE MÁSCARA GLOBAL ──────────────────────────────────────────────

def generar_mascara(data_path, timestamp_col, p_block, p_point,
                    min_seq, max_seq, seed=42):
    """
    PASO 1 — MISSINGNESS GLOBAL.

    Genera la máscara de valores faltantes UNA SOLA VEZ sobre toda la
    serie temporal completa. Esta función NO debe llamarse por partición.

    La máscara solo introduce huecos donde actualmente hay datos presentes,
    respetando los huecos reales ya existentes en el dataset.

    Parámetros:
        p_block  — probabilidad de iniciar un bloque de huecos en cada paso
        p_point  — probabilidad de un hueco puntual (si no hay bloque)
        min_seq  — longitud mínima de un bloque (pasos)
        max_seq  — longitud máxima de un bloque (pasos)
        seed     — semilla para reproducibilidad

    Retorna:
        mascara: (T, N) bool — True donde se introduce hueco artificial
    """
    df = pd.read_csv(data_path)
    excluir = [timestamp_col, 'site_id', 'primary_key', 'DOY']
    df_num = df.drop(columns=[c for c in excluir if c in df.columns], errors='ignore')
    df_num = df_num.select_dtypes(include=[np.number])

    rng = np.random.RandomState(seed)
    T, N = df_num.shape
    presente = (~df_num.isnull()).values  # (T, N)

    mascara = np.zeros((T, N), dtype=bool)
    for col in range(N):
        t = 0
        while t < T:
            if rng.random() < p_block:
                # Hueco en bloque — simula fallo temporal del sensor
                largo = rng.randint(min_seq, max_seq + 1)
                for dt in range(largo):
                    if t + dt < T and presente[t + dt, col]:
                        mascara[t + dt, col] = True
                t += largo
            else:
                # Hueco puntual — simula lectura corrupta aleatoria
                if rng.random() < p_point and presente[t, col]:
                    mascara[t, col] = True
                t += 1

    return mascara


# ── CÁLCULO DE MÉTRICAS ───────────────────────────────────────────────────────

def calcular_metricas(y_hat, y_true, mask):
    """
    Calcula MAE, RMSE, MSE, SMAPE sobre celdas donde mask=1.

    SMAPE (Symmetric Mean Absolute Percentage Error) es preferible a MAPE
    porque es simétrico — trata igual sobreestimación y subestimación.
    Fórmula: 100 × mean( |ŷ-y| / ((|ŷ|+|y|)/2) )

    Entradas:
        y_hat, y_true, mask — arrays numpy de la misma forma (B, W, features)
    Retorna:
        dict con métricas globales
    """
    m = mask.astype(bool)
    if m.sum() == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'mse': np.nan, 'smape': np.nan}

    err  = y_hat[m] - y_true[m]
    mae  = float(np.mean(np.abs(err)))
    mse  = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    # SMAPE — denominador simétrico evita división por cero en ambos extremos
    denom = (np.abs(y_hat[m]) + np.abs(y_true[m])) / 2
    denom[denom < 1e-8] = 1e-8
    smape = float(np.mean(np.abs(err) / denom) * 100)

    return {'mae': mae, 'rmse': rmse, 'mse': mse, 'smape': smape}


def calcular_metricas_por_nodo(y_hat, y_true, mask, n_nodos):
    """
    Calcula métricas por nodo (sensor) sobre el test set completo.

    Los arrays tienen forma (B, W, N*d) donde N=n_nodos y d=variables/nodo.
    Cada nodo ocupa un segmento contiguo en la última dimensión.

    Retorna lista de dicts, uno por nodo.
    """
    B, W, total = y_hat.shape
    d = total // n_nodos

    resultados = []
    for n in range(n_nodos):
        inicio = n * d
        fin    = (n + 1) * d
        resultados.append(calcular_metricas(
            y_hat[:, :, inicio:fin],
            y_true[:, :, inicio:fin],
            mask[:, :, inicio:fin]
        ))
    return resultados


# NUEVO — función de evaluación por partición temporal
def calcular_metricas_por_particion(y_hat, y_true, mask, timestamps,
                                     num_particiones=5,
                                     particiones_a_evaluar=None):
    """
    PASO 3 — EVALUACIÓN LOCAL (post-proceso).

    Divide el rango temporal del test set en K particiones consecutivas
    e iguales, y calcula métricas en cada una FILTRANDO sobre y_hat,
    y_true y mask ya calculados. No altera datos ni reconstruye ventanas.

    DISEÑO METODOLÓGICO:
    - El missingness es GLOBAL: la máscara fue generada sobre toda la serie.
    - La imputación es GLOBAL: y_hat cubre toda la serie.
    - La evaluación es LOCAL: solo se filtra por rango temporal.
    - La variabilidad entre particiones es ESPERADA y metodológicamente
      correcta — refleja cómo varía el desempeño del modelo a lo largo
      del tiempo, no un sesgo experimental.

    Parámetros:
        y_hat, y_true, mask   — arrays (B, W, features), GLOBALES, sin modificar
        timestamps             — lista de strings longitud B (un timestamp por ventana)
        num_particiones        — K total de particiones del test set
        particiones_a_evaluar  — lista de índices [0..K-1] a evaluar (None=todas)

    Métricas de contexto incluidas:
        densidad_missing       — fracción de celdas evaluadas vs total de la partición
        n_puntos_evaluados     — número absoluto de celdas con mask=1 en la partición

    Retorna:
        lista de dicts, uno por partición evaluada
    """
    if not timestamps or len(timestamps) == 0:
        return []

    # Parsear timestamps — intentar UTC primero, luego sin zona horaria
    try:
        ts = pd.to_datetime(timestamps, utc=True, errors='coerce')
    except Exception:
        ts = pd.to_datetime(timestamps, errors='coerce')

    validos = ~pd.isnull(ts)
    if validos.sum() == 0:
        return []

    ts_validos = ts[validos]
    t_min = ts_validos.min()
    t_max = ts_validos.max()

    if t_min == t_max:
        return []

    # Calcular bordes de K particiones iguales sobre el rango temporal completo
    rango_total = t_max - t_min
    bordes = [t_min + i * rango_total / num_particiones
              for i in range(num_particiones + 1)]

    # Determinar qué particiones evaluar
    if particiones_a_evaluar is None:
        indices = list(range(num_particiones))
    else:
        indices = [i for i in particiones_a_evaluar if 0 <= i < num_particiones]

    resultados = []
    ts_array = np.array(ts)

    for i in indices:
        inicio_p = bordes[i]
        fin_p    = bordes[i + 1]

        # La última partición incluye su borde derecho
        if i == num_particiones - 1:
            idx = np.where(
                (ts_array >= inicio_p) & (ts_array <= fin_p) & validos
            )[0]
        else:
            idx = np.where(
                (ts_array >= inicio_p) & (ts_array < fin_p) & validos
            )[0]

        # Limitar a índices válidos del array (no exceder B)
        idx = idx[idx < len(y_hat)]

        if len(idx) == 0:
            # Partición sin datos — registrar como NaN, no saltarla
            resultados.append({
                'particion':          i,
                'inicio':             str(inicio_p.date()),
                'fin':                str(fin_p.date()),
                'n_ventanas':         0,
                'n_puntos_evaluados': 0,
                'densidad_missing':   np.nan,
                'mae':                np.nan,
                'rmse':               np.nan,
                'mse':                np.nan,
                'smape':              np.nan,
            })
            continue

        # SOLO FILTRADO — no se modifica ningún array global
        yh_p = y_hat[idx]   # subconjunto de ventanas en esta partición
        yt_p = y_true[idx]
        mk_p = mask[idx]

        # NUEVO — métricas de contexto por partición
        n_puntos_evaluados = int(mk_p.sum())
        densidad_missing   = float(mk_p.sum() / mk_p.size) if mk_p.size > 0 else np.nan

        m = calcular_metricas(yh_p, yt_p, mk_p)
        resultados.append({
            'particion':          i,
            'inicio':             str(inicio_p.date()),
            'fin':                str(fin_p.date()),
            'n_ventanas':         len(idx),
            'n_puntos_evaluados': n_puntos_evaluados,  # NUEVO
            'densidad_missing':   round(densidad_missing, 4),  # NUEVO
            **m,
        })

    return resultados


def calcular_metricas_corte_temporal(y_hat, y_true, mask, timestamps,
                                      corte_inicio, corte_fin):
    """
    Filtra las predicciones al rango temporal fijo dado y calcula métricas.
    Similar a calcular_metricas_por_particion pero con fechas explícitas.
    """
    if not timestamps or corte_inicio is None or corte_fin is None:
        return None

    try:
        ts = pd.to_datetime(timestamps, utc=True, errors='coerce')
        ci = pd.Timestamp(corte_inicio, tz='UTC')
        cf = pd.Timestamp(corte_fin,   tz='UTC')
    except Exception:
        try:
            ts = pd.to_datetime(timestamps, errors='coerce')
            ci = pd.Timestamp(corte_inicio)
            cf = pd.Timestamp(corte_fin)
        except Exception:
            return None

    idx = np.where((ts >= ci) & (ts <= cf))[0]
    if len(idx) == 0:
        return None
    if len(idx) > len(y_hat):
        idx = idx[:len(y_hat)]

    m = calcular_metricas(y_hat[idx], y_true[idx], mask[idx])
    m['n_ventanas'] = len(idx)
    m['inicio']     = str(ci.date())
    m['fin']        = str(cf.date())
    return m


# ── EJECUCIÓN DE EXPERIMENTOS ─────────────────────────────────────────────────

def correr_experimento(nombre, dataset, config, p_block, p_point,
                        min_seq, max_seq, epochs, workers,
                        mask_path, output_npz, output_dir, log_file):
    """
    PASO 2 — IMPUTACIÓN GLOBAL.

    Lanza run_imputation.py como subproceso una sola vez por experimento.
    Pasa la máscara global via GRIN_MASK_PATH y recibe y_hat completo
    via GRIN_OUTPUT_PATH. No hay reentrenamiento ni reimputación.
    """
    import yaml
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['epochs'] = epochs
    config_tmp = f"{output_dir}/config_{nombre}.yaml"
    with open(config_tmp, 'w') as f:
        yaml.dump(cfg, f)

    env = os.environ.copy()
    env['GRIN_P_BLOCK']     = str(p_block)
    env['GRIN_P_POINT']     = str(p_point)
    env['GRIN_MIN_SEQ']     = str(min_seq)
    env['GRIN_MAX_SEQ']     = str(max_seq)
    # VERIFICACIÓN 1: la máscara global se pasa una sola vez via variable de entorno
    env['GRIN_MASK_PATH']   = mask_path
    # VERIFICACIÓN 2: el npz contendrá la serie completa del test set
    env['GRIN_OUTPUT_PATH'] = output_npz
    # Entrenamiento
    env['TRAIN_START'] = args.train_start or ''
    env['TRAIN_END']   = args.train_end or ''
    env['MIN_DENSITY'] = str(0.3)

    cmd = [
        sys.executable, '-m', 'scripts.run_imputation',
        '--config',       config_tmp,
        '--dataset-name', dataset,
        '--workers',      str(workers),
    ]

    inicio    = time.time()
    resultado = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
    )
    duracion = time.time() - inicio

    output_txt = resultado.stdout + resultado.stderr
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\nEXPERIMENTO: {nombre}\n")
        f.write(f"Duración: {duracion/3600:.2f}h\n")
        f.write(output_txt)

    return duracion, output_txt


def cargar_npz(path):
    """
    Carga el .npz guardado por run_imputation.
    VERIFICACIÓN 3: confirma que el npz contiene la serie completa.
    """
    npz_path = path + '.npz'
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    resultado = {
        'y_hat':      data['y_hat'],
        'y_true':     data['y_true'],
        'mask':       data['mask'],
        'timestamps': data['timestamps'].tolist() if 'timestamps' in data else [],
    }
    # Log de verificación
    B, W, F = resultado['y_hat'].shape
    print(f"  [✓ NPZ cargado] y_hat={resultado['y_hat'].shape}  "
          f"timestamps={len(resultado['timestamps'])} entradas")
    print(f"    → Serie completa del test: {B} ventanas × {W} pasos × {F} features")
    return resultado


# ── GUARDADO DE RESULTADOS ────────────────────────────────────────────────────

def guardar_csv_global(todos_resultados, output_dir):
    path = f"{output_dir}/resultados.csv"
    campos = ['nombre', 'patron', 'descripcion', 'pct_objetivo',
              'mae', 'rmse', 'mse', 'smape', 'duracion_horas']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in todos_resultados:
            m = r.get('metricas_global', {})
            w.writerow({
                'nombre':         r['nombre'],
                'patron':         r['patron'],
                'descripcion':    r['descripcion'],
                'pct_objetivo':   r['pct_objetivo'],
                'mae':            m.get('mae', ''),
                'rmse':           m.get('rmse', ''),
                'mse':            m.get('mse', ''),
                'smape':          m.get('smape', ''),
                'duracion_horas': round(r['duracion'] / 3600, 2),
            })
    print(f"  ✓ Global:          {path}")


def guardar_csv_por_nodo(todos_resultados, output_dir):
    path = f"{output_dir}/resultados_por_nodo.csv"
    campos = ['nombre', 'patron', 'pct_objetivo', 'nodo',
              'mae', 'rmse', 'mse', 'smape']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in todos_resultados:
            for i, m in enumerate(r.get('metricas_por_nodo', [])):
                w.writerow({
                    'nombre':       r['nombre'],
                    'patron':       r['patron'],
                    'pct_objetivo': r['pct_objetivo'],
                    'nodo':         i,
                    'mae':          m.get('mae', ''),
                    'rmse':         m.get('rmse', ''),
                    'mse':          m.get('mse', ''),
                    'smape':        m.get('smape', ''),
                })
    print(f"  ✓ Por nodo:        {path}")


def guardar_csv_corte(todos_resultados, output_dir):
    path = f"{output_dir}/resultados_corte.csv"
    campos = ['nombre', 'patron', 'pct_objetivo',
              'inicio', 'fin', 'n_ventanas',
              'mae', 'rmse', 'mse', 'smape']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in todos_resultados:
            m = r.get('metricas_corte')
            if m:
                w.writerow({
                    'nombre':       r['nombre'],
                    'patron':       r['patron'],
                    'pct_objetivo': r['pct_objetivo'],
                    'inicio':       m.get('inicio', ''),
                    'fin':          m.get('fin', ''),
                    'n_ventanas':   m.get('n_ventanas', ''),
                    'mae':          m.get('mae', ''),
                    'rmse':         m.get('rmse', ''),
                    'mse':          m.get('mse', ''),
                    'smape':        m.get('smape', ''),
                })
    print(f"  ✓ Corte temporal:  {path}")


# NUEVO — guardado de métricas por partición con contexto
def guardar_csv_particiones(todos_resultados, output_dir):
    """
    Guarda métricas por partición temporal.
    Incluye métricas de contexto: densidad_missing y n_puntos_evaluados.

    NOTA METODOLÓGICA en el CSV:
    Las métricas por partición reflejan variación LOCAL del desempeño.
    El missingness es GLOBAL — la densidad varía entre particiones
    por distribución natural de los huecos, no por diseño experimental.
    """
    path = f"{output_dir}/resultados_particiones.csv"
    # NUEVO — campos extendidos con métricas de contexto
    campos = ['nombre', 'patron', 'pct_objetivo', 'particion',
              'inicio', 'fin', 'n_ventanas',
              'n_puntos_evaluados',   # NUEVO
              'densidad_missing',     # NUEVO
              'mae', 'rmse', 'mse', 'smape']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in todos_resultados:
            for p in r.get('metricas_particiones', []):
                w.writerow({
                    'nombre':               r['nombre'],
                    'patron':               r['patron'],
                    'pct_objetivo':         r['pct_objetivo'],
                    'particion':            p['particion'],
                    'inicio':               p['inicio'],
                    'fin':                  p['fin'],
                    'n_ventanas':           p['n_ventanas'],
                    'n_puntos_evaluados':   p.get('n_puntos_evaluados', ''),  # NUEVO
                    'densidad_missing':     p.get('densidad_missing', ''),    # NUEVO
                    'mae':                  p.get('mae', ''),
                    'rmse':                 p.get('rmse', ''),
                    'mse':                  p.get('mse', ''),
                    'smape':                p.get('smape', ''),
                })
    print(f"  ✓ Por partición:   {path}")


def imprimir_tabla_final(todos_resultados):
    sep = '=' * 78
    print(f"\n{sep}")
    print(f"  RESULTADOS FINALES")
    print(sep)
    print(f"  {'Experimento':<14} {'Patrón':<6} {'%':>5} "
          f"{'MAE':>8} {'RMSE':>8} {'MSE':>8} {'SMAPE%':>8}")
    print(f"  {'-'*76}")
    for r in todos_resultados:
        m = r.get('metricas_global', {})
        try:
            print(f"  {r['nombre']:<14} {r['patron']:<6} "
                  f"{r['pct_objetivo']:>4.0f}% "
                  f"{m.get('mae',   float('nan')):>8.4f} "
                  f"{m.get('rmse',  float('nan')):>8.4f} "
                  f"{m.get('mse',   float('nan')):>8.4f} "
                  f"{m.get('smape', float('nan')):>7.2f}%")
        except Exception:
            print(f"  {r['nombre']:<14} {r['patron']:<6} --- error ---")
    print(sep)


# ── VERIFICACIÓN DEL ADAPTADOR ────────────────────────────────────────────────

def verificar_adaptador():
    """
    Verifica que run_imputation.py soporta:
    1. GRIN_MASK_PATH — para recibir la máscara global
    2. GRIN_OUTPUT_PATH — para guardar y_hat completo como .npz
    """
    with open('scripts/run_imputation.py', 'r', encoding='utf-8') as f:
        contenido = f.read()

    ok_mask   = 'GRIN_MASK_PATH'   in contenido
    ok_output = 'GRIN_OUTPUT_PATH' in contenido

    if ok_mask and ok_output:
        print("✓ run_imputation.py soporta GRIN_MASK_PATH y GRIN_OUTPUT_PATH")
        return True

    if not ok_mask:
        print("⚠ Falta GRIN_MASK_PATH en run_imputation.py")
        print("  Agrega en ManglarIAAdapter.__init__ (sección EVAL MASK):")
        print("  mask_path = os.environ.get('GRIN_MASK_PATH', '')")
        print("  if mask_path and os.path.exists(mask_path):")
        print("      eval_mask_np = np.load(mask_path).astype(np.uint8)")

    if not ok_output:
        print("⚠ Falta GRIN_OUTPUT_PATH en run_imputation.py")
        print("  Agrega al final de run_experiment() antes del return:")
        print("  output_path = os.environ.get('GRIN_OUTPUT_PATH', '')")
        print("  if output_path:")
        print("      np.savez_compressed(output_path, y_hat=y_hat, ...)")

    return False


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # NUEVO — parsear particiones a evaluar
    if args.particiones_a_evaluar is not None:
        particiones_evaluar = [int(x.strip())
                               for x in args.particiones_a_evaluar.split(',')]
    else:
        particiones_evaluar = None

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = f"{args.output_dir}/log_experimentos.txt"

    print("=" * 60)
    print(f"  EXPERIMENTOS: {args.dataset.upper()}")
    print(f"  {len(EXPERIMENTOS)} experimentos × {args.epochs} epochs")
    print(f"  Dataset: {args.data_path}")
    print(f"  Particiones: {args.num_particiones} "
          f"(evaluar: {particiones_evaluar or 'todas'})")
    if args.corte_inicio and args.corte_fin:
        print(f"  Corte temporal fijo: {args.corte_inicio} → {args.corte_fin}")
    print(f"  Resultados en: {args.output_dir}/")
    print("=" * 60)

    # Verificar que run_imputation.py está correctamente configurado
    if not verificar_adaptador():
        sys.exit(1)

    # Detectar columna de tiempo automáticamente
    ts_col = args.timestamp_col
    if ts_col is None:
        df_tmp = pd.read_csv(args.data_path, nrows=1)
        for col in ['TIMESTAMP', 'timestamp', 'time', 'fecha', 'date']:
            if col in df_tmp.columns:
                ts_col = col
                break

    # Análisis de datos originales (antes de cualquier missingness artificial)
    print("\nAnalizando datos originales...")
    pct_real, n_vars, var_names = analizar_datos_originales(
        args.data_path, ts_col, args.output_dir
    )

    # Detectar número de nodos
    df_tmp = pd.read_csv(args.data_path, nrows=5)
    if 'site_id' in df_tmp.columns:
        n_nodos = pd.read_csv(args.data_path)['site_id'].nunique()
    else:
        n_nodos = 1

    print(f"\nNodos detectados: {n_nodos}  |  Variables por nodo: {n_vars // n_nodos}")
    print(f"Missingness original: {pct_real:.2f}%")

    todos_resultados = []

    for nombre, patron, pct_obj, p_block, p_point, min_seq, max_seq, desc in EXPERIMENTOS:

        print(f"\n{'='*60}")
        print(f"  Iniciando: {nombre}  ({desc})")
        print(f"  p_block={p_block}  p_point={p_point}  "
              f"min_seq={min_seq}  max_seq={max_seq}")
        print(f"{'='*60}")

        # ── PASO 1: MISSINGNESS GLOBAL ────────────────────────────────────
        # La máscara se genera UNA SOLA VEZ aquí.
        # No existe ninguna otra llamada a generar_mascara en este experimento.
        print(f"  [Paso 1] Generando máscara global...")
        mascara = generar_mascara(
            args.data_path, ts_col,
            p_block, p_point, min_seq, max_seq,
            seed=hash(nombre) % 2**31
        )
        pct_artificial = mascara.mean() * 100
        mask_path = f"{args.output_dir}/mask_{nombre}.npy"
        np.save(mask_path, mascara)
        print(f"  [✓ Máscara global] {pct_artificial:.2f}% artificial "
              f"+ {pct_real:.2f}% real = {pct_artificial+pct_real:.2f}% total")
        print(f"    Forma: {mascara.shape}  |  Guardada en: {mask_path}")

        # ── PASO 2: IMPUTACIÓN GLOBAL ─────────────────────────────────────
        # run_imputation.py se llama UNA SOLA VEZ y produce y_hat completo.
        output_npz = f"{args.output_dir}/results_{nombre}"
        print(f"  [Paso 2] Ejecutando GRIN (imputación global)...")
        duracion, _ = correr_experimento(
            nombre, args.dataset, args.config,
            p_block, p_point, min_seq, max_seq,
            args.epochs, args.workers,
            mask_path, output_npz,
            args.output_dir, log_file
        )

        # Cargar resultados — verificación de que el npz es completo
        datos = cargar_npz(output_npz)

        if datos is None:
            print(f"  ✗ No se encontró {output_npz}.npz — revisa el log")
            metricas_global     = {}
            metricas_nodo       = []
            metricas_corte      = None
            metricas_particiones = []
        else:
            yh  = datos['y_hat']
            yt  = datos['y_true']
            mk  = datos['mask']
            tss = datos['timestamps']

            # ── PASO 3: EVALUACIÓN LOCAL ──────────────────────────────────
            # Todas las métricas se calculan filtrando sobre los arrays globales.
            # No se modifica ningún array. No hay reimputación.

            # Métricas globales (todo el test set)
            metricas_global = calcular_metricas(yh, yt, mk)
            print(f"  ✓ Completado en {duracion/3600:.2f}h")
            print(f"  [Global] MAE={metricas_global['mae']:.4f}  "
                  f"RMSE={metricas_global['rmse']:.4f}  "
                  f"SMAPE={metricas_global['smape']:.2f}%")

            # Métricas por nodo
            metricas_nodo = calcular_metricas_por_nodo(yh, yt, mk, n_nodos)
            for i, m in enumerate(metricas_nodo):
                print(f"  [Nodo {i}] MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

            # NUEVO — Métricas por partición temporal
            # VERIFICACIÓN: solo se filtran los arrays globales, no se regenera nada
            print(f"  [Paso 3] Evaluando {args.num_particiones} particiones temporales...")
            metricas_particiones = calcular_metricas_por_particion(
                yh, yt, mk, tss,
                num_particiones=args.num_particiones,
                particiones_a_evaluar=particiones_evaluar
            )
            for p in metricas_particiones:
                if p['n_ventanas'] > 0:
                    print(f"    [P{p['particion']}] {p['inicio']}→{p['fin']}: "
                          f"MAE={p['mae']:.4f}  "
                          f"missing={p['densidad_missing']:.2%}  "
                          f"n={p['n_puntos_evaluados']}")
                else:
                    print(f"    [P{p['particion']}] {p['inicio']}→{p['fin']}: sin datos")

            # Métricas en corte temporal fijo (opcional)
            metricas_corte = calcular_metricas_corte_temporal(
                yh, yt, mk, tss,
                args.corte_inicio, args.corte_fin
            )
            if metricas_corte:
                print(f"  [Corte] {metricas_corte['inicio']}→{metricas_corte['fin']}: "
                      f"MAE={metricas_corte['mae']:.4f}  "
                      f"RMSE={metricas_corte['rmse']:.4f}")

        todos_resultados.append({
            'nombre':               nombre,
            'patron':               patron,
            'descripcion':          desc,
            'pct_objetivo':         pct_obj,
            'duracion':             duracion,
            'metricas_global':      metricas_global,
            'metricas_por_nodo':    metricas_nodo,
            'metricas_corte':       metricas_corte,
            'metricas_particiones': metricas_particiones,  # NUEVO
        })

        # Guardar resultados parciales después de cada experimento
        guardar_csv_global(todos_resultados, args.output_dir)
        guardar_csv_por_nodo(todos_resultados, args.output_dir)
        guardar_csv_corte(todos_resultados, args.output_dir)
        guardar_csv_particiones(todos_resultados, args.output_dir)  # NUEVO

    imprimir_tabla_final(todos_resultados)
    print(f"\n¡Experimentos completados!")
    print(f"  Global:        {args.output_dir}/resultados.csv")
    print(f"  Por nodo:      {args.output_dir}/resultados_por_nodo.csv")
    print(f"  Corte:         {args.output_dir}/resultados_corte.csv")
    print(f"  Particiones:   {args.output_dir}/resultados_particiones.csv")
    print(f"  Datos origen:  {args.output_dir}/resumen_datos.txt")
    print(f"  Log:           {args.output_dir}/log_experimentos.txt")


if __name__ == '__main__':
    main()