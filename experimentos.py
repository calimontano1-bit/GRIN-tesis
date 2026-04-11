"""
experimentos.py
==================
Script genérico de experimentos MCAR/MAR/MNAR para cualquier dataset
compatible con run_imputation.py.

Genera métricas globales, por nodo y por corte temporal.

Uso:
    python experimentos.py \
        --dataset manglaria \
        --config config/grin/manglaria.yaml \
        --data-path manglaria_abril_timeseries.csv \
        --output-dir resultados_manglaria_v2 \
        --epochs 12 \
        --corte-inicio "2022-04-15" \
        --corte-fin "2022-04-30"

    python experimentos.py \
        --dataset mexflux \
        --config config/grin/mexflux.yaml \
        --data-path mexflux.csv \
        --output-dir resultados_mexflux \
        --epochs 12 \
        --corte-inicio "2025-11-01" \
        --corte-fin "2025-11-30"

Resultados en output-dir/:
    resultados.csv          — tabla global por experimento
    resultados_por_nodo.csv — métricas por nodo por experimento
    resultados_corte.csv    — métricas en el corte temporal
    resumen_datos.txt       — info de datos faltantes originales
    log_experimentos.txt    — log completo
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
EXPERIMENTOS = [
    # (nombre, patron, pct_objetivo, p_block, p_point, min_seq, max_seq, desc)
    ("mcar_1pct",  "MCAR", 1.0,  0.000, 0.008, 4,  6,  "MCAR ~1%"),
    ("mcar_3pct",  "MCAR", 3.0,  0.000, 0.028, 4,  6,  "MCAR ~3%"),
    ("mcar_5pct",  "MCAR", 5.0,  0.000, 0.048, 4,  6,  "MCAR ~5%"),
    ("mcar_10pct", "MCAR", 10.0, 0.000, 0.098, 4,  6,  "MCAR ~10%"),
    ("mcar_15pct", "MCAR", 15.0, 0.000, 0.148, 4,  6,  "MCAR ~15%"),
    ("mar_1pct",   "MAR",  1.0,  0.001, 0.003, 4,  6,  "MAR ~1%"),
    ("mar_3pct",   "MAR",  3.0,  0.004, 0.012, 4,  6,  "MAR ~3%"),
    ("mar_5pct",   "MAR",  5.0,  0.006, 0.020, 4,  6,  "MAR ~5%"),
    ("mar_10pct",  "MAR",  10.0, 0.012, 0.040, 4,  6,  "MAR ~10%"),
    ("mar_15pct",  "MAR",  15.0, 0.018, 0.060, 4,  6,  "MAR ~15%"),
    ("mnar_1pct",  "MNAR", 1.0,  0.001, 0.000, 6,  12, "MNAR ~1%"),
    ("mnar_3pct",  "MNAR", 3.0,  0.003, 0.000, 6,  12, "MNAR ~3%"),
    ("mnar_5pct",  "MNAR", 5.0,  0.006, 0.000, 6,  12, "MNAR ~5%"),
    ("mnar_10pct", "MNAR", 10.0, 0.011, 0.000, 6,  12, "MNAR ~10%"),
    ("mnar_15pct", "MNAR", 15.0, 0.017, 0.000, 6,  12, "MNAR ~15%"),
]
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',       type=str, required=True,
                        help='Nombre del dataset (manglaria, mexflux, etc.)')
    parser.add_argument('--config',        type=str, required=True,
                        help='Ruta al yaml de configuración')
    parser.add_argument('--data-path',     type=str, required=True,
                        help='Ruta al archivo CSV de datos')
    parser.add_argument('--output-dir',    type=str, default='resultados_experimentos')
    parser.add_argument('--epochs',        type=int, default=12)
    parser.add_argument('--workers',       type=int, default=8)
    parser.add_argument('--corte-inicio',  type=str, default=None,
                        help='Inicio del corte temporal (YYYY-MM-DD)')
    parser.add_argument('--corte-fin',     type=str, default=None,
                        help='Fin del corte temporal (YYYY-MM-DD)')
    parser.add_argument('--timestamp-col', type=str, default=None,
                        help='Nombre de la columna de tiempo en el CSV')
    parser.add_argument('--num-particiones', type=int, default=5,
                        help='Número total de particiones temporales (default: 5)')
    parser.add_argument('--particiones-a-evaluar', type=str, default=None,
                        help='Índices a evaluar separados por coma, ej: "0,2,4". '
                            'None = todas')
    return parser.parse_args()


# ── ANÁLISIS DE DATOS ORIGINALES ──────────────────────────────────────────────

def analizar_datos_originales(data_path, timestamp_col, output_dir):
    """
    Genera resumen de datos faltantes originales del dataset.
    Guarda resumen_datos.txt en output_dir.
    """
    df = pd.read_csv(data_path)

    # Detectar columna de tiempo automáticamente si no se especifica
    if timestamp_col is None:
        for col in ['TIMESTAMP', 'timestamp', 'time', 'fecha', 'date']:
            if col in df.columns:
                timestamp_col = col
                break

    # Excluir columnas no numéricas para el análisis
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
    lineas.append("=" * 60)
    lineas.append(f"  Dataset:          {data_path}")
    lineas.append(f"  Filas:            {T}")
    lineas.append(f"  Variables:        {N}")
    lineas.append(f"  Celdas totales:   {total_celdas:,}")
    lineas.append(f"  Valores faltantes:{total_nulos:,} ({pct_global:.2f}%)")
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


# ── GENERACIÓN DE MÁSCARAS ────────────────────────────────────────────────────

def generar_mascara(data_path, timestamp_col, p_block, p_point,
                    min_seq, max_seq, seed=42):
    """
    Genera máscara artificial (T, N) sobre celdas actualmente presentes.
    Compatible con cualquier CSV — excluye columnas no numéricas.
    """
    df = pd.read_csv(data_path)
    excluir = [timestamp_col, 'site_id', 'primary_key', 'DOY']
    df_num = df.drop(columns=[c for c in excluir if c in df.columns], errors='ignore')
    df_num = df_num.select_dtypes(include=[np.number])

    rng = np.random.RandomState(seed)
    T, N = df_num.shape
    presente = (~df_num.isnull()).values

    mascara = np.zeros((T, N), dtype=bool)
    for col in range(N):
        t = 0
        while t < T:
            if rng.random() < p_block:
                largo = rng.randint(min_seq, max_seq + 1)
                for dt in range(largo):
                    if t + dt < T and presente[t + dt, col]:
                        mascara[t + dt, col] = True
                t += largo
            else:
                if rng.random() < p_point and presente[t, col]:
                    mascara[t, col] = True
                t += 1
    return mascara


# ── CÁLCULO DE MÉTRICAS ───────────────────────────────────────────────────────

def calcular_metricas(y_hat, y_true, mask):
    """
    Calcula MAE, RMSE, MSE, MAPE sobre celdas donde mask=1.
    
    Entradas:
        y_hat, y_true, mask — arrays numpy de la misma forma
    Retorna:
        dict con métricas globales
    """
    m = mask.astype(bool)
    if m.sum() == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'mse': np.nan, 'mape': np.nan}

    err  = y_hat[m] - y_true[m]
    mae  = float(np.mean(np.abs(err)))
    mse  = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    # Después — SMAPE:
    denom = (np.abs(y_hat[m]) + np.abs(y_true[m])) / 2
    denom[denom < 1e-8] = 1e-8
    smape = float(np.mean(np.abs(err) / denom) * 100)

    return {'mae': mae, 'rmse': rmse, 'mse': mse, 'smape': smape}


def calcular_metricas_por_nodo(y_hat, y_true, mask, n_nodos):
    """
    Calcula métricas por nodo (sensor).
    
    y_hat, y_true, mask tienen forma (B, W, N*d) o (B, W, N)
    donde N es n_nodos.
    
    Retorna lista de dicts, uno por nodo.
    """
    B, W, total = y_hat.shape
    d = total // n_nodos  # variables por nodo

    resultados = []
    for n in range(n_nodos):
        inicio = n * d
        fin    = (n + 1) * d
        yh_n   = y_hat[:, :, inicio:fin]
        yt_n   = y_true[:, :, inicio:fin]
        mk_n   = mask[:, :, inicio:fin]
        resultados.append(calcular_metricas(yh_n, yt_n, mk_n))

    return resultados

def calcular_metricas_por_particion(y_hat, y_true, mask, timestamps,
                                     num_particiones=5,
                                     particiones_a_evaluar=None):
    """
    Divide el test set en K particiones temporales consecutivas
    y calcula métricas en cada una.

    Parámetros:
        y_hat, y_true, mask  — arrays (B, W, features)
        timestamps           — lista de strings, longitud B
        num_particiones      — K total de particiones
        particiones_a_evaluar — lista de índices a evaluar (None = todas)

    Retorna:
        lista de dicts, uno por partición evaluada
    """
    if not timestamps or len(timestamps) == 0:
        return []

    # Parsear timestamps — intentar con UTC primero, luego sin zona horaria
    try:
        ts = pd.to_datetime(timestamps, utc=True, errors='coerce')
    except Exception:
        ts = pd.to_datetime(timestamps, errors='coerce')

    # Eliminar NaT
    validos = ~pd.isnull(ts)
    if validos.sum() == 0:
        return []

    ts_validos = ts[validos]
    t_min = ts_validos.min()
    t_max = ts_validos.max()

    if t_min == t_max:
        return []

    # Calcular bordes de particiones
    rango_total = t_max - t_min
    bordes = [t_min + i * rango_total / num_particiones
              for i in range(num_particiones + 1)]

    # Qué particiones evaluar
    if particiones_a_evaluar is None:
        indices = list(range(num_particiones))
    else:
        indices = [i for i in particiones_a_evaluar if 0 <= i < num_particiones]

    resultados = []
    ts_array = np.array(ts)

    for i in indices:
        inicio = bordes[i]
        fin    = bordes[i + 1]

        # Última partición incluye el borde derecho
        if i == num_particiones - 1:
            idx = np.where(
                (ts_array >= inicio) & (ts_array <= fin) & validos
            )[0]
        else:
            idx = np.where(
                (ts_array >= inicio) & (ts_array < fin) & validos
            )[0]

        # Filtrar a índices válidos para el array
        idx = idx[idx < len(y_hat)]

        if len(idx) == 0:
            resultados.append({
                'particion':    i,
                'inicio':       str(inicio.date()),
                'fin':          str(fin.date()),
                'n_ventanas':   0,
                'mae':          np.nan,
                'rmse':         np.nan,
                'mse':          np.nan,
                'mape':         np.nan,
            })
            continue

        yh_p = y_hat[idx]
        yt_p = y_true[idx]
        mk_p = mask[idx]

        m = calcular_metricas(yh_p, yt_p, mk_p)
        resultados.append({
            'particion':  i,
            'inicio':     str(inicio.date()),
            'fin':        str(fin.date()),
            'n_ventanas': len(idx),
            **m,
        })

    return resultados

def calcular_metricas_corte_temporal(y_hat, y_true, mask, timestamps,
                                      corte_inicio, corte_fin):
    """
    Filtra las predicciones al rango temporal dado y calcula métricas.
    timestamps: lista de strings con fechas del test set
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

    # timestamps corresponde a ventanas — cada elemento es un paso de tiempo
    # Tomamos las ventanas que caen en el rango
    # y_hat tiene forma (B, W, features) — B=ventanas
    # Para el corte usamos las ventanas cuyos índices caen en el rango
    if len(idx) > len(y_hat):
        idx = idx[:len(y_hat)]

    yh_c  = y_hat[idx]
    yt_c  = y_true[idx]
    mk_c  = mask[idx]

    m = calcular_metricas(yh_c, yt_c, mk_c)
    m['n_ventanas'] = len(idx)
    m['inicio']     = str(ci.date())
    m['fin']        = str(cf.date())
    return m


# ── EJECUCIÓN DE EXPERIMENTOS ─────────────────────────────────────────────────

def correr_experimento(nombre, dataset, config, p_block, p_point,
                        min_seq, max_seq, epochs, workers,
                        mask_path, output_npz, output_dir, log_file):
    """Lanza run_imputation.py como subproceso y devuelve el .npz."""

    # Config temporal con epochs correctos
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
    env['GRIN_MASK_PATH']   = mask_path
    env['GRIN_OUTPUT_PATH'] = output_npz  # sin .npz — numpy lo agrega

    cmd = [
        sys.executable, '-m', 'scripts.run_imputation',
        '--config',       config_tmp,
        '--dataset-name', dataset,
        '--workers',      str(workers),
    ]

    inicio   = time.time()
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
    """Carga el .npz guardado por run_imputation."""
    npz_path = path + '.npz'
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    return {
        'y_hat':      data['y_hat'],
        'y_true':     data['y_true'],
        'mask':       data['mask'],
        'timestamps': data['timestamps'].tolist() if 'timestamps' in data else [],
    }


# ── GUARDADO DE RESULTADOS ────────────────────────────────────────────────────

def guardar_csv_global(todos_resultados, output_dir):
    path = f"{output_dir}/resultados.csv"
    campos = ['nombre', 'patron', 'descripcion', 'pct_objetivo',
              'mae', 'rmse', 'mse', 'mape', 'duracion_horas']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in todos_resultados:
            m = r.get('metricas_global', {})
            w.writerow({
                'nombre':        r['nombre'],
                'patron':        r['patron'],
                'descripcion':   r['descripcion'],
                'pct_objetivo':  r['pct_objetivo'],
                'mae':           m.get('mae', ''),
                'rmse':          m.get('rmse', ''),
                'mse':           m.get('mse', ''),
                'mape':          m.get('mape', ''),
                'duracion_horas': round(r['duracion'] / 3600, 2),
            })
    print(f"  ✓ Global: {path}")


def guardar_csv_por_nodo(todos_resultados, output_dir):
    path = f"{output_dir}/resultados_por_nodo.csv"
    campos = ['nombre', 'patron', 'pct_objetivo', 'nodo',
              'mae', 'rmse', 'mse', 'mape']
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
                    'mape':         m.get('mape', ''),
                })
    print(f"  ✓ Por nodo: {path}")


def guardar_csv_corte(todos_resultados, output_dir):
    path = f"{output_dir}/resultados_corte.csv"
    campos = ['nombre', 'patron', 'pct_objetivo',
              'inicio', 'fin', 'n_ventanas',
              'mae', 'rmse', 'mse', 'mape']
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
                    'mape':         m.get('mape', ''),
                })
    print(f"  ✓ Corte temporal: {path}")

def guardar_csv_particiones(todos_resultados, output_dir):
    path = f"{output_dir}/resultados_particiones.csv"
    campos = ['nombre', 'patron', 'pct_objetivo', 'particion',
              'inicio', 'fin', 'n_ventanas',
              'mae', 'rmse', 'mse', 'mape']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in todos_resultados:
            for p in r.get('metricas_particiones', []):
                w.writerow({
                    'nombre':       r['nombre'],
                    'patron':       r['patron'],
                    'pct_objetivo': r['pct_objetivo'],
                    'particion':    p['particion'],
                    'inicio':       p['inicio'],
                    'fin':          p['fin'],
                    'n_ventanas':   p['n_ventanas'],
                    'mae':          p.get('mae', ''),
                    'rmse':         p.get('rmse', ''),
                    'mse':          p.get('mse', ''),
                    'mape':         p.get('mape', ''),
                })
    print(f"  ✓ Por partición: {path}")

def imprimir_tabla_final(todos_resultados):
    sep = '=' * 75
    print(f"\n{sep}")
    print(f"  RESULTADOS FINALES")
    print(sep)
    print(f"  {'Experimento':<14} {'Patrón':<6} {'%':>5} "
          f"{'MAE':>8} {'RMSE':>8} {'MSE':>8} {'MAPE%':>8}")
    print(f"  {'-'*73}")
    for r in todos_resultados:
        m = r.get('metricas_global', {})
        try:
            print(f"  {r['nombre']:<14} {r['patron']:<6} "
                  f"{r['pct_objetivo']:>4.0f}% "
                  f"{m.get('mae', float('nan')):>8.4f} "
                  f"{m.get('rmse', float('nan')):>8.4f} "
                  f"{m.get('mse', float('nan')):>8.4f} "
                  f"{m.get('mape', float('nan')):>7.2f}%")
        except Exception:
            print(f"  {r['nombre']:<14} {r['patron']:<6} --- error ---")
    print(sep)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    # Parsear particiones a evaluar
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
    if args.corte_inicio and args.corte_fin:
        print(f"  Corte temporal: {args.corte_inicio} → {args.corte_fin}")
    print(f"  Resultados en: {args.output_dir}/")
    print("=" * 60)

    # Detectar columna de tiempo
    ts_col = args.timestamp_col
    if ts_col is None:
        df_tmp = pd.read_csv(args.data_path, nrows=1)
        for col in ['TIMESTAMP', 'timestamp', 'time', 'fecha', 'date']:
            if col in df_tmp.columns:
                ts_col = col
                break

    # Análisis de datos originales
    print("\nAnalizando datos originales...")
    pct_real, n_vars, var_names = analizar_datos_originales(
        args.data_path, ts_col, args.output_dir
    )

    # Detectar número de nodos
    # Para datasets con site_id (múltiples sensores), N = n_sensores
    # Para datasets de un sensor, N = 1
    df_tmp = pd.read_csv(args.data_path, nrows=5)
    if 'site_id' in df_tmp.columns:
        n_nodos = df_tmp['site_id'].nunique()
        if n_nodos == 0:
            n_nodos = pd.read_csv(args.data_path)['site_id'].nunique()
    else:
        n_nodos = 1

    print(f"\nNodos detectados: {n_nodos}")
    print(f"Variables por nodo: {n_vars // n_nodos}")

    todos_resultados = []

    for nombre, patron, pct_obj, p_block, p_point, min_seq, max_seq, desc in EXPERIMENTOS:

        print(f"\n{'='*60}")
        print(f"  Iniciando: {nombre}  ({desc})")
        print(f"  p_block={p_block}, p_point={p_point}")
        print(f"{'='*60}")

        # Generar y guardar máscara
        mascara = generar_mascara(
            args.data_path, ts_col,
            p_block, p_point, min_seq, max_seq,
            seed=hash(nombre) % 2**31
        )
        pct_artificial = mascara.mean() * 100
        mask_path = f"{args.output_dir}/mask_{nombre}.npy"
        np.save(mask_path, mascara)
        print(f"  Huecos artificiales: {pct_artificial:.2f}% "
              f"(+{pct_real:.2f}% reales = {pct_artificial+pct_real:.2f}% total)")

        # Ruta para guardar el npz de resultados
        output_npz = f"{args.output_dir}/results_{nombre}"

        # Correr experimento
        duracion, _ = correr_experimento(
            nombre, args.dataset, args.config,
            p_block, p_point, min_seq, max_seq,
            args.epochs, args.workers,
            mask_path, output_npz,
            args.output_dir, log_file
        )

        # Cargar resultados del npz
        datos = cargar_npz(output_npz)

        if datos is None:
            print(f"  ✗ No se encontró {output_npz}.npz — revisa el log")
            metricas_global  = {}
            metricas_nodo    = []
            metricas_corte   = None
            metricas_particiones = []  # ← agregar esta línea
        else:
            yh  = datos['y_hat']
            yt  = datos['y_true']
            mk  = datos['mask']
            tss = datos['timestamps']

            # Métricas globales
            metricas_global = calcular_metricas(yh, yt, mk)
            print(f"  ✓ Completado en {duracion/3600:.2f}h")
            print(f"  MAE={metricas_global['mae']:.4f}  "
                  f"RMSE={metricas_global['rmse']:.4f}  "
                  f"MSE={metricas_global['mse']:.4f}")

            # Métricas por nodo
            metricas_nodo = calcular_metricas_por_nodo(yh, yt, mk, n_nodos)
            for i, m in enumerate(metricas_nodo):
                print(f"    Nodo {i}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

            # Métricas por partición temporal
            metricas_particiones = calcular_metricas_por_particion(
                yh, yt, mk, tss,
                num_particiones=args.num_particiones,
                particiones_a_evaluar=particiones_evaluar
            )
            if metricas_particiones:
                print(f"  Particiones evaluadas: {len(metricas_particiones)}")
                for p in metricas_particiones:
                    if p['n_ventanas'] > 0:
                        print(f"    [{p['particion']}] {p['inicio']}→{p['fin']}: "
                            f"MAE={p['mae']:.4f}  n={p['n_ventanas']}")
                    else:
                        print(f"    [{p['particion']}] {p['inicio']}→{p['fin']}: sin datos")    

            # Métricas en corte temporal
            metricas_corte = calcular_metricas_corte_temporal(
                yh, yt, mk, tss,
                args.corte_inicio, args.corte_fin
            )
            if metricas_corte:
                print(f"  Corte {metricas_corte['inicio']}→{metricas_corte['fin']}: "
                      f"MAE={metricas_corte['mae']:.4f}  "
                      f"RMSE={metricas_corte['rmse']:.4f}")

        todos_resultados.append({
            'nombre':            nombre,
            'patron':            patron,
            'descripcion':       desc,
            'pct_objetivo':      pct_obj,
            'duracion':          duracion,
            'metricas_global':   metricas_global,
            'metricas_por_nodo': metricas_nodo,
            'metricas_corte':    metricas_corte,
            'metricas_particiones': metricas_particiones,
        })

        # Guardar parcialmente después de cada experimento
        guardar_csv_global(todos_resultados, args.output_dir)
        guardar_csv_por_nodo(todos_resultados, args.output_dir)
        guardar_csv_corte(todos_resultados, args.output_dir)
        guardar_csv_particiones(todos_resultados, args.output_dir)

    imprimir_tabla_final(todos_resultados)
    print(f"\n¡Experimentos completados!")
    print(f"  Global:        {args.output_dir}/resultados.csv")
    print(f"  Por nodo:      {args.output_dir}/resultados_por_nodo.csv")
    print(f"  Corte:         {args.output_dir}/resultados_corte.csv")
    print(f"  Datos origen:  {args.output_dir}/resumen_datos.txt")
    print(f"  Log:           {args.output_dir}/log_experimentos.txt")
    print(f"  Particiones:   {args.output_dir}/resultados_particiones.csv")


if __name__ == '__main__':
    main()