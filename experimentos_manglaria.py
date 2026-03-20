"""
experimentos_manglaria.py
=========================
Corre GRIN sobre datos reales de ManglarIA introduciendo huecos
artificiales bajo patrones MCAR, MAR y MNAR en distintos porcentajes.

Los huecos se introducen por variable individual (no por fila completa),
respetando los 0.2% de huecos reales ya existentes.

Diseño:
    5 niveles: 1%, 3%, 5%, 10%, 15%
    3 patrones: MCAR, MAR, MNAR
    Total: 15 experimentos

Uso:
    python experimentos_manglaria.py

Resultados en: resultados_manglaria/
"""

import os
import sys
import csv
import time
import subprocess
import datetime
import yaml
import shutil
import numpy as np
import pandas as pd

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
DATA_PATH     = 'manglaria_abril_timeseries.csv'
TIMESTAMP_COL = 'TIMESTAMP'
TIMESTAMP_FMT = '%d/%m/%Y %H:%M'
BASE_CONFIG   = 'config/grin/manglaria.yaml'
OUTPUT_DIR    = 'resultados_manglaria'
EPOCHS        = 12
WORKERS       = 8

# Experimentos: (nombre, patron, pct_objetivo, p_block, p_point, min_seq, max_seq)
# pct_objetivo incluye el 0.2% real ya existente
# Los parámetros de máscara se aplican sobre las celdas sin hueco real
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


def cargar_datos():
    """Carga el CSV y devuelve DataFrame con índice temporal."""
    df = pd.read_csv(DATA_PATH)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], format=TIMESTAMP_FMT)
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df = df.set_index(TIMESTAMP_COL)
    return df.select_dtypes(include=[np.number])


def generar_mascara_artificial(df, p_block, p_point, min_seq, max_seq, seed=42):
    """
    Genera una máscara de huecos artificiales sobre celdas actualmente presentes.
    
    Respeta los huecos reales existentes — solo introduce huecos nuevos
    donde actualmente hay datos.
    
    Retorna:
        mascara_artificial: (T, N) bool — True donde se introduce hueco artificial
    """
    rng = np.random.RandomState(seed)
    T, N = df.shape
    presente = (~df.isnull()).values  # (T, N) — True donde hay dato

    mascara = np.zeros((T, N), dtype=bool)

    for col in range(N):
        t = 0
        while t < T:
            # Hueco en bloque
            if rng.random() < p_block:
                largo = rng.randint(min_seq, max_seq + 1)
                for dt in range(largo):
                    if t + dt < T and presente[t + dt, col]:
                        mascara[t + dt, col] = True
                t += largo
            else:
                # Hueco puntual
                if rng.random() < p_point and presente[t, col]:
                    mascara[t, col] = True
                t += 1

    return mascara


def guardar_mascara_temporal(mascara, nombre):
    """Guarda la máscara como archivo .npy para que el adaptador la lea."""
    path = f"{OUTPUT_DIR}/mask_{nombre}.npy"
    np.save(path, mascara)
    return path


def crear_config_temporal(nombre):
    """Crea yaml temporal para el experimento."""
    with open(BASE_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    config['epochs'] = EPOCHS
    config_path = f"{OUTPUT_DIR}/config_{nombre}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path


def parsear_metricas(output_texto):
    """Extrae métricas del output de run_imputation.py."""
    metricas = {}
    for linea in output_texto.split('\n'):
        for metrica in ['test_mae', 'test_mse', 'test_mre', 'test_mape']:
            if f"'{metrica}'" in linea or f'"{metrica}"' in linea:
                try:
                    valor = float(linea.split(':')[-1].strip().rstrip(',}'))
                    metricas[metrica] = valor
                except (ValueError, IndexError):
                    pass
    return metricas


def correr_experimento(nombre, p_block, p_point, min_seq, max_seq, log_file):
    """Corre un experimento y devuelve métricas."""
    print(f"\n{'='*60}")
    print(f"  Iniciando: {nombre}")
    print(f"  p_block={p_block}, p_point={p_point}, "
          f"min_seq={min_seq}, max_seq={max_seq}")
    print(f"{'='*60}")

    config_path = crear_config_temporal(nombre)

    env = os.environ.copy()
    env['GRIN_P_BLOCK']      = str(p_block)
    env['GRIN_P_POINT']      = str(p_point)
    env['GRIN_MIN_SEQ']      = str(min_seq)
    env['GRIN_MAX_SEQ']      = str(max_seq)
    env['GRIN_MASK_PATH']    = f"{OUTPUT_DIR}/mask_{nombre}.npy"
    env['GRIN_DATASET_TYPE'] = 'manglaria'

    cmd = [
        sys.executable, "-m", "scripts.run_imputation",
        "--config", config_path,
        "--dataset-name", "manglaria",
        "--workers", str(WORKERS),
    ]

    inicio = time.time()
    resultado = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
    )
    duracion = time.time() - inicio

    output_completo = resultado.stdout + resultado.stderr

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"EXPERIMENTO: {nombre}\n")
        f.write(f"Duración: {duracion/3600:.2f}h\n")
        f.write(output_completo)

    metricas = parsear_metricas(output_completo)

    if metricas:
        print(f"  ✓ Completado en {duracion/3600:.2f}h")
        print(f"  test_mae={metricas.get('test_mae', 'N/A'):.4f}  "
              f"test_mre={metricas.get('test_mre', 'N/A'):.2f}%")
    else:
        print(f"  ✗ Error — revisa {log_file}")
        print(f"  Últimas líneas:\n{output_completo[-300:]}")

    return metricas, duracion


def guardar_csv(todos_resultados):
    """Guarda tabla de resultados."""
    csv_path = f"{OUTPUT_DIR}/resultados.csv"
    campos = ['nombre', 'patron', 'descripcion', 'pct_objetivo',
              'p_block', 'p_point', 'min_seq', 'max_seq',
              'test_mae', 'test_mse', 'test_mre', 'test_mape',
              'duracion_horas']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for r in todos_resultados:
            fila = {
                'nombre':         r['nombre'],
                'patron':         r['patron'],
                'descripcion':    r['descripcion'],
                'pct_objetivo':   r['pct_objetivo'],
                'p_block':        r['p_block'],
                'p_point':        r['p_point'],
                'min_seq':        r['min_seq'],
                'max_seq':        r['max_seq'],
                'test_mae':       r['metricas'].get('test_mae', ''),
                'test_mse':       r['metricas'].get('test_mse', ''),
                'test_mre':       r['metricas'].get('test_mre', ''),
                'test_mape':      r['metricas'].get('test_mape', ''),
                'duracion_horas': round(r['duracion'] / 3600, 2),
            }
            writer.writerow(fila)
    print(f"Tabla guardada en: {csv_path}")


def imprimir_tabla_final(todos_resultados):
    """Imprime tabla resumen en consola."""
    print(f"\n{'='*75}")
    print(f"  RESULTADOS FINALES — ManglarIA")
    print(f"{'='*75}")
    print(f"  {'Experimento':<14} {'Patrón':<6} {'%Obj':>5} "
          f"{'MAE':>8} {'MRE%':>8} {'MSE':>8} {'Horas':>7}")
    print(f"  {'-'*73}")

    for r in todos_resultados:
        m = r['metricas']
        mae  = m.get('test_mae', float('nan'))
        mre  = m.get('test_mre', float('nan'))
        mse  = m.get('test_mse', float('nan'))
        h    = r['duracion'] / 3600
        try:
            print(f"  {r['nombre']:<14} {r['patron']:<6} {r['pct_objetivo']:>4.0f}% "
                  f"{mae:>8.4f} {mre:>7.2f}% {mse:>8.4f} {h:>6.2f}h")
        except (TypeError, ValueError):
            print(f"  {r['nombre']:<14} {r['patron']:<6} {r['pct_objetivo']:>4.0f}% "
                  f"{'---':>8} {'---':>8} {'---':>8} {h:>6.2f}h")

    print(f"{'='*75}")


def verificar_adaptador():
    """
    Verifica que el ManglarIAAdapter en run_imputation.py
    lee la variable de entorno GRIN_MASK_PATH para la eval_mask.
    Si no, indica qué cambiar.
    """
    with open('scripts/run_imputation.py', 'r', encoding='utf-8') as f:
        contenido = f.read()

    if 'GRIN_MASK_PATH' in contenido:
        print("✓ ManglarIAAdapter ya lee GRIN_MASK_PATH")
        return True
    else:
        print("⚠ El ManglarIAAdapter necesita un cambio para leer la máscara externa.")
        print("  En scripts/run_imputation.py, dentro de ManglarIAAdapter.__init__,")
        print("  busca la sección '── EVAL MASK ──' y reemplázala por:\n")
        print("""
        # ── EVAL MASK ─────────────────────────────────────────────────────
        mask_path = os.environ.get('GRIN_MASK_PATH', '')
        if mask_path and os.path.exists(mask_path):
            # Huecos artificiales introducidos por experimentos_manglaria.py
            artificial = np.load(mask_path).astype(np.uint8)  # (T, N)
            eval_mask_np = artificial
        else:
            # Fallback: huecos reales del CSV
            eval_mask_np = (df.isnull()).values.astype(np.uint8)
        self._eval_mask = eval_mask_np.reshape(T, N, 1)
        """)
        print("  Haz ese cambio y vuelve a correr este script.\n")
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = f"{OUTPUT_DIR}/log_experimentos.txt"

    print("=" * 60)
    print("  EXPERIMENTOS MANGLARIA: MCAR vs MAR vs MNAR")
    print(f"  {len(EXPERIMENTOS)} experimentos × {EPOCHS} epochs")
    print(f"  Dataset: {DATA_PATH}")
    print(f"  Resultados en: {OUTPUT_DIR}/")
    print("=" * 60)

    if not verificar_adaptador():
        sys.exit(1)

    # Cargar datos una sola vez para generar máscaras
    print("\nCargando datos y generando máscaras...")
    df = cargar_datos()
    T, N = df.shape
    pct_real = df.isnull().mean().mean() * 100
    print(f"  {T} pasos × {N} variables, {pct_real:.2f}% huecos reales")

    todos_resultados = []

    for nombre, patron, pct_obj, p_block, p_point, min_seq, max_seq, desc in EXPERIMENTOS:

        # Generar y guardar máscara artificial
        mascara = generar_mascara_artificial(
            df, p_block, p_point, min_seq, max_seq,
            seed=hash(nombre) % 2**31
        )
        pct_artificial = mascara.mean() * 100
        guardar_mascara_temporal(mascara, nombre)
        print(f"  {nombre}: {pct_artificial:.2f}% huecos artificiales "
              f"(+{pct_real:.2f}% reales = {pct_artificial+pct_real:.2f}% total)")

        metricas, duracion = correr_experimento(
            nombre, p_block, p_point, min_seq, max_seq, log_file
        )

        todos_resultados.append({
            'nombre':      nombre,
            'patron':      patron,
            'descripcion': desc,
            'pct_objetivo': pct_obj,
            'p_block':     p_block,
            'p_point':     p_point,
            'min_seq':     min_seq,
            'max_seq':     max_seq,
            'metricas':    metricas,
            'duracion':    duracion,
        })

        guardar_csv(todos_resultados)

    imprimir_tabla_final(todos_resultados)
    print(f"\n¡Experimentos completados!")
    print(f"  Tabla:   {OUTPUT_DIR}/resultados.csv")
    print(f"  Log:     {OUTPUT_DIR}/log_experimentos.txt")


if __name__ == '__main__':
    main()