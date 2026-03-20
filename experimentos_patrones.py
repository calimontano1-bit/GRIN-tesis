"""
experimentos_patrones.py
========================
Corre GRIN bajo 6 configuraciones de datos faltantes (MCAR, MAR, MNAR)
y genera una tabla comparativa + gráfica de resultados.

Uso:
    python experimentos_patrones.py

Los resultados se guardan en: resultados_patrones/
    - resultados.csv       ← tabla con todas las métricas
    - comparacion.png      ← gráfica comparativa
    - log_experimentos.txt ← log completo de cada corrida

Tiempo estimado: 12-15 horas (6 experimentos × ~2h cada uno)
"""

import os
import sys
import csv
import time
import subprocess
import datetime
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── EXPERIMENTOS ──────────────────────────────────────────────────────────────
# Cada entrada: (nombre, patron, p_block, p_point, min_seq, max_seq, descripcion)
EXPERIMENTOS = [
    # Mismo % faltante (~5%, ~20%, ~60%), patrón diferente
    # nombre,          patron,  p_block, p_point, min_seq, max_seq, descripcion

    # MCAR — solo huecos puntuales aleatorios
    # % = p_point directamente
    ("mcar_5pct",    "MCAR", 0.000, 0.050, 4,  6, "MCAR ~5%"),
    ("mcar_20pct",   "MCAR", 0.000, 0.200, 4,  6, "MCAR ~20%"),
    ("mcar_60pct",   "MCAR", 0.000, 0.600, 4,  6, "MCAR ~60%"),

    # MAR — mezcla bloques cortos (media=5) + puntuales
    # % ≈ p_block×5 + p_point  →  repartido 60/40 entre bloques y puntuales
    ("mar_5pct",     "MAR",  0.006, 0.020, 4,  6, "MAR ~5%"),
    ("mar_20pct",    "MAR",  0.024, 0.080, 4,  6, "MAR ~20%"),
    ("mar_60pct",    "MAR",  0.072, 0.240, 4,  6, "MAR ~60%"),

    # MNAR — solo bloques largos (media=9), sin puntuales
    # % ≈ p_block×9  →  p_block = % / 9
    ("mnar_5pct",    "MNAR", 0.006, 0.000, 6, 12, "MNAR ~5%"),
    ("mnar_20pct",   "MNAR", 0.022, 0.000, 6, 12, "MNAR ~20%"),
    ("mnar_60pct",   "MNAR", 0.067, 0.000, 6, 12, "MNAR ~60%"),
]

EPOCHS      = 4
BATCH_SIZE  = 64
WORKERS     = 8
BASE_CONFIG = "config/grin/synthetic.yaml"
OUTPUT_DIR  = "resultados_patrones"
# ──────────────────────────────────────────────────────────────────────────────


def crear_config_temporal(nombre, p_block, p_point, min_seq, max_seq):
    """Crea un yaml temporal con los parámetros del experimento."""
    with open(BASE_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    config['epochs']     = EPOCHS
    config['batch_size'] = BATCH_SIZE
    # Estos parámetros los lee el SyntheticAdapter vía args o los pasamos
    # directamente como variables de entorno que el script lee
    config_path = f"{OUTPUT_DIR}/config_{nombre}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path


def parsear_metricas(output_texto):
    """Extrae métricas del output de run_imputation.py."""
    metricas = {}
    for linea in output_texto.split('\n'):
        for metrica in ['test_mae', 'test_mse', 'test_mre', 'test_mape', 'test_loss']:
            if f"'{metrica}'" in linea or f'"{metrica}"' in linea:
                try:
                    # Busca el número después del ':'
                    partes = linea.split(':')
                    valor = float(partes[-1].strip().rstrip(',}'))
                    metricas[metrica] = valor
                except (ValueError, IndexError):
                    pass
    return metricas


def parsear_metricas_por_epoch(output_texto):
    """Extrae val_mae por epoch del output."""
    epochs = []
    for linea in output_texto.split('\n'):
        if 'val_mae=' in linea and 'Epoch' in linea:
            try:
                epoch_num = int(linea.split('Epoch')[1].split(':')[0].strip())
                val_mae   = float(linea.split('val_mae=')[1].split(',')[0].split(']')[0])
                epochs.append((epoch_num, val_mae))
            except (ValueError, IndexError):
                pass
    return epochs


def estimar_porcentaje_faltante(p_block, p_point, min_seq, max_seq):
    """Estima el % aproximado de datos faltantes."""
    long_media_bloque = (min_seq + max_seq) / 2
    # P(dato faltante) ≈ p_block * long_media + p_point (aproximación)
    p_total = min(p_block * long_media_bloque + p_point, 1.0)
    return round(p_total * 100, 1)


def correr_experimento(nombre, p_block, p_point, min_seq, max_seq, log_file):
    """Corre un experimento y devuelve las métricas."""
    print(f"\n{'='*60}")
    print(f"  Iniciando: {nombre}")
    print(f"  p_block={p_block}, p_point={p_point}, "
          f"min_seq={min_seq}, max_seq={max_seq}")
    print(f"{'='*60}")

    config_path = crear_config_temporal(nombre, p_block, p_point, min_seq, max_seq)

    # Variables de entorno para pasar parámetros al SyntheticAdapter
    env = os.environ.copy()
    env['GRIN_P_BLOCK']  = str(p_block)
    env['GRIN_P_POINT']  = str(p_point)
    env['GRIN_MIN_SEQ']  = str(min_seq)
    env['GRIN_MAX_SEQ']  = str(max_seq)

    cmd = [
        sys.executable, "-m", "scripts.run_imputation",
        "--config", config_path,
        "--dataset-name", "synthetic",
        "--workers", str(WORKERS),
    ]

    inicio = time.time()
    resultado = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
    )
    duracion = time.time() - inicio

    output_completo = resultado.stdout + resultado.stderr

    # Guardar log
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"EXPERIMENTO: {nombre}\n")
        f.write(f"Inicio: {datetime.datetime.now()}\n")
        f.write(f"Duración: {duracion/3600:.2f}h\n")
        f.write(output_completo)
        f.write(f"\n{'='*60}\n")

    metricas = parsear_metricas(output_completo)
    epochs_data = parsear_metricas_por_epoch(output_completo)

    if metricas:
        print(f"  ✓ Completado en {duracion/3600:.2f}h")
        print(f"  test_mae={metricas.get('test_mae', 'N/A'):.4f}  "
              f"test_mre={metricas.get('test_mre', 'N/A'):.2f}%")
    else:
        print(f"  ✗ Error — revisa {log_file}")
        print(f"  Últimas líneas: {output_completo[-500:]}")

    return metricas, epochs_data, duracion


def graficar_resultados(todos_resultados):
    """Genera gráfica comparativa de todos los experimentos."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    nombres    = [r['nombre'] for r in todos_resultados]
    patrones   = [r['patron'] for r in todos_resultados]
    maes       = [r['metricas'].get('test_mae', 0) for r in todos_resultados]
    mres       = [r['metricas'].get('test_mre', 0) for r in todos_resultados]
    mses       = [r['metricas'].get('test_mse', 0) for r in todos_resultados]

    colores_patron = {'MCAR': '#4fc3f7', 'MAR': '#81c784', 'MNAR': '#ff8a65'}
    colores = [colores_patron[p] for p in patrones]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('#0f1117')

    metricas_plot = [
        (maes, 'MAE', 'Mean Absolute Error'),
        (mres, 'MRE (%)', 'Mean Relative Error'),
        (mses, 'MSE', 'Mean Squared Error'),
    ]

    for ax, (valores, ylabel, titulo) in zip(axes, metricas_plot):
        ax.set_facecolor('#1a1d27')
        bars = ax.bar(range(len(nombres)), valores, color=colores,
                      edgecolor='#333344', linewidth=0.8)

        # Etiquetas de valor sobre cada barra
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(valores)*0.01,
                    f'{val:.3f}', ha='center', va='bottom',
                    color='white', fontsize=8)

        ax.set_xticks(range(len(nombres)))
        ax.set_xticklabels(nombres, rotation=35, ha='right',
                           color='#aaaaaa', fontsize=8)
        ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=10)
        ax.set_title(titulo, color='white', fontsize=11, pad=10)
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333344')
        ax.grid(True, alpha=0.15, color='white', axis='y')
        ax.set_ylim(0, max(valores) * 1.2 if max(valores) > 0 else 1)

    # Leyenda de patrones
    from matplotlib.patches import Patch
    leyenda = [Patch(color=c, label=p) for p, c in colores_patron.items()]
    fig.legend(handles=leyenda, loc='lower center', ncol=3,
               facecolor='#1a1d27', edgecolor='#333344',
               labelcolor='white', fontsize=10,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle('GRIN — Comparación por patrón de datos faltantes\n'
                 'MCAR vs MAR vs MNAR',
                 color='white', fontsize=13, y=1.02)

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/comparacion.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nGráfica guardada en: {output_path}")


def guardar_csv(todos_resultados):
    """Guarda tabla de resultados en CSV."""
    csv_path = f"{OUTPUT_DIR}/resultados.csv"
    campos = ['nombre', 'patron', 'descripcion', 'p_block', 'p_point',
              'min_seq', 'max_seq', 'pct_faltante_est',
              'test_mae', 'test_mse', 'test_mre', 'test_mape',
              'duracion_horas']

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for r in todos_resultados:
            fila = {
                'nombre':            r['nombre'],
                'patron':            r['patron'],
                'descripcion':       r['descripcion'],
                'p_block':           r['p_block'],
                'p_point':           r['p_point'],
                'min_seq':           r['min_seq'],
                'max_seq':           r['max_seq'],
                'pct_faltante_est':  r['pct_faltante'],
                'test_mae':          r['metricas'].get('test_mae', ''),
                'test_mse':          r['metricas'].get('test_mse', ''),
                'test_mre':          r['metricas'].get('test_mre', ''),
                'test_mape':         r['metricas'].get('test_mape', ''),
                'duracion_horas':    round(r['duracion'] / 3600, 2),
            }
            writer.writerow(fila)

    print(f"Tabla guardada en: {csv_path}")
    return csv_path


def imprimir_tabla_final(todos_resultados):
    """Imprime tabla resumen en consola."""
    print(f"\n{'='*80}")
    print(f"  RESULTADOS FINALES")
    print(f"{'='*80}")
    print(f"{'Experimento':<16} {'Patrón':<6} {'%Falt':>6} "
        f"{'MAE':>8} {'MRE%':>8} {'MSE':>8} {'MAPE':>8} {'Horas':>7}")
    print(f"{'-'*90}")

    for r in todos_resultados:
        m = r['metricas']
        print(f"{r['nombre']:<16} {r['patron']:<6} {float(r['pct_faltante']):>5.1f}% "
            f"{m.get('test_mae','---'):>8.4f} "
            f"{m.get('test_mre','---'):>7.2f}% "
            f"{m.get('test_mse','---'):>8.4f} "
            f"{m.get('test_mape','---'):>8.4f} "
            f"{r['duracion']/3600:>6.2f}h")

    print(f"{'='*80}")


def main():
    # Verificar que estamos en el directorio correcto
    if not os.path.exists(BASE_CONFIG):
        print(f"ERROR: No se encontró {BASE_CONFIG}")
        print("Asegúrate de correr este script desde la carpeta raíz de grin/")
        sys.exit(1)

    # Crear directorio de resultados
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_file  = f"{OUTPUT_DIR}/log_experimentos.txt"
    inicio_total = time.time()

    print("=" * 60)
    print("  EXPERIMENTOS GRIN: MCAR vs MAR vs MNAR")
    print(f"  {len(EXPERIMENTOS)} experimentos × {EPOCHS} epochs")
    print(f"  Resultados en: {OUTPUT_DIR}/")
    print("=" * 60)

    # Verificar que run_imputation.py lee variables de entorno
    # Si no, hay que modificar el SyntheticAdapter para leerlas
    _parchear_run_imputation_si_necesario()

    todos_resultados = []

    for nombre, patron, p_block, p_point, min_seq, max_seq, desc in EXPERIMENTOS:
        pct = estimar_porcentaje_faltante(p_block, p_point, min_seq, max_seq)
        metricas, epochs_data, duracion = correr_experimento(
            nombre, p_block, p_point, min_seq, max_seq, log_file
        )
        todos_resultados.append({
            'nombre':      nombre,
            'patron':      patron,
            'descripcion': desc,
            'p_block':     p_block,
            'p_point':     p_point,
            'min_seq':     min_seq,
            'max_seq':     max_seq,
            'pct_faltante': pct,
            'metricas':    metricas,
            'epochs_data': epochs_data,
            'duracion':    duracion,
        })

        # Guardar resultados parciales después de cada experimento
        guardar_csv(todos_resultados)

    duracion_total = time.time() - inicio_total
    print(f"\nTiempo total: {duracion_total/3600:.2f}h")

    imprimir_tabla_final(todos_resultados)
    guardar_csv(todos_resultados)

    if any(r['metricas'] for r in todos_resultados):
        graficar_resultados(todos_resultados)

    print(f"\n¡Experimentos completados!")
    print(f"  Tabla:   {OUTPUT_DIR}/resultados.csv")
    print(f"  Gráfica: {OUTPUT_DIR}/comparacion.png")
    print(f"  Log:     {OUTPUT_DIR}/log_experimentos.txt")


def _parchear_run_imputation_si_necesario():
    with open('scripts/run_imputation.py', 'r', encoding='utf-8') as f:
        contenido = f.read()

    if 'GRIN_P_BLOCK' in contenido:
        print("✓ run_imputation.py ya lee variables de entorno")
        return

    print("⚠ Falta el parche en run_imputation.py")
    print("  Asegúrate de que ChargedParticles use os.environ.get()")
    sys.exit(1)


if __name__ == '__main__':
    main()