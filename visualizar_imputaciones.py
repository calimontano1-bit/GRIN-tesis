"""
visualizar_imputaciones.py
==========================
Módulo de visualización de imputaciones GRIN para ManglarIA.

Genera un grid de subgráficas (small multiples) mostrando:
  - Serie original (ground truth)
  - Serie imputada
  - Puntos imputados resaltados

Uso dentro del loop de experimentos:
    from visualizar_imputaciones import plot_imputations_grid

    # Después de cargar el .npz:
    datos = cargar_npz(output_npz)
    if datos is not None:
        plot_imputations_grid(
            data_true     = datos['y_true'],
            data_imputed  = datos['y_hat'],
            mask          = datos['mask'],
            feature_names = VAR_NAMES,      # lista de nombres de variables
            exp_name      = nombre,
            output_dir    = args.output_dir,
        )

Para filtrar por patrón y porcentaje al correr:
    python experimentos.py ... --viz-patron MCAR --viz-pct 5

"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')   # sin pantalla, genera archivos directamente
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ── PALETA VISUAL ─────────────────────────────────────────────────────────────
COLOR_ORIGINAL = '#2C3E50'    # azul oscuro — serie original
COLOR_IMPUTED  = '#95A5A6'    # gris claro  — serie imputada (fondo)
COLOR_POINTS   = '#E74C3C'    # rojo        — puntos imputados resaltados
ALPHA_ORIG     = 0.85
ALPHA_IMP      = 0.55
MARKER_SIZE    = 4


def _reconstruir_serie(data, timestamps=None):
    """
    Reconstruye una serie temporal (T, features) a partir de
    ventanas (B, W, features) tomando el último paso de cada ventana.

    Esto evita duplicar datos de solapamiento entre ventanas contiguas.

    Retorna:
        serie:      (T_reconstruida, features)
        ts_serie:   lista de timestamps si se proporcionan, sino None
    """
    B, W, F = data.shape
    # Tomamos solo el último paso de cada ventana → T_reconstruida ≈ B
    serie = data[:, -1, :]   # (B, F) — sin copiar la dimensión W completa
    if timestamps is not None and len(timestamps) == B:
        return serie, list(timestamps)
    return serie, None


def plot_imputations_grid(
    data_true,
    data_imputed,
    mask,
    feature_names,
    exp_name,
    output_dir='.',
    timestamps=None,
    n_cols=6,
    figsize=(20, 14),
):
    """
    Genera un grid de subgráficas mostrando la imputación de cada variable.

    Parámetros:
        data_true    — array (B, W, features) con valores reales
        data_imputed — array (B, W, features) con predicciones del modelo
        mask         — array (B, W, features) con 1 donde hay hueco imputado
        feature_names — lista de strings con nombre de cada variable
        exp_name     — string con nombre del experimento (título de figura)
        output_dir   — directorio donde guardar la imagen
        timestamps   — lista opcional de strings con fechas del eje X
        n_cols       — número de columnas del grid (default: 6)
        figsize      — tamaño de figura en pulgadas

    Guarda:
        {output_dir}/viz_{exp_name}.png
    """
    # Reconstruir series temporales desde ventanas
    true_series,   ts = _reconstruir_serie(data_true,     timestamps)
    imputed_series, _  = _reconstruir_serie(data_imputed,  timestamps)
    mask_series,    _  = _reconstruir_serie(mask,          timestamps)

    T, n_features = true_series.shape

    # Ajustar número de variables al mínimo entre array y lista de nombres
    n_vars = min(n_features, len(feature_names))
    n_rows = int(np.ceil(n_vars / n_cols))

    # Eje X
    x = np.arange(T)

    # ── FIGURA ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor='#FAFAFA')
    fig.suptitle(
        f'Imputaciones GRIN — {exp_name}',
        fontsize=13, fontweight='bold', color='#2C3E50',
        y=0.98
    )

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        left=0.04, right=0.97,
        top=0.93,  bottom=0.06
    )

    for v in range(n_vars):
        row = v // n_cols
        col = v % n_cols
        ax  = fig.add_subplot(gs[row, col])

        y_true = true_series[:, v]
        y_hat  = imputed_series[:, v]
        m      = mask_series[:, v].astype(bool)

        # Serie imputada de fondo (completa)
        ax.plot(x, y_hat,
                color=COLOR_IMPUTED, linewidth=0.8,
                alpha=ALPHA_IMP, zorder=1, label='_nolegend_')

        # Serie original encima
        ax.plot(x, y_true,
                color=COLOR_ORIGINAL, linewidth=0.9,
                alpha=ALPHA_ORIG, zorder=2, label='_nolegend_')

        # Puntos imputados resaltados
        if m.sum() > 0:
            ax.scatter(
                x[m], y_hat[m],
                color=COLOR_POINTS, s=MARKER_SIZE**2,
                zorder=3, linewidths=0, label='_nolegend_'
            )

        # Estética del subplot
        ax.set_title(feature_names[v], fontsize=7, pad=3,
                     color='#34495E', fontweight='semibold')
        ax.tick_params(axis='both', labelsize=5, length=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        ax.set_facecolor('#F8F9FA')

        # Sombrear zonas imputadas
        if m.sum() > 0:
            _sombrear_zonas(ax, x, m)

    # Ocultar subplots vacíos
    for v in range(n_vars, n_rows * n_cols):
        row = v // n_cols
        col = v % n_cols
        fig.add_subplot(gs[row, col]).set_visible(False)

    # Leyenda global
    legend_elements = [
        Line2D([0], [0], color=COLOR_ORIGINAL, linewidth=1.5, label='Original'),
        Line2D([0], [0], color=COLOR_IMPUTED,  linewidth=1.5,
               alpha=0.7, label='Imputado'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLOR_POINTS, markersize=5, label='Punto imputado'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor='#BDC3C7',
        bbox_to_anchor=(0.5, 0.01)
    )

    # Guardar
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'viz_{exp_name}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close(fig)
    print(f"  [✓ Visualización] {out_path}")
    return out_path


def _sombrear_zonas(ax, x, mask_bool):
    """
    Sombrea con color suave las zonas donde hay huecos imputados.
    Agrupa posiciones contiguas en bloques para eficiencia.
    """
    ymin, ymax = ax.get_ylim()
    in_block = False
    start    = 0
    for i, val in enumerate(mask_bool):
        if val and not in_block:
            start    = x[i]
            in_block = True
        elif not val and in_block:
            ax.axvspan(start - 0.5, x[i - 1] + 0.5,
                       color=COLOR_POINTS, alpha=0.08, zorder=0)
            in_block = False
    if in_block:
        ax.axvspan(start - 0.5, x[-1] + 0.5,
                   color=COLOR_POINTS, alpha=0.08, zorder=0)


# ── INTEGRACIÓN CON experimentos.py ──────────────────────────────────────────

def args_visualizacion(parser):
    """
    Agrega argumentos CLI de visualización a un ArgumentParser existente.

    Uso en experimentos.py:
        from visualizar_imputaciones import args_visualizacion
        parser = args_visualizacion(parser)   # dentro de parse_args()

    Argumentos añadidos:
        --viz-patron   patrón a visualizar (MCAR, MAR, MNAR, o 'todos')
        --viz-pct      porcentaje a visualizar (1, 3, 5, 10, 15, o 0 = todos)
    """
    parser.add_argument(
        '--viz-patron', type=str, default='todos',
        help="Patrón de missingness a visualizar: MCAR, MAR, MNAR, o 'todos' (default)"
    )
    parser.add_argument(
        '--viz-pct', type=int, default=0,
        help="Porcentaje objetivo a visualizar: 1, 3, 5, 10, 15, o 0 = todos (default)"
    )
    return parser


def debe_visualizar(nombre, patron, pct_objetivo, viz_patron, viz_pct):
    """
    Decide si este experimento debe generar visualización.

    Parámetros:
        nombre       — ej. 'mcar_5pct'
        patron       — ej. 'MCAR'
        pct_objetivo — ej. 5.0
        viz_patron   — argumento --viz-patron del CLI
        viz_pct      — argumento --viz-pct del CLI (0 = todos)

    Retorna:
        True si debe visualizarse, False si no
    """
    if viz_patron.lower() != 'todos':
        if patron.upper() != viz_patron.upper():
            return False
    if viz_pct != 0:
        if int(pct_objetivo) != viz_pct:
            return False
    return True