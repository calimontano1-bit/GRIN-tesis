"""
visualizar_imputaciones.py
==========================
Modulo de visualizacion de imputaciones GRIN.
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


COLOR_ORIGINAL = '#2C3E50'
COLOR_IMPUTED = '#95A5A6'
COLOR_POINTS = '#E74C3C'
ALPHA_ORIG = 0.85
ALPHA_IMP = 0.55
MARKER_SIZE = 4


def _reconstruir_serie(data, timestamps=None):
    """
    Reconstruye una serie temporal (T, features) a partir de
    ventanas (B, W, features) tomando el ultimo paso de cada ventana.
    """
    b, _, _ = data.shape
    serie = data[:, -1, :]
    if timestamps is not None and len(timestamps) == b:
        return serie, list(timestamps)
    return serie, None


def _expandir_feature_names(feature_names, n_features, n_nodes=1, d=None, site_names=None):
    if feature_names is None:
        return [f'f_{i}' for i in range(n_features)]
    if len(feature_names) == n_features:
        return list(feature_names)
    if d is None and n_nodes > 0 and n_features % n_nodes == 0:
        d = n_features // n_nodes
    if d is not None and len(feature_names) == d and n_nodes > 1:
        if site_names is None or len(site_names) != n_nodes:
            site_names = [f'node_{i}' for i in range(n_nodes)]
        expanded = []
        for site in site_names:
            expanded.extend([f"{site}::{name}" for name in feature_names])
        if len(expanded) == n_features:
            return expanded
    return [f'f_{i}' for i in range(n_features)]


def _sombrear_zonas(ax, x, mask_bool):
    in_block = False
    start = 0
    for i, val in enumerate(mask_bool):
        if val and not in_block:
            start = x[i]
            in_block = True
        elif not val and in_block:
            ax.axvspan(start - 0.5, x[i - 1] + 0.5,
                       color=COLOR_POINTS, alpha=0.08, zorder=0)
            in_block = False
    if in_block:
        ax.axvspan(start - 0.5, x[-1] + 0.5,
                   color=COLOR_POINTS, alpha=0.08, zorder=0)


def _plot_single_grid(
    data_true,
    data_imputed,
    mask,
    feature_names,
    exp_name,
    output_dir='.',
    timestamps=None,
    observed_mask=None,
    n_cols=6,
    figsize=(20, 14),
    y_percentile_clip=(1, 99),
    filename_suffix='',
):
    true_series, _ = _reconstruir_serie(data_true, timestamps)
    imputed_series, _ = _reconstruir_serie(data_imputed, timestamps)
    mask_series, _ = _reconstruir_serie(mask, timestamps)
    observed_series = None
    if observed_mask is not None:
        observed_series, _ = _reconstruir_serie(observed_mask, timestamps)

    t, n_features = true_series.shape
    n_vars = min(n_features, len(feature_names))
    n_rows = int(np.ceil(n_vars / n_cols))
    x = np.arange(t)

    fig = plt.figure(figsize=figsize, facecolor='#FAFAFA')
    fig.suptitle(
        f'Imputaciones GRIN - {exp_name}{filename_suffix}',
        fontsize=13, fontweight='bold', color='#2C3E50',
        y=0.98
    )

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        left=0.04, right=0.97,
        top=0.93, bottom=0.06
    )

    for v in range(n_vars):
        row = v // n_cols
        col = v % n_cols
        ax = fig.add_subplot(gs[row, col])

        y_true = true_series[:, v].copy()
        y_hat = imputed_series[:, v].copy()
        m = mask_series[:, v].astype(bool)

        if observed_series is not None:
            observed = observed_series[:, v].astype(bool)
            y_true_plot = y_true.copy()
            y_true_plot[~observed] = np.nan
        else:
            y_true_plot = y_true

        ax.plot(x, y_true_plot,
                color=COLOR_ORIGINAL, linewidth=0.9,
                alpha=ALPHA_ORIG, zorder=2, label='_nolegend_')

        if m.sum() > 0:
            ax.scatter(
                x[m], y_hat[m],
                color=COLOR_POINTS, s=MARKER_SIZE ** 2,
                zorder=3, linewidths=0, label='_nolegend_'
            )

        vals = np.concatenate([
            y_true_plot[np.isfinite(y_true_plot)],
            y_hat[np.isfinite(y_hat)]
        ])
        if vals.size > 0:
            lo, hi = np.percentile(vals, y_percentile_clip)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                pad = 0.05 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)

        ax.set_title(feature_names[v], fontsize=7, pad=3,
                     color='#34495E', fontweight='semibold')
        ax.tick_params(axis='both', labelsize=5, length=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
        ax.set_facecolor('#F8F9FA')

        if m.sum() > 0:
            _sombrear_zonas(ax, x, m)

    for v in range(n_vars, n_rows * n_cols):
        row = v // n_cols
        col = v % n_cols
        fig.add_subplot(gs[row, col]).set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=COLOR_ORIGINAL, linewidth=1.5, label='Original observado'),
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

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"viz_{exp_name}{filename_suffix}.png")
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close(fig)
    print(f"  [OK Visualizacion] {out_path}")
    return out_path


def plot_imputations_grid(
    data_true,
    data_imputed,
    mask,
    feature_names,
    exp_name,
    output_dir='.',
    timestamps=None,
    observed_mask=None,
    site_names=None,
    n_nodes=1,
    d=None,
    n_cols=6,
    figsize=(20, 14),
    y_percentile_clip=(1, 99),
):
    n_features = data_true.shape[-1]

    if n_nodes is None or n_nodes < 1:
        n_nodes = 1
    if d is None and n_nodes > 0 and n_features % n_nodes == 0:
        d = n_features // n_nodes
    if d is None:
        d = n_features
    
    feature_names = _expandir_feature_names(
        feature_names,
        n_features,
        n_nodes=n_nodes,
        d=d,
        site_names=site_names
    )

    # ─── CASO ESPECIAL: muchos nodos con 1 sola variable ───
    # Ejemplo: ManglarIA (27 nodos × 1 variable)
    # Queremos una sola cuadrícula, no una imagen por variable.
    if d == 1 and n_nodes > 1:
        return _plot_single_grid(
            data_true=data_true,
            data_imputed=data_imputed,
            mask=mask,
            feature_names=feature_names[:n_features],
            exp_name=exp_name,
            output_dir=output_dir,
            timestamps=timestamps,
            observed_mask=observed_mask,
            n_cols=n_cols,
            figsize=figsize,
            y_percentile_clip=y_percentile_clip,
            filename_suffix='',          # una sola imagen sin sufijo de nodo
        )

    outputs = []
    if n_nodes > 1 and d * n_nodes <= n_features:
        if site_names is None or len(site_names) != n_nodes:
            site_names = [f'node_{i}' for i in range(n_nodes)]
        for node_idx, site_name in enumerate(site_names):
            start = node_idx * d
            end = start + d
            node_observed = observed_mask[:, :, start:end] if observed_mask is not None else None
            node_feature_names = [name.split('::', 1)[-1] for name in feature_names[start:end]]
            outputs.append(_plot_single_grid(
                data_true=data_true[:, :, start:end],
                data_imputed=data_imputed[:, :, start:end],
                mask=mask[:, :, start:end],
                feature_names=node_feature_names,
                exp_name=exp_name,
                output_dir=output_dir,
                timestamps=timestamps,
                observed_mask=node_observed,
                n_cols=n_cols,
                figsize=figsize,
                y_percentile_clip=y_percentile_clip,
                filename_suffix=f'_{site_name}',
            ))
        return outputs

    return _plot_single_grid(
        data_true=data_true,
        data_imputed=data_imputed,
        mask=mask,
        feature_names=feature_names[:n_features],
        exp_name=exp_name,
        output_dir=output_dir,
        timestamps=timestamps,
        observed_mask=observed_mask,
        n_cols=n_cols,
        figsize=figsize,
        y_percentile_clip=y_percentile_clip,
    )


def args_visualizacion(parser):
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
    if viz_patron.lower() != 'todos':
        if patron.upper() != viz_patron.upper():
            return False
    if viz_pct != 0:
        if int(pct_objetivo) != viz_pct:
            return False
    return True
