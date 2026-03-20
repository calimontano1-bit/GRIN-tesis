"""
visualize_imputation.py
=======================
Carga el modelo GRIN entrenado, aplica imputación sobre datos de test,
y genera una gráfica comparando valores reales vs imputados vs huecos.

Uso:
    python visualize_imputation.py

Ajusta CKPT_PATH si quieres usar otro checkpoint.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
CKPT_PATH   = 'logs/synthetic/grin/2026-03-05_13-39-35_267551687/epoch=3-step=10935.ckpt'
DATA_PATH   = 'datasets/synthetic/charged_varying.npz'

# Qué visualizar
N_PASOS     = 50      # pasos de tiempo a mostrar (max 36 por window)
NODO        = 0       # qué partícula/sensor mostrar (0-9)
SIMULACION  = 42      # qué simulación del dataset usar (0-4999)

# Parámetros del dataset (deben coincidir con synthetic.yaml)
P_BLOCK     = 0.025
P_POINT     = 0.025
MIN_SEQ     = 4
MAX_SEQ     = 9
WINDOW      = 36
# ──────────────────────────────────────────────────────────────────────────────


def cargar_datos():
    """Carga el dataset sintético y prepara datos de una simulación."""
    from lib.datasets.synthetic import ChargedParticles

    print("Cargando dataset...")
    raw = ChargedParticles(
        static_adj=False,
        p_block=P_BLOCK,
        p_point=P_POINT,
        min_seq=MIN_SEQ,
        max_seq=MAX_SEQ,
        use_exogenous=False
    )

    # Extraemos una simulación específica
    # raw.loc:  (5000, 50, 10, 2) → tomamos simulación SIMULACION
    # raw.mask: (5000, 50, 10, 2) → 1=dato visible, 0 = hueco
    loc       = raw.loc[SIMULACION].numpy()       # (50, 10, 2)
    mask      = raw.mask[SIMULACION].numpy()      # (50, 10, 2)
    eval_mask = raw.eval_mask[SIMULACION].numpy() # (50, 10, 2)

    return loc, mask, eval_mask, raw


def cargar_modelo():
    """Carga el modelo GRIN desde el checkpoint guardado."""
    from lib.nn.models.grin import GRINet

    print("Cargando modelo desde checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location='cpu')

    hp = ckpt['hyper_parameters']
    model = GRINet(
        adj=hp['adj'],
        d_in=hp['d_in'],
        d_hidden=hp['d_hidden'],
        d_ff=hp['d_ff'],
        ff_dropout=hp['ff_dropout'],
        n_layers=hp['n_layers'],
        kernel_size=hp['kernel_size'],
        decoder_order=hp['decoder_order'],
        d_emb=hp['d_emb'],
        layer_norm=hp['layer_norm'],
        merge=hp['merge']
    )

    # Extraemos solo los pesos del modelo (sin el filler wrapper)
    state_dict = ckpt['state_dict']
    model_state = {
        k.replace('model.', ''): v
        for k, v in state_dict.items()
        if k.startswith('model.')
    }
    model.load_state_dict(model_state)
    model.eval()

    return model, hp['adj']


def imputar(model, adj, loc, mask):
    """
    Aplica el modelo sobre una ventana de datos con huecos.
    
    Entrada:
        loc:  (T, N, d) — posiciones reales
        mask: (T, N, d) — 1=visible, 0=hueco
    
    Salida:
        y_hat: (T, N, d) — valores imputados por el modelo
    """
    T, N, d = loc.shape

    # El modelo necesita batches: añadimos dimensión batch → (1, T, N, d)
    x_masked = loc * mask                                    # aplicar huecos
    x_tensor = torch.tensor(x_masked).float().unsqueeze(0)  # (1, T, N, d)
    m_tensor  = torch.tensor(mask).byte().unsqueeze(0)      # (1, T, N, d) — Byte requerido por gril.py

    # Nota: GRINet.forward NO recibe adj como argumento —
    # lo tiene internamente como self.adj cargado desde el checkpoint.
    # En modo eval (model.eval()) solo devuelve imputation, no una tupla.
    with torch.no_grad():
        imputation = model(x_tensor, mask=m_tensor)

    # imputation: (1, T, N, d) → (T, N, d)
    y_hat = imputation.squeeze(0).numpy()
    return y_hat


def graficar(loc, mask, eval_mask, y_hat):
    """
    Genera la gráfica de comparación real vs imputado.
    
    Para el NODO seleccionado muestra:
    - Variable x (coordenada horizontal de la partícula)
    - Variable y (coordenada vertical de la partícula)
    
    Colores:
    - Azul sólido:    valor real (siempre visible para referencia)
    - Verde sólido:   predicción del modelo en zonas visibles
    - Rojo sólido:    predicción del modelo en huecos (la imputación real)
    - Fondo gris:     zonas donde había huecos en la máscara
    """
    T = min(N_PASOS, loc.shape[0])
    tiempo = np.arange(T)

    # Máscara combinada: hueco si mask=0 O eval_mask=1
    training_mask = mask[:T, NODO, :] & (1 - eval_mask[:T, NODO, :])
    es_hueco = (training_mask == 0)  # True donde el modelo NO vio el dato

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('#0f1117')

    gs = GridSpec(2, 1, figure=fig, hspace=0.45)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]

    nombres_var = ['Coordenada X', 'Coordenada Y']
    colores = ['#4fc3f7', '#81c784']

    for dim, (ax, nombre, color) in enumerate(zip(axes, nombres_var, colores)):
        ax.set_facecolor('#1a1d27')

        real   = loc[:T, NODO, dim]
        imput  = y_hat[:T, NODO, dim]
        hueco  = es_hueco[:, dim]

        # Sombrear zonas con huecos
        en_hueco = False
        inicio   = 0
        for t in range(T + 1):
            if t < T and hueco[t]:
                if not en_hueco:
                    inicio   = t
                    en_hueco = True
            else:
                if en_hueco:
                    ax.axvspan(inicio - 0.5, t - 0.5,
                               alpha=0.25, color='#ff6b6b', zorder=1)
                    en_hueco = False

        # Línea real (referencia completa)
        ax.plot(tiempo, real, color=color, linewidth=1.5,
                alpha=0.5, linestyle='--', label='Valor real', zorder=2)

        # Predicción en zonas visibles
        imput_visible = np.where(~hueco, imput, np.nan)
        ax.plot(tiempo, imput_visible, color=color, linewidth=2,
                alpha=0.9, label='Predicción (zona visible)', zorder=3)

        # Predicción en huecos — esto es la imputación
        imput_hueco = np.where(hueco, imput, np.nan)
        ax.plot(tiempo, imput_hueco, color='#ff6b6b', linewidth=2.5,
                alpha=1.0, label='Imputación (zona con hueco)', zorder=4)

        # Puntos en los huecos para resaltarlos
        idx_huecos = np.where(hueco)[0]
        if len(idx_huecos) > 0:
            ax.scatter(idx_huecos, real[idx_huecos],
                       color='white', s=20, zorder=5, alpha=0.6)
            ax.scatter(idx_huecos, imput[idx_huecos],
                       color='#ff6b6b', s=35, zorder=6,
                       edgecolors='white', linewidths=0.5)

        ax.set_title(f'{nombre} — Partícula {NODO}',
                     color='white', fontsize=12, pad=10)
        ax.set_xlabel('Paso de tiempo', color='#aaaaaa', fontsize=10)
        ax.set_ylabel('Posición', color='#aaaaaa', fontsize=10)
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333344')
        ax.grid(True, alpha=0.15, color='white')

        # Calcular MAE solo en los huecos
        if len(idx_huecos) > 0:
            mae_huecos = np.mean(np.abs(real[idx_huecos] - imput[idx_huecos]))
            ax.text(0.02, 0.95, f'MAE en huecos: {mae_huecos:.4f}',
                    transform=ax.transAxes, color='#ffcc80',
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='#1a1d27',
                              alpha=0.8, edgecolor='#ffcc80'))

    # Leyenda global
    parches = [
        mpatches.Patch(color='#4fc3f7', alpha=0.5, label='Valor real (referencia)'),
        mpatches.Patch(color='#4fc3f7', label='Predicción en zona visible'),
        mpatches.Patch(color='#ff6b6b', label='Imputación en hueco'),
        mpatches.Patch(color='#ff6b6b', alpha=0.25, label='Zona con hueco'),
    ]
    fig.legend(handles=parches, loc='lower center', ncol=4,
               facecolor='#1a1d27', edgecolor='#333344',
               labelcolor='white', fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        f'GRIN — Imputación de datos sintéticos\n'
        f'Simulación {SIMULACION} · {len(idx_huecos)} huecos en {T} pasos',
        color='white', fontsize=14, y=0.98
    )

    # Guardar
    output_path = 'imputacion_visualizacion.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nGráfica guardada en: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("  Visualización de imputaciones GRIN")
    print("=" * 60)

    loc, mask, eval_mask, raw = cargar_datos()
    model, adj = cargar_modelo()

    print(f"Simulación {SIMULACION}: {loc.shape[0]} pasos, "
          f"{loc.shape[1]} nodos, {loc.shape[2]} variables")

    n_huecos = int((mask[:N_PASOS, NODO, :] == 0).sum())
    print(f"Huecos en partícula {NODO}: {n_huecos} de "
          f"{N_PASOS * loc.shape[2]} valores")

    print("Aplicando modelo...")
    y_hat = imputar(model, adj, loc[:WINDOW], mask[:WINDOW])

    print("Generando gráfica...")
    graficar(loc[:WINDOW], mask[:WINDOW], eval_mask[:WINDOW], y_hat)

    # Imprimir tabla de comparación para X e Y
    training_mask = mask[:WINDOW, NODO, :] & (1 - eval_mask[:WINDOW, NODO, :])

    for dim, nombre_dim in enumerate(['X', 'Y']):
        huecos_dim = np.where(training_mask[:, dim] == 0)[0]
        if len(huecos_dim) > 0:
            print(f"\nComparación en huecos — Variable {nombre_dim}, Partícula {NODO}:")
            print(f"{'Paso':>6} {'Real':>10} {'Imputado':>10} {'Error abs':>10}")
            print("-" * 40)
            for t in huecos_dim[:10]:
                real_val  = loc[t, NODO, dim]
                imput_val = y_hat[t, NODO, dim]
                error     = abs(real_val - imput_val)
                print(f"{t:>6} {real_val:>10.4f} {imput_val:>10.4f} {error:>10.4f}")
            if len(huecos_dim) > 10:
                print(f"  ... y {len(huecos_dim) - 10} huecos más")

    print("\n¡Listo! Abre 'imputacion_visualizacion.png' para ver los resultados.")

if __name__ == '__main__':
    main()