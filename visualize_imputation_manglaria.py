"""
visualize_imputation_manglaria.py
================================

Visualiza imputaciones generadas por experimentos3.py (GRIN).

Uso:
    python visualize_imputation_manglaria.py

Requiere:
    Archivo .npz generado por experimentos3.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CONFIG ─────────────────────────────────────────────────────────────
NPZ_PATH = 'resultados_manglaria/results_mcar_5pct.npz'

N_PASOS  = 200      # pasos a visualizar
NODO     = 0        # nodo (sensor)
VAR      = 0        # variable dentro del nodo
# ───────────────────────────────────────────────────────────────────────


def cargar_npz(path):
    data = np.load(path, allow_pickle=True)
    return {
        'y_hat': data['y_hat'],     # (B, W, F)
        'y_true': data['y_true'],
        'mask': data['mask'],       # eval_mask
        'timestamps': data['timestamps']
    }


def reconstruir_series(y_hat, y_true, mask, n_nodos):
    """
    Convierte (B, W, F) → (T, N, d)
    """
    B, W, F = y_hat.shape
    d = F // n_nodos

    T = B * W

    yh = y_hat.reshape(T, n_nodos, d)
    yt = y_true.reshape(T, n_nodos, d)
    mk = mask.reshape(T, n_nodos, d)

    return yh, yt, mk


def graficar(y_hat, y_true, mask):
    T = min(N_PASOS, y_hat.shape[0])
    tiempo = np.arange(T)

    real  = y_true[:T, NODO, VAR]
    imput = y_hat[:T, NODO, VAR]
    hueco = mask[:T, NODO, VAR].astype(bool)

    fig, ax = plt.subplots(figsize=(14, 6))

    # sombrear huecos
    en_hueco = False
    inicio = 0
    for t in range(T + 1):
        if t < T and hueco[t]:
            if not en_hueco:
                inicio = t
                en_hueco = True
        else:
            if en_hueco:
                ax.axvspan(inicio, t, alpha=0.2)
                en_hueco = False

    # real
    ax.plot(tiempo, real, linestyle='--', label='Real')

    # predicción visible
    ax.plot(tiempo, np.where(~hueco, imput, np.nan),
            label='Predicción (visible)')

    # imputación real
    ax.plot(tiempo, np.where(hueco, imput, np.nan),
            linewidth=2.5, label='Imputación')

    ax.set_title(f'Manglaria — Nodo {NODO}, Variable {VAR}')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Valor')

    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('imputacion_manglaria.png', dpi=150)
    print("Guardado en imputacion_manglaria.png")


def main():
    print("Cargando resultados...")

    data = cargar_npz(NPZ_PATH)

    # Detectar nodos automáticamente (manglaria = 1 nodo normalmente)
    F = data['y_hat'].shape[2]
    n_nodos = 1  # puedes ajustar si tienes grafo

    yh, yt, mk = reconstruir_series(
        data['y_hat'],
        data['y_true'],
        data['mask'],
        n_nodos
    )

    print(f"Serie reconstruida: {yh.shape}")
    print(f"Huecos evaluados: {mk.sum()}")

    graficar(yh, yt, mk)


if __name__ == '__main__':
    main()