"""
explorar_sensor.py
==================
Exploración de datos de sensor ambiental para tesis ManglarIA.
Solo output en consola, sin gráficas.

Uso:
    python explorar_sensor.py ruta/al/archivo.csv
"""

import sys
import os
import numpy as np
import pandas as pd

TIMESTAMP_COL = 'TIMESTAMP'
TIMESTAMP_FMT = '%d/%m/%Y %H:%M'


def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)

    if TIMESTAMP_COL not in df.columns:
        print(f"ERROR: No se encontró columna '{TIMESTAMP_COL}'")
        print(f"Columnas disponibles: {df.columns.tolist()}")
        sys.exit(1)

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], format=TIMESTAMP_FMT)
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df = df.set_index(TIMESTAMP_COL)
    df = df.select_dtypes(include=[np.number])
    return df


def analizar(df):
    sep = '=' * 65

    # RESUMEN GENERAL
    print(f"\n{sep}")
    print(f"  RESUMEN GENERAL")
    print(sep)

    total_celdas = df.shape[0] * df.shape[1]
    total_nulos  = df.isnull().sum().sum()
    pct_global   = total_nulos / total_celdas * 100

    diffs    = df.index.to_series().diff().dropna()
    freq     = diffs.mode()[0]
    huecos_t = int((diffs > freq).sum())

    print(f"  Período:                 {df.index[0]}  →  {df.index[-1]}")
    print(f"  Filas (pasos de tiempo): {df.shape[0]}")
    print(f"  Variables numéricas:     {df.shape[1]}")
    print(f"  Frecuencia de muestreo:  {freq}")
    print(f"  Saltos en la línea de tiempo: {huecos_t} "
          f"({'hay interrupciones' if huecos_t > 0 else 'serie continua'})")
    print(f"  Valores faltantes totales: {total_nulos:,} de {total_celdas:,} "
          f"({pct_global:.1f}%)")

    # FALTANTES POR VARIABLE
    print(f"\n{sep}")
    print(f"  DATOS FALTANTES POR VARIABLE")
    print(sep)
    print(f"  {'Variable':<40} {'Faltantes':>9} {'%':>7}  Estado")
    print(f"  {'-'*63}")

    pcts = df.isnull().mean() * 100
    for col in pcts.sort_values(ascending=False).index:
        pct = pcts[col]
        n   = int(df[col].isnull().sum())
        if pct == 0:
            estado = 'completa'
        elif pct < 5:
            estado = 'leve'
        elif pct < 20:
            estado = 'moderada'
        else:
            estado = 'severa'
        print(f"  {col[:40]:<40} {n:>9,} {pct:>6.1f}%  {estado}")

    # ESTADÍSTICAS DESCRIPTIVAS
    print(f"\n{sep}")
    print(f"  ESTADÍSTICAS DESCRIPTIVAS")
    print(sep)
    print(f"  {'Variable':<40} {'Media':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*83}")

    for col in df.columns:
        media = df[col].mean()
        std   = df[col].std()
        vmin  = df[col].min()
        vmax  = df[col].max()
        print(f"  {col[:40]:<40} {media:>10.3f} {std:>10.3f} {vmin:>10.3f} {vmax:>10.3f}")

    # ANÁLISIS DE BLOQUES DE HUECOS
    print(f"\n{sep}")
    print(f"  ANÁLISIS DE BLOQUES DE HUECOS")
    print(sep)

    faltante_fila = df.isnull().any(axis=1)
    bloques = []
    contador = 0
    for v in faltante_fila:
        if v:
            contador += 1
        else:
            if contador > 0:
                bloques.append(contador)
                contador = 0
    if contador > 0:
        bloques.append(contador)

    if bloques:
        media_b = np.mean(bloques)
        max_b   = int(np.max(bloques))
        n_b     = len(bloques)
        pct_b   = sum(b > 1 for b in bloques) / len(bloques) * 100
        pct_f   = faltante_fila.mean() * 100

        print(f"  Filas con algún hueco:        {int(faltante_fila.sum()):,} ({pct_f:.1f}%)")
        print(f"  Número de bloques de huecos:  {n_b}")
        print(f"  Longitud media de bloque:     {media_b:.1f} pasos  ({media_b*0.5:.1f} horas)")
        print(f"  Bloque más largo:             {max_b} pasos  ({max_b*0.5:.1f} horas)")
        print(f"  Bloques de más de 1 paso:     {pct_b:.1f}%")

        print(f"\n  Orientación del patrón:")
        if media_b < 2:
            print(f"  → MCAR predominante — huecos puntuales dispersos sin patrón claro")
        elif media_b < 6:
            print(f"  → MAR / mezcla — bloques cortos mezclados con huecos puntuales")
        else:
            print(f"  → MNAR predominante — bloques largos, posible fallo de sensor")

        if max_b > 48:
            print(f"\n  Atención: el bloque más largo es de {max_b} pasos "
                  f"= {max_b*0.5:.0f} horas consecutivas sin datos.")
            print(f"  Puede indicar mantenimiento o fallo prolongado del sensor.")
    else:
        print(f"  No se detectaron huecos en las filas.")

    # PATRÓN TEMPORAL
    print(f"\n{sep}")
    print(f"  PATRÓN TEMPORAL DE HUECOS")
    print(sep)

    por_hora = faltante_fila.groupby(faltante_fila.index.hour).mean() * 100
    hora_max = por_hora.idxmax()
    hora_min = por_hora.idxmin()
    var_hora = por_hora.max() - por_hora.min()

    print(f"  Hora con más huecos:   {hora_max:02d}:00  ({por_hora[hora_max]:.1f}%)")
    print(f"  Hora con menos huecos: {hora_min:02d}:00  ({por_hora[hora_min]:.1f}%)")
    if var_hora > 10:
        print(f"  → Variación significativa por hora ({var_hora:.1f}pp) — sugiere MAR diario")
    else:
        print(f"  → Sin variación significativa por hora ({var_hora:.1f}pp)")

    dias_nom = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    por_dia  = faltante_fila.groupby(faltante_fila.index.dayofweek).mean() * 100
    dia_max  = por_dia.idxmax()
    dia_min  = por_dia.idxmin()
    var_dia  = por_dia.max() - por_dia.min()

    print(f"  Día con más huecos:    {dias_nom[dia_max]}  ({por_dia[dia_max]:.1f}%)")
    print(f"  Día con menos huecos:  {dias_nom[dia_min]}  ({por_dia[dia_min]:.1f}%)")
    if var_dia > 10:
        print(f"  → Variación significativa por día ({var_dia:.1f}pp) — sugiere MAR semanal")
    else:
        print(f"  → Sin variación significativa por día ({var_dia:.1f}pp)")

    # RESUMEN PARA TESIS
    print(f"\n{sep}")
    print(f"  RESUMEN PARA TESIS")
    print(sep)
    vars_completas = int((pcts == 0).sum())
    vars_leves     = int(((pcts > 0) & (pcts < 5)).sum())
    vars_moderadas = int(((pcts >= 5) & (pcts < 20)).sum())
    vars_severas   = int((pcts >= 20).sum())

    print(f"  El dataset tiene {df.shape[0]} observaciones de {df.shape[1]} variables")
    print(f"  con frecuencia de muestreo de {freq}.")
    print(f"  El {pct_global:.1f}% de los valores totales están faltantes.")
    print(f"  Variables completas: {vars_completas}  |  leves (<5%): {vars_leves}  |  "
          f"moderadas (5-20%): {vars_moderadas}  |  severas (>20%): {vars_severas}")

    if bloques:
        patron = 'MCAR' if media_b < 2 else 'MAR' if media_b < 6 else 'MNAR'
        print(f"  Se detectaron {n_b} bloques de datos faltantes con longitud")
        print(f"  media de {media_b:.1f} pasos ({media_b*0.5:.1f} horas), orientando el patrón hacia {patron}.")

    print(sep)


def main():
    if len(sys.argv) < 2:
        print("Uso: python explorar_sensor.py ruta/al/archivo.csv")
        sys.exit(1)

    ruta_csv = sys.argv[1]
    if not os.path.exists(ruta_csv):
        print(f"ERROR: No se encontró el archivo: {ruta_csv}")
        sys.exit(1)

    df = cargar_datos(ruta_csv)
    analizar(df)


if __name__ == '__main__':
    main()