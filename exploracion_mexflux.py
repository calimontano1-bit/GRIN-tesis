import pandas as pd

df = pd.read_csv('mexflux.csv')
print('Valores únicos de site_id:', df['site_id'].unique())
print('Total filas:', len(df))
print()
print('Por sensor:')
for s in df['site_id'].unique():
    sub = df[df['site_id'] == s]
    print(f'  {s}: {len(sub)} filas, desde {sub["timestamp"].min()} hasta {sub["timestamp"].max()}')
print()
print('Valores nulos por columna:')
print(df.isnull().sum())