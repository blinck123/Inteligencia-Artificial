"""Auditoría del dataset real antes de entrenar: estructura, balance, pacientes, duplicados exactos y casi-duplicados,
calidad básica de imagen. Genera resultados/auditoria_dataset.csv y resultados/auditoria_resumen.txt.
Uso: python auditar_dataset.py [--umbral-hash 6] [--min-lado 224] [--min-nitidez 50]
No modifica ni inventa datos. Los casos marcados requieren revisión humana (no se eliminan automáticamente)."""
import argparse, hashlib
from itertools import combinations
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from main import ROOT, OUT, DIM_OPCIONAL

IMG = ROOT / 'dataset' / 'imagenes'

def dhash(im, n=8):
    g = im.convert('L').resize((n + 1, n), Image.LANCZOS); a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

def nitidez(im):
    g = np.asarray(im.convert('L').filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    return float(g.var())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--umbral-hash', type=int, default=6, help='distancia de Hamming dHash (de 64) para casi-duplicado')
    ap.add_argument('--min-lado', type=int, default=224); ap.add_argument('--min-nitidez', type=float, default=50.)
    a = ap.parse_args(); OUT.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / 'dataset' / 'anotaciones.csv', dtype={'paciente_id': str, 'imagen': str})
    if df.empty: raise ValueError('anotaciones.csv está vacío: agrega datos reales autorizados.')
    prob = []; filas = []; hs = {}; sha = {}
    for r in df.itertuples():
        p = IMG / r.imagen; f = {'imagen': r.imagen, 'paciente_id': r.paciente_id, 'problemas': []}
        if not p.exists(): f['problemas'].append('archivo_inexistente'); filas.append(f); continue
        try:
            with Image.open(p) as im:
                im.load(); f['ancho'], f['alto'] = im.size
                if min(im.size) < a.min_lado: f['problemas'].append('baja_resolucion')
                f['nitidez'] = nitidez(im)
                if f['nitidez'] < a.min_nitidez: f['problemas'].append('posible_desenfoque')
                hs[r.imagen] = dhash(im)
        except Exception as e: f['problemas'].append(f'ilegible:{type(e).__name__}')
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h in sha: f['problemas'].append(f'duplicado_exacto:{sha[h]}')
        else: sha[h] = r.imagen
        filas.append(f)
    info = {f['imagen']: f for f in filas}; pac = df.set_index('imagen').paciente_id
    for x, y in combinations(hs, 2):
        d = int((hs[x] != hs[y]).sum())
        if d <= a.umbral_hash:
            info[y]['problemas'].append(f'casi_duplicado:{x}(d={d},{"mismo" if pac[x]==pac[y] else "DISTINTO"}_paciente)')
    ex = [dict(f, problemas=';'.join(f['problemas'])) for f in filas]
    pd.DataFrame(ex).to_csv(OUT / 'auditoria_dataset.csv', index=False)
    L = []; pp = df.groupby('paciente_id').diagnostico.agg(['nunique', 'first'])
    L.append(f'Imágenes: {len(df)} | Pacientes: {len(pp)}')
    L.append('Imágenes por clase: ' + str(df.diagnostico.value_counts().sort_index().to_dict()))
    L.append('Pacientes por clase (diagnóstico único): ' + str(pp[pp["nunique"] == 1]["first"].value_counts().sort_index().to_dict()))
    L.append(f'Pacientes con diagnóstico mixto: {(pp["nunique"] > 1).sum()}')
    L.append(f'Pacientes con >1 imagen: {(df.groupby("paciente_id").size() > 1).sum()}')
    for d, cats in DIM_OPCIONAL.items():
        if d not in df.columns or df[d].isna().all(): continue
        L.append(f'{d}: ' + str(df[d].value_counts(dropna=False).to_dict()) + (' | INVÁLIDAS: ' + str(set(df[d].dropna()) - set(cats)) if not set(df[d].dropna()).issubset(cats) else ''))
    n = sum(1 for f in filas if f['problemas']); L.append(f'Imágenes con alertas: {n}/{len(df)} (ver auditoria_dataset.csv)')
    pos = int(pp['first'].eq(1).sum()); neg = int(pp['first'].eq(0).sum())
    if min(pos, neg) < 10: L.append(f'ADVERTENCIA: pocos pacientes en una clase (0:{neg}, 1:{pos}); la partición 70/15/15 y la CV por paciente serán poco fiables.')
    open(OUT / 'auditoria_resumen.txt', 'w', encoding='utf-8').write('\n'.join(L)); print('\n'.join(L))

if __name__ == '__main__': main()
