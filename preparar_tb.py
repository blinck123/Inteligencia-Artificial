"""Construye dataset/imagenes y dataset/anotaciones.csv a partir de radiografías de tórax descargadas.

Reconoce:
  * Shenzhen / Montgomery (NLM): CHNCXR_####_X.png, MCUCXR_####_X.png  (X=0 normal, 1 tuberculosis). paciente_id = CHNCXR_####.
  * Carpetas 'Normal' y 'Tuberculosis' (p. ej. TB Chest X-ray Database de Kaggle). Sin ID de paciente real: se usa el nombre
    de archivo, lo que ASUME una imagen por paciente (no verificable; documentarlo como limitación).
Redimensiona a lado máximo --lado (conserva proporción, 8 bits) para entrenar rápido; no toca los originales.
Uso: python preparar_tb.py --origen RUTA [--lado 512] [--max-por-clase 200]
"""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from main import ROOT, SEED

EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

def etiqueta(p, kw_tb=('tubercul',), kw_norm=('normal',)):
    m = re.match(r'^((?:CHNCXR|MCUCXR)_\d+)_([01])$', p.stem, re.I)
    if m: return m.group(1).upper(), int(m.group(2))
    carpeta = ' '.join(x.lower() for x in p.parts[-4:-1])
    txt = carpeta + ' ' + p.stem.lower()
    if any(k in txt for k in kw_tb): return p.stem, 1
    if any(k in txt for k in kw_norm): return p.stem, 0
    return None

def a_8bit(im):
    if im.mode in ('I;16', 'I', 'F'):
        a = np.asarray(im, dtype=np.float64); lo, hi = np.percentile(a, [0.5, 99.5])
        return Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype('uint8'))
    return im.convert('L')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--origen', required=True); ap.add_argument('--lado', type=int, default=512)
    ap.add_argument('--kw-tb', default='tubercul', help='palabras (coma) en carpeta/archivo que indican TB, p. ej. tb,tuberculosis')
    ap.add_argument('--kw-normal', default='normal', help='palabras (coma) que indican normal, p. ej. normal,health')
    ap.add_argument('--max-por-clase', type=int, default=0, help='0 = todas; útil para pruebas rápidas en CPU')
    a = ap.parse_args()
    rows = []; omitidos = 0
    for p in sorted(Path(a.origen).rglob('*')):
        if p.suffix.lower() not in EXT or any('mask' in x.lower() for x in p.parts): continue
        e = etiqueta(p, tuple(k.strip().lower() for k in a.kw_tb.split(',')), tuple(k.strip().lower() for k in a.kw_normal.split(',')))
        if e is None: omitidos += 1; continue
        rows.append({'src': p, 'paciente_id': e[0], 'diagnostico': e[1]})
    if not rows: raise SystemExit('No se encontraron imágenes con etiqueta reconocible.')
    df = pd.DataFrame(rows)
    if a.max_por_clase:
        df = pd.concat([g.sample(min(len(g), a.max_por_clase), random_state=SEED) for _, g in df.groupby('diagnostico')])
    dest = ROOT / 'dataset' / 'imagenes'; dest.mkdir(parents=True, exist_ok=True)
    out = []
    for r in df.itertuples():
        name = f'{r.src.parent.name}_{r.src.stem}.png' if not re.match(r'^(CHNCXR|MCUCXR)', r.src.stem, re.I) else r.src.stem + '.png'
        with Image.open(r.src) as im:
            im = a_8bit(im); im.thumbnail((a.lado, a.lado), Image.LANCZOS); im.save(dest / name)
        out.append({'imagen': name, 'paciente_id': r.paciente_id, 'diagnostico': r.diagnostico})
    res = pd.DataFrame(out).sort_values('imagen'); res.to_csv(ROOT / 'dataset' / 'anotaciones.csv', index=False)
    print(f'{len(res)} imágenes | clases: {res.diagnostico.value_counts().sort_index().to_dict()} | omitidas sin etiqueta: {omitidos}')
    if res.paciente_id.nunique() == len(res) and not res.paciente_id.str.match(r'^(CHN|MCU)CXR', case=False).any():
        print('AVISO: sin ID de paciente real; se asume 1 imagen = 1 paciente.')

if __name__ == '__main__': main()
