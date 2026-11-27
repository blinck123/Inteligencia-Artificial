"""Validación cruzada por paciente (StratifiedGroupKFold) para los modelos base y multitarea.

Uso: python validacion_cruzada.py --folds 5 --epochs 10 --batch-size 8 --pretrained
Ningún paciente aparece en más de una partición dentro de una misma iteración.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from main import ROOT, OUT, DIM, SEED, OralDataset, CNN, run_epoch, seed, read_and_validate

METRICAS = ['accuracy', 'precision', 'recall', 'especificidad', 'f1', 'auc']

def metricas(y, p):
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'accuracy': accuracy_score(y, pred), 'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0), 'especificidad': tn / (tn + fp) if tn + fp else np.nan,
            'f1': f1_score(y, pred, zero_division=0), 'auc': roc_auc_score(y, p)}

def comprobar(df, y, groups, k):
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    splits = list(cv.split(df, y, groups))
    for i, (tr, te) in enumerate(splits, 1):
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            raise ValueError(f'El fold {i} no contiene ambas clases: hay pocos pacientes independientes para k={k}. Reduce --folds.')
        assert not set(groups[tr]) & set(groups[te])
    return splits

def entrenar_fold(multi, tr, va, te, args, device):
    seed()
    model = CNN(multi, args.pretrained).to(device)
    mk = lambda d, sh: DataLoader(OralDataset(d), batch_size=args.batch_size, shuffle=sh, generator=torch.Generator().manual_seed(SEED))
    ltr, lva, lte = mk(tr, True), mk(va, False), mk(te, False)
    opt = torch.optim.Adam(model.parameters(), lr=.0001)
    best, best_state = -1, None
    for _ in range(args.epochs):
        run_epoch(model, ltr, device, opt)
        _, y, p, _, _ = run_epoch(model, lva, device)
        auc = roc_auc_score(y, p) if len(set(y)) > 1 else -1
        if auc > best or best_state is None:
            best = auc; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    _, y, p, _, _ = run_epoch(model, lte, device)
    return y, p

def ic95(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return np.nan, np.nan
    h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return max(0., x.mean() - h), min(1., x.mean() + h)  # métricas en [0,1]; con pocos folds el IC es muy ancho

def bootstrap_pacientes(df_oof, B=2000, modelos=('base', 'multitarea')):
    """IC95% bootstrap (remuestreando pacientes) de AUC por modelo y de la diferencia pareada multitarea - base."""
    rng = np.random.default_rng(SEED)
    pac = df_oof.paciente_id.unique(); grupos = {g: d for g, d in df_oof.groupby('paciente_id')}
    aucs = {m: [] for m in modelos}; dif = []
    for _ in range(B):
        d = pd.concat([grupos[g] for g in rng.choice(pac, len(pac))])
        if d.real.nunique() < 2: continue
        a = {m: roc_auc_score(d.real, d[f'prob_{m}']) for m in aucs}
        for m in aucs: aucs[m].append(a[m])
        if len(modelos) == 2: dif.append(a['multitarea'] - a['base'])
    return aucs, np.array(dif)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5); ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=8); ap.add_argument('--pretrained', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=2000)
    args = ap.parse_args()
    if min(args.folds, args.epochs, args.batch_size) < 1 or args.folds < 2: ap.error('Valores inválidos')
    out = OUT / 'cv'; out.mkdir(parents=True, exist_ok=True)
    df = read_and_validate().reset_index(drop=True)
    y = df.diagnostico.to_numpy(); groups = df.paciente_id.to_numpy()
    MODELOS = [('base', False)] + ([('multitarea', True)] if DIM else [])
    splits = comprobar(df, y, groups, args.folds)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filas, oof = [], []
    for i, (tr_idx, te_idx) in enumerate(splits, 1):
        # Subconjunto de validación (≈1/6 de los pacientes del entrenamiento) para elegir la mejor época.
        tr_df = df.iloc[tr_idx]
        inner = StratifiedGroupKFold(n_splits=min(6, max(2, tr_df.paciente_id.nunique() // 2)), shuffle=True, random_state=SEED)
        a, b = next(inner.split(tr_df, tr_df.diagnostico, tr_df.paciente_id))
        tr, va, te = tr_df.iloc[a], tr_df.iloc[b], df.iloc[te_idx]
        if tr.diagnostico.nunique() < 2 or va.diagnostico.nunique() < 2:
            raise ValueError(f'Fold {i}: el subconjunto interno de validación no tiene ambas clases. Reduce --folds.')
        res = {}
        for name, multi in MODELOS:
            yt, p = entrenar_fold(multi, tr, va, te, args, device)
            m = metricas(yt, p); filas.append({'fold': i, 'modelo': name, 'n_test': len(te), 'pacientes_test': te.paciente_id.nunique(), **m})
            res[name] = p; print(f'fold {i} {name}: AUC={m["auc"]:.3f} acc={m["accuracy"]:.3f}', flush=True)
        oof.append(pd.DataFrame({'fold': i, 'imagen': te.imagen.to_numpy(), 'paciente_id': te.paciente_id.to_numpy(), 'real': yt, **{f'prob_{k}': q for k, q in res.items()}}))
    f = pd.DataFrame(filas); f.to_csv(out / 'metricas_por_fold.csv', index=False)
    oof = pd.concat(oof); oof.to_csv(out / 'predicciones_oof.csv', index=False)
    resumen = []
    for name, _ in MODELOS:
        sub = f[f.modelo == name]
        for m in METRICAS:
            lo, hi = ic95(sub[m]); resumen.append({'modelo': name, 'metrica': m, 'media': sub[m].mean(), 'desv_est': sub[m].std(ddof=1), 'ic95_inf': lo, 'ic95_sup': hi})
    pd.DataFrame(resumen).to_csv(out / 'resumen_cv.csv', index=False)
    if len(MODELOS) < 2:
        print(pd.DataFrame(resumen).round(3).to_string(index=False))
        aucs, _ = bootstrap_pacientes(oof, args.bootstrap, ['base'])
        print('AUC out-of-fold base: %.3f IC95%% bootstrap [%.3f, %.3f]' % (roc_auc_score(oof.real, oof.prob_base), *np.percentile(aucs['base'], [2.5, 97.5])))
        return
    w = f.pivot(index='fold', columns='modelo', values='auc'); d = w['multitarea'] - w['base']
    t = stats.ttest_rel(w['multitarea'], w['base']) if len(w) > 2 else None
    try: wil = stats.wilcoxon(w['multitarea'], w['base']).pvalue
    except ValueError: wil = np.nan
    aucs, dif = bootstrap_pacientes(oof, args.bootstrap)
    q = lambda x: np.percentile(x, [2.5, 97.5])
    comp = pd.DataFrame([{'diferencia_auc_media_folds': d.mean(), 'ic95_t_inf': ic95(d)[0], 'ic95_t_sup': ic95(d)[1],
                          'p_t_pareada': t.pvalue if t else np.nan, 'p_wilcoxon': wil,
                          'auc_oof_base': roc_auc_score(oof.real, oof.prob_base), 'auc_oof_base_ic95_inf': q(aucs['base'])[0], 'auc_oof_base_ic95_sup': q(aucs['base'])[1],
                          'auc_oof_multitarea': roc_auc_score(oof.real, oof.prob_multitarea), 'auc_oof_multi_ic95_inf': q(aucs['multitarea'])[0], 'auc_oof_multi_ic95_sup': q(aucs['multitarea'])[1],
                          'dif_auc_oof_bootstrap_ic95_inf': q(dif)[0], 'dif_auc_oof_bootstrap_ic95_sup': q(dif)[1], 'folds': len(w), 'bootstrap_validos': len(dif)}])
    comp.to_csv(out / 'comparacion_pareada.csv', index=False)
    print(pd.DataFrame(resumen).round(3).to_string(index=False)); print(comp.T.round(4).to_string(header=False))

if __name__ == '__main__': main()
