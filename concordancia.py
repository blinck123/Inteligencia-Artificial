from itertools import combinations
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from main import ROOT, OUT, DIM, DIM_OPCIONAL, ORDINALES
import pingouin as pg

def main():
    OUT.mkdir(exist_ok=True)
    df=pd.read_csv(ROOT/'dataset'/'evaluadores.csv')
    if df.empty: raise ValueError('Completa evaluadores.csv con anotaciones reales independientes.')
    if df.duplicated(['imagen','evaluador','condicion']).any(): raise ValueError('Anotaciones duplicadas.')
    DIM.clear(); DIM.update({d:v for d,v in DIM_OPCIONAL.items() if d in df.columns and df[d].notna().any()})
    if not DIM: raise ValueError('evaluadores.csv no tiene columnas de la rúbrica con datos.')
    rows=[]; iccs=[]
    for cond,group in df.groupby('condicion'):
        for d,cats in DIM.items():
            if not set(group[d].dropna()).issubset(cats): raise ValueError(f'Categoría inválida: {d}')
            pivot=group.pivot(index='imagen',columns='evaluador',values=d)
            ordinal=d in ORDINALES
            for a,b in combinations(pivot.columns,2):
                pair=pivot[[a,b]].dropna()
                if len(pair)<2: continue
                x=pair[a].map(cats.index); y=pair[b].map(cats.index)
                rows.append({'condicion':cond,'dimension':d,'evaluador_a':a,'evaluador_b':b,'n':len(pair),'kappa':cohen_kappa_score(x,y,labels=list(range(len(cats))),weights='quadratic' if ordinal else None),'acuerdo':(x==y).mean()})
            if ordinal and pivot.shape[1]>=2:
                complete=pivot.dropna()
                if len(complete)>=3:
                    long=complete.stack().rename('puntaje').reset_index(); long.puntaje=long.puntaje.map(cats.index)
                    # ICC2 como análisis complementario, condicionado a justificar distancias iguales.
                    result=pg.intraclass_corr(data=long,targets='imagen',raters='evaluador',ratings='puntaje').query("Type == 'ICC2'")
                    iccs.append({'condicion':cond,'dimension':d,'n':len(complete),'icc2':result.ICC.iloc[0]})
    pd.DataFrame(rows).to_csv(OUT/'kappa.csv',index=False)
    pd.DataFrame(iccs,columns=['condicion','dimension','n','icc2']).to_csv(OUT/'icc_exploratorio.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
