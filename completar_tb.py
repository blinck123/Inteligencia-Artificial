"""Complemento para main.py original corregido. No modifica las imágenes ni los resultados originales."""
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)
from skimage.feature import hog
import main as m
import matplotlib.pyplot as plt

ROOT=m.ROOT
OLD=ROOT/'resultados'
OUT=ROOT/'resultados_controlados'
IMG=ROOT/'dataset'/'imagenes'

def preparar(umbral):
    m.OUT=OUT
    df=m.read_and_validate()
    audit=pd.read_csv(OLD/'auditoria_dataset.csv').fillna('')
    par={x:x for x in audit.imagen}
    if not set(df.imagen)<=set(par): raise ValueError('Auditoria no corresponde al dataset actual')
    def root(x):
        while par[x]!=x:
            par[x]=par[par[x]]; x=par[x]
        return x
    def unir(a,b):
        if a not in par or b not in par: raise ValueError('Referencia ausente en auditoria')
        par[root(a)]=root(b)
    for r in audit.itertuples():
        for s in r.problemas.split(';'):
            match=re.fullmatch(r'casi_duplicado:(.*)\(d=(\d+),.*\)',s)
            if match and int(match[2])<=umbral: unir(r.imagen,match[1])
            elif s.startswith('duplicado_exacto:'): unir(r.imagen,s.split(':',1)[1])
    for _,g in df.groupby('paciente_id'):
        names=g.imagen.tolist()
        for x in names[1:]: unir(names[0],x)
    df['grupo_similitud']=df.imagen.map(root)
    df.to_csv(OUT/'grupos.csv',index=False)
    # Seleccionar particiones por grupos usando solamente tamaños y etiquetas.
    def split(d,frac):
        best=None
        for a,b in GroupShuffleSplit(n_splits=300,test_size=frac,random_state=42).split(d,groups=d.grupo_similitud):
            x,y=d.iloc[a],d.iloc[b]
            if x.diagnostico.nunique()!=2 or y.diagnostico.nunique()!=2: continue
            err=sum(abs((y.diagnostico==k).sum()/(d.diagnostico==k).sum()-frac) for k in [0,1])
            if best is None or err<best[0]: best=(err,x,y)
        if best is None: raise ValueError('No hay particion viable con este umbral')
        return best[1],best[2]
    tr,tmp=split(df,.30); va,te=split(tmp,.50)
    parts=[tr,va,te]
    for i in range(3):
        for j in range(i): assert not set(parts[i].grupo_similitud)&set(parts[j].grupo_similitud)
    for n,d in zip(['train','valid','test'],parts):
        d.to_csv(OUT/f'{n}.csv',index=False); print(n,len(d),d.diagnostico.value_counts().to_dict(),flush=True)
    print('Mayor grupo:',df.grupo_similitud.value_counts().max(), 'Umbral:',umbral)
    (OUT/'protocolo.txt').write_text(f'dHash <= {umbral}; grupos transitivos; IDs de paciente no verificados; semilla 42; particiones aproximadas 70/15/15; control parcial de similitud.',encoding='utf-8')
    return parts

def features(d):
    x=[]
    for name in d.imagen:
        with Image.open(IMG/name) as im:
            a=np.asarray(im.convert('L').resize((128,128)))
        x.append(hog(a,pixels_per_cell=(16,16),cells_per_block=(2,2),orientations=9))
    return np.asarray(x)

def svm(parts):
    tr,va,te=parts
    print('Extrayendo HOG...',flush=True)
    x,v,z=[features(d) for d in parts]
    best=None
    for c in [.01,.1,1,10]:
        model=make_pipeline(StandardScaler(),LinearSVC(C=c,class_weight='balanced',random_state=42,max_iter=10000))
        model.fit(x,tr.diagnostico)
        auc=roc_auc_score(va.diagnostico,model.decision_function(v))
        if best is None or auc>best[0]: best=(auc,c,model)
    score=best[2].decision_function(z); pred=(score>=0).astype(int); y=te.diagnostico.to_numpy()
    r=dict(modelo='HOG_SVM',accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred),especificidad=recall_score(y,pred,pos_label=0),f1=f1_score(y,pred),auc=roc_auc_score(y,score))
    pd.DataFrame({'imagen':te.imagen,'real':y,'score':score,'prediccion':pred}).to_csv(OUT/'svm_predicciones.csv',index=False)
    pd.DataFrame([r]).to_csv(OUT/'svm_metricas.csv',index=False)
    (OUT/'svm_config.txt').write_text(f'C={best[1]}; AUC validacion={best[0]}',encoding='utf-8')
    ConfusionMatrixDisplay.from_predictions(y,pred,labels=[0,1]); plt.savefig(OUT/'svm_confusion.png'); plt.close()
    return r,y,score

def gradcam(te,original=False):
    folder=OLD if original else OUT
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=m.CNN(False,False).to(device); model.load_state_dict(torch.load(folder/'base.pt',map_location=device,weights_only=True)); model.eval()
    acts=[]
    hook=model.backbone.layer4[-1].register_forward_hook(lambda mod,inp,out: acts.append(out))
    dest=folder/'gradcam'; dest.mkdir(exist_ok=True)
    sample=pd.concat([g.sample(min(4,len(g)),random_state=42) for _,g in te.groupby('diagnostico')]).reset_index(drop=True)
    ds=m.OralDataset(sample)
    rows=[]
    try:
        for i,r in sample.iterrows():
            acts.clear(); x,_,_=ds[i]; model.zero_grad(); logits,_=model(x.unsqueeze(0).to(device)); activation=acts[-1]
            grad=torch.autograd.grad(logits[0,1],activation)[0]
            cam=(grad.mean(dim=(2,3),keepdim=True)*activation).sum(1).relu()[0].detach().cpu().numpy(); cam/=max(float(cam.max()),1e-12)
            with Image.open(IMG/r.imagen) as im: img=np.asarray(im.convert('L').resize((224,224)))
            fig,ax=plt.subplots(1,2,figsize=(7,3.5)); ax[0].imshow(img,cmap='gray'); ax[1].imshow(img,cmap='gray'); ax[1].imshow(cam,cmap='jet',alpha=.4,extent=(0,224,224,0));
            prob=float(logits.softmax(1)[0,1].detach().cpu()); ax[0].set_title('Radiografia'); ax[1].set_title(f'Grad-CAM TB | p={prob:.3f}');
            for a in ax: a.axis('off')
            fig.tight_layout(); fig.savefig(dest/f'gradcam_{i}.png',dpi=150); plt.close(fig)
            rows.append(dict(archivo=f'gradcam_{i}.png',imagen=r.imagen,real=r.diagnostico,probabilidad=prob))
    finally: hook.remove()
    pd.DataFrame(rows).to_csv(dest/'casos.csv',index=False)

def cli():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--umbral',type=int,choices=range(7),default=0); p.add_argument('--preparar-only',action='store_true'); p.add_argument('--gradcam-original',action='store_true'); a=p.parse_args()
    if a.gradcam_original: gradcam(pd.read_csv(OLD/'test.csv'),True); return
    OUT.mkdir(exist_ok=True); parts=preparar(a.umbral)
    if a.preparar_only: return
    a.pretrained=True; m.OUT=OUT
    s,ys,ss=svm(parts); r,y,prob=m.train('base',False,parts,a)
    pd.DataFrame([s,r]).to_csv(OUT/'comparacion.csv',index=False)
    fig,ax=plt.subplots(); RocCurveDisplay.from_predictions(ys,ss,name='HOG + SVM',ax=ax); RocCurveDisplay.from_predictions(y,prob,name='ResNet18',ax=ax); fig.savefig(OUT/'roc_comparativa.png'); plt.close(fig)
    print(pd.DataFrame([s,r]).to_string(index=False)); gradcam(parts[2])
if __name__=='__main__': cli()
