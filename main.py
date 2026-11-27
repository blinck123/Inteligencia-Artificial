import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados'
# Rúbrica radiológica OPCIONAL (solo si existen anotaciones reales de expertos en anotaciones.csv).
# Editar categorías según el protocolo de anotación. Si las columnas no existen o están vacías, solo se entrena el modelo base.
DIM_OPCIONAL = {'cavitacion':['ausente','presente'], 'infiltrado':['ausente','presente'], 'derrame_pleural':['ausente','presente'], 'nodulos':['ausente','presente'], 'zona_afectada':['superior','media','inferior','difusa']}
ORDINALES = []  # dimensiones ordinales (kappa cuadrático); vacío: todas son nominales/binarias
DIM = {}  # dimensiones activas; se completa con activar_rubrica()
SEED = 42

def activar_rubrica(df, estricto=True):
    """Activa las dimensiones de DIM_OPCIONAL presentes en df. Columnas totalmente vacías se ignoran;
    parcialmente completas son un error (si estricto) para no entrenar con etiquetas faltantes."""
    DIM.clear()
    for d, cats in DIM_OPCIONAL.items():
        if d not in df.columns or df[d].isna().all(): continue
        if estricto and df[d].isna().any(): raise ValueError(f'La columna {d} está parcialmente vacía: completa todas las filas o déjala vacía.')
        if not set(df[d].dropna()).issubset(cats): raise ValueError(f'Categorías inválidas en {d}: {set(df[d].dropna())-set(cats)}')
        DIM[d] = cats
    return DIM

def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

class OralDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        with Image.open(ROOT / 'dataset' / 'imagenes' / r.imagen) as im:
            x = self.transform(im.convert('RGB'))
        return x, torch.tensor(int(r.diagnostico)), torch.tensor([DIM[d].index(r[d]) for d in DIM], dtype=torch.long)

class CNN(nn.Module):
    def __init__(self, multi, pretrained=False):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        n = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.diag = nn.Linear(n,2)
        self.aux = nn.ModuleDict({d:nn.Linear(n,len(v)) for d,v in DIM.items()}) if multi else nn.ModuleDict()
    def forward(self,x):
        z = self.backbone(x)
        return self.diag(z), {d:h(z) for d,h in self.aux.items()}

def read_and_validate():
    df = pd.read_csv(ROOT/'dataset'/'anotaciones.csv', dtype={'paciente_id':str,'imagen':str})
    if df.empty: raise ValueError('Completa anotaciones.csv con datos reales.')
    required = ['imagen','paciente_id','diagnostico']
    if df[required].isna().any().any(): raise ValueError('Hay anotaciones o identificadores faltantes.')
    if df.imagen.duplicated().any(): raise ValueError('Hay nombres de imagen duplicados.')
    if set(df.diagnostico) != {0,1}: raise ValueError('Se requieren ambas clases: 0 (normal) y 1 (tuberculosis).')
    import hashlib
    hashes = {}
    conservar = []
    duplicados = []
    for idx, fila in df.iterrows():
        name = fila.imagen
        p = ROOT/'dataset'/'imagenes'/name
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        with Image.open(p) as im: im.verify()
        if h in hashes:
            anterior = hashes[h]
            if int(fila.diagnostico) != int(anterior.diagnostico):
                raise ValueError(f'Duplicado con etiquetas contradictorias: {name} y {anterior.imagen}')
            for d in DIM_OPCIONAL:
                if d in df.columns:
                    a, b = fila[d], anterior[d]
                    if not (pd.isna(a) and pd.isna(b)) and (pd.isna(a) or pd.isna(b) or a != b):
                        raise ValueError(f'Duplicado con anotaciones distintas en {d}: {name}')
            duplicados.append({'imagen_excluida': name, 'imagen_conservada': anterior.imagen, 'diagnostico': fila.diagnostico})
            continue
        hashes[h] = fila
        conservar.append(idx)
    df = df.loc[conservar].copy().reset_index(drop=True)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(duplicados, columns=['imagen_excluida','imagen_conservada','diagnostico']).to_csv(OUT/'duplicados_excluidos.csv', index=False)
    print(f'Duplicados excluidos: {len(duplicados)}. Imagenes unicas: {len(df)}')
    if set(df.diagnostico) != {0,1}: raise ValueError('Se requieren ambas clases tras depurar duplicados.')
    activar_rubrica(df)
    return df

def load_data():
    df = read_and_validate()
    # Estratificación por paciente cuando cada paciente tiene una sola clase.
    g = df.groupby('paciente_id').diagnostico
    if (g.nunique()>1).any(): raise ValueError('Pacientes con diagnósticos mixtos: definir una partición por grupos apropiada antes de continuar.')
    p = g.first().reset_index()
    a,b = train_test_split(p,test_size=.30,stratify=p.diagnostico,random_state=SEED)
    v,t = train_test_split(b,test_size=.5,stratify=b.diagnostico,random_state=SEED)
    parts = [df[df.paciente_id.isin(q.paciente_id)].copy() for q in (a,v,t)]
    for name,part in zip(['train','valid','test'],parts): part.to_csv(OUT/f'{name}.csv',index=False)
    return parts

def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    ce = nn.CrossEntropyLoss(); total=0; count=0; ys=[]; ps=[]; aux_y=[]; aux_p=[]
    with torch.set_grad_enabled(optimizer is not None):
        for x,y,r in loader:
            x,y,r=x.to(device),y.to(device),r.to(device)
            logits,aux=model(x); loss=ce(logits,y)
            if aux:
                loss=loss+.5*torch.stack([ce(aux[d],r[:,i]) for i,d in enumerate(DIM)]).mean()
                aux_y.extend(r.cpu().tolist()); aux_p.extend(torch.stack([aux[d].argmax(1) for d in DIM],1).cpu().tolist())
            if optimizer is not None:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total+=loss.item()*len(y); count+=len(y)
            ys.extend(y.cpu().tolist()); ps.extend(logits.detach().softmax(1)[:,1].cpu().tolist())
    return total/count,np.array(ys),np.array(ps),np.array(aux_y),np.array(aux_p)

def train(name,multi,parts,args):
    seed(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=CNN(multi,args.pretrained).to(device)
    loaders=[DataLoader(OralDataset(p),batch_size=args.batch_size,shuffle=(i==0),generator=torch.Generator().manual_seed(SEED)) for i,p in enumerate(parts)]
    optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
    best=-1; history=[]; path=OUT/f'{name}.pt'
    for ep in range(args.epochs):
        tr,yt,pt,_,_=run_epoch(model,loaders[0],device,optimizer)
        va,y,p,_,_=run_epoch(model,loaders[1],device)
        score=roc_auc_score(y,p)
        history.append({'epoca':ep+1,'perdida_train':tr,'perdida_valid':va,'accuracy_train':accuracy_score(yt,pt>=.5),'accuracy_valid':accuracy_score(y,p>=.5),'auc_valid':score})
        if score>best: best=score; torch.save(model.state_dict(),path)
        print(name,history[-1],flush=True)
    h=pd.DataFrame(history); h.to_csv(OUT/f'{name}_historial.csv',index=False)
    for metric in ['perdida','accuracy']:
        h.plot(x='epoca',y=[f'{metric}_train',f'{metric}_valid']); plt.tight_layout(); plt.savefig(OUT/f'{name}_{metric}.png'); plt.close()
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    _,y,p,ay,ap=run_epoch(model,loaders[2],device)
    pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    ConfusionMatrixDisplay.from_predictions(y,pred,labels=[0,1]); plt.tight_layout(); plt.savefig(OUT/f'{name}_confusion.png'); plt.close()
    pd.DataFrame({'imagen':parts[2].imagen.to_numpy(),'real':y,'probabilidad':p,'prediccion':pred}).to_csv(OUT/f'{name}_predicciones.csv',index=False)
    if multi:
        pd.DataFrame([{'dimension':d,'accuracy':accuracy_score(ay[:,i],ap[:,i]),'f1_macro':f1_score(ay[:,i],ap[:,i],average='macro',labels=list(range(len(DIM[d]))),zero_division=0)} for i,d in enumerate(DIM)]).to_csv(OUT/'rubrica_prediccion.csv',index=False)
    return {'modelo':name,'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'especificidad':tn/(tn+fp),'f1':f1_score(y,pred,zero_division=0),'auc':roc_auc_score(y,p)},y,p

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=10); parser.add_argument('--batch-size',type=int,default=8); parser.add_argument('--pretrained',action='store_true'); args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1: parser.error('Usa valores positivos')
    OUT.mkdir(exist_ok=True); parts=load_data()
    results=[]; fig,ax=plt.subplots()
    for name,multi in [('base',False)]+([('multitarea',True)] if DIM else []):
        r,y,p=train(name,multi,parts,args); results.append(r)
        RocCurveDisplay.from_predictions(y,p,name=name,ax=ax)
    fig.tight_layout(); fig.savefig(OUT/'roc_comparativa.png'); plt.close(fig)
    df=pd.DataFrame(results); df.to_csv(OUT/'comparacion.csv',index=False); print(df.to_string(index=False))
    df.set_index('modelo').T.plot.bar(rot=30,ylim=(0,1)); plt.tight_layout(); plt.savefig(OUT/'comparacion.png'); plt.close()

if __name__=='__main__': main()
