"""Optional fixed full-profile controls; no model search."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .settings import PROFILE_COLS, RANDOM_SEED, CV_SPLITS, CV_REPEATS, EXPECTED_CLASSES
from .models import make_lr, make_rf, metrics
try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    torch = nn = None
    TORCH_IMPORT_ERROR = str(exc)

CNN_MAX_EPOCHS = 70

CNN_PATIENCE = 10

CNN_LR = 1e-2

CNN_WEIGHT_DECAY = 1e-3

CNN_INNER_SPLITS = 5

def run_profile20_lr(dat: pd.DataFrame, kind='LR', workers=1):
    """Manuscript LR or V9G RF with their original fold seeds and ordering."""
    required = ["Class", "FieldID", "TreeKey"] + PROFILE_COLS
    miss = [c for c in required if c not in dat.columns]
    if miss:
        raise ValueError(f"Profile20_LR missing columns: {miss}")

    y = dat["Class"].astype(int).to_numpy()
    groups = dat["TreeKey"].astype(str).to_numpy()

    # Repeated OOF CV: same split design used in V9G.
    cv_rows = []
    for rep in range(CV_REPEATS):
        seed = RANDOM_SEED + rep * 1009
        splitter = StratifiedGroupKFold(
            n_splits=CV_SPLITS, shuffle=True, random_state=seed
        )
        oof = np.full(len(dat), -999, int)
        for fold, (tr, te) in enumerate(splitter.split(dat, y, groups), 1):
            model = make_lr(seed + fold) if kind == 'LR' else make_rf(seed + fold, workers)
            model.fit(dat.iloc[tr][PROFILE_COLS], y[tr])
            oof[te] = model.predict(dat.iloc[te][PROFILE_COLS]).astype(int)
        ok = oof != -999
        cv_rows.append({"Repeat": rep, "N": int(ok.sum()), **metrics(y[ok], oof[ok])})
    cv_rep = pd.DataFrame(cv_rows)

    # LOFO.
    pred_rows = []
    field_rows = []
    fields = sorted(dat["FieldID"].astype(str).unique())
    for i, held in enumerate(fields):
        te = dat["FieldID"].astype(str).to_numpy() == held
        tr = ~te
        # Profile20_RF was method index 32 in V9G's locked method inventory.
        model = make_lr(RANDOM_SEED + i + 1) if kind == 'LR' else make_rf(RANDOM_SEED + 32 + int(held), workers)
        model.fit(dat.loc[tr, PROFILE_COLS], y[tr])
        pred = model.predict(dat.loc[te, PROFILE_COLS]).astype(int)
        met = metrics(y[te], pred)
        field_rows.append({"HeldField": held, "N": int(te.sum()), **met})
        test = dat.loc[te]
        for pos, (idx, row) in enumerate(test.iterrows()):
            pred_rows.append({
                "RowIndex": int(idx), "TreeKey": row["TreeKey"],
                "HeldField": held, "True": int(y[te][pos]), "Pred": int(pred[pos]),
            })
    preds = pd.DataFrame(pred_rows)
    fields_df = pd.DataFrame(field_rows)
    lofo_met = metrics(preds["True"], preds["Pred"])
    lofo_met.update({
        "FieldUnweightedAccuracy": fields_df["Accuracy"].mean(),
        "WorstFieldAccuracy": fields_df["Accuracy"].min(),
        "BestFieldAccuracy": fields_df["Accuracy"].max(),
        "FieldAccuracySD": fields_df["Accuracy"].std(ddof=0),
    })
    summary = {
        "Method": f"Profile20_{kind}", "N": len(preds), **lofo_met,
        "Accuracy_Mean": cv_rep["Accuracy"].mean(),
        "BalancedAccuracy_Mean": cv_rep["BalancedAccuracy"].mean(),
        "MacroF1_Mean": cv_rep["MacroF1"].mean(),
        "QWK_Mean": cv_rep["QWK"].mean(),
        "CV_minus_LOFO_Accuracy": cv_rep["Accuracy"].mean() - lofo_met["Accuracy"],
        "NFeatures": 20,
        "Features": ";".join(PROFILE_COLS),
        "Role": f"full-profile representation control with the locked {kind} classifier",
    }
    return pd.DataFrame([summary]), cv_rep, fields_df, preds


class Profile20CNN(nn.Module if nn is not None else object):
    """Small fixed 1-D CNN for the 20-bin normalized vertical return profile."""
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),       # 20 -> 10
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(5),           # fixed 32 x 5 latent map
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(32 * 5, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _seed_torch(seed: int):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        try:
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        except Exception:
            pass


def _profile_imputer_fit(X: np.ndarray) -> np.ndarray:
    X=np.asarray(X,float)
    med=np.nanmedian(X,axis=0)
    med[~np.isfinite(med)]=0.0
    return med


def _profile_imputer_apply(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    X=np.asarray(X,float).copy()
    bad=~np.isfinite(X)
    if bad.any():
        X[bad]=np.take(med,np.where(bad)[1])
    # The locked p1...p20 variables are bin proportions. Numerical drift is
    # harmless, but non-negative clipping keeps the physical input domain valid.
    X=np.maximum(X,0.0)
    return X.astype(np.float32)


def _balanced_class_weights(y: np.ndarray, n_classes: int = 5) -> np.ndarray:
    y=np.asarray(y,int)
    counts=np.bincount(y,minlength=n_classes).astype(float)
    w=np.zeros(n_classes,float)
    present=counts>0
    w[present]=len(y)/(present.sum()*counts[present])
    return w.astype(np.float32)


def _train_cnn_epochs(X: np.ndarray, y: np.ndarray, seed: int, epochs: int,
                      X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                      patience: int | None = None):
    """Train fixed CNN with full-batch updates; validation, when supplied, selects epoch only."""
    if torch is None:
        raise RuntimeError("PyTorch is unavailable: " + TORCH_IMPORT_ERROR)
    _seed_torch(seed)
    device=torch.device("cpu")
    model=Profile20CNN(n_classes=len(EXPECTED_CLASSES)).to(device)
    weights=torch.tensor(_balanced_class_weights(y,len(EXPECTED_CLASSES)),dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=weights)
    optim=torch.optim.AdamW(model.parameters(),lr=CNN_LR,weight_decay=CNN_WEIGHT_DECAY)

    xt=torch.tensor(X[:,None,:],dtype=torch.float32,device=device)
    yt=torch.tensor(np.asarray(y,int),dtype=torch.long,device=device)
    xv=yv=None
    if X_val is not None and y_val is not None:
        xv=torch.tensor(X_val[:,None,:],dtype=torch.float32,device=device)
        yv=torch.tensor(np.asarray(y_val,int),dtype=torch.long,device=device)

    best_loss=np.inf; best_epoch=epochs; stale=0
    for epoch in range(1,epochs+1):
        model.train()
        optim.zero_grad(set_to_none=True)
        loss=criterion(model(xt),yt)
        loss.backward()
        optim.step()
        if xv is not None:
            model.eval()
            with torch.no_grad():
                vloss=float(criterion(model(xv),yv).cpu())
            if vloss < best_loss - 1e-5:
                best_loss=vloss; best_epoch=epoch; stale=0
            else:
                stale+=1
            if patience is not None and stale>=patience:
                break
    return model,int(best_epoch),float(best_loss) if np.isfinite(best_loss) else np.nan


def _fit_cnn_outer(train: pd.DataFrame, y_train: np.ndarray, groups_train: np.ndarray, seed: int):
    """Nested early stopping entirely within the outer training data, then refit all outer training data."""
    Xraw=train[PROFILE_COLS].to_numpy(float)
    y_train=np.asarray(y_train,int)
    groups_train=np.asarray(groups_train,str)

    # Inner split is fixed and used only to choose epoch count; no architecture or
    # hyperparameter search is performed.
    inner=StratifiedGroupKFold(n_splits=CNN_INNER_SPLITS,shuffle=True,random_state=seed+991)
    itr,iva=next(inner.split(Xraw,y_train,groups_train))
    med_inner=_profile_imputer_fit(Xraw[itr])
    Xi=_profile_imputer_apply(Xraw[itr],med_inner)
    Xv=_profile_imputer_apply(Xraw[iva],med_inner)
    _,best_epoch,_=_train_cnn_epochs(
        Xi,y_train[itr],seed+17,CNN_MAX_EPOCHS,Xv,y_train[iva],CNN_PATIENCE
    )
    # Guard against very short stochastic stops, then refit from scratch using ALL
    # outer-training trees for the selected number of epochs.
    best_epoch=max(8,min(int(best_epoch),CNN_MAX_EPOCHS))
    med_full=_profile_imputer_fit(Xraw)
    Xfull=_profile_imputer_apply(Xraw,med_full)
    model,_,_=_train_cnn_epochs(Xfull,y_train,seed+31,best_epoch)
    return model,med_full,best_epoch


def _predict_cnn(model, test: pd.DataFrame, med: np.ndarray) -> np.ndarray:
    X=_profile_imputer_apply(test[PROFILE_COLS].to_numpy(float),med)
    model.eval()
    with torch.no_grad():
        logits=model(torch.tensor(X[:,None,:],dtype=torch.float32))
        return torch.argmax(logits,dim=1).cpu().numpy().astype(int)


def run_profile20_cnn(dat: pd.DataFrame):
    """Fixed Profile20 CNN with the same outer repeated group CV + LOFO design."""
    if torch is None:
        raise RuntimeError("Profile20_CNN requested but PyTorch unavailable: " + TORCH_IMPORT_ERROR)
    required=["Class","FieldID","TreeKey"]+PROFILE_COLS
    miss=[c for c in required if c not in dat.columns]
    if miss:
        raise ValueError(f"Profile20_CNN missing columns: {miss}")

    y=dat["Class"].astype(int).to_numpy()
    groups=dat["TreeKey"].astype(str).to_numpy()
    cv_rows=[]
    for rep in range(CV_REPEATS):
        seed=RANDOM_SEED+rep*1009
        outer=StratifiedGroupKFold(n_splits=CV_SPLITS,shuffle=True,random_state=seed)
        oof=np.full(len(dat),-999,int); epoch_log=[]
        for fold,(tr,te) in enumerate(outer.split(dat,y,groups),1):
            model,med,ep=_fit_cnn_outer(dat.iloc[tr],y[tr],groups[tr],seed+fold*101)
            oof[te]=_predict_cnn(model,dat.iloc[te],med)
            epoch_log.append(ep)
        ok=oof!=-999
        cv_rows.append({"Repeat":rep,"N":int(ok.sum()),"MeanSelectedEpoch":float(np.mean(epoch_log)),**metrics(y[ok],oof[ok])})
    cv_rep=pd.DataFrame(cv_rows)

    pred_rows=[]; field_rows=[]
    fields=sorted(dat["FieldID"].astype(str).unique())
    for i,held in enumerate(fields):
        te=dat["FieldID"].astype(str).to_numpy()==held; tr=~te
        model,med,ep=_fit_cnn_outer(dat.loc[tr],y[tr],groups[tr],RANDOM_SEED+5000+i*211)
        pred=_predict_cnn(model,dat.loc[te],med)
        met=metrics(y[te],pred)
        field_rows.append({"HeldField":held,"N":int(te.sum()),"SelectedEpoch":ep,**met})
        test=dat.loc[te]
        yy=y[te]
        for pos,(idx,row) in enumerate(test.iterrows()):
            pred_rows.append({"RowIndex":int(idx),"TreeKey":row["TreeKey"],"HeldField":held,
                              "True":int(yy[pos]),"Pred":int(pred[pos])})
    preds=pd.DataFrame(pred_rows); fields_df=pd.DataFrame(field_rows)
    lofo_met=metrics(preds["True"],preds["Pred"])
    lofo_met.update({
        "FieldUnweightedAccuracy":fields_df["Accuracy"].mean(),
        "WorstFieldAccuracy":fields_df["Accuracy"].min(),
        "BestFieldAccuracy":fields_df["Accuracy"].max(),
        "FieldAccuracySD":fields_df["Accuracy"].std(ddof=0),
    })
    summary={
        "Method":"Profile20_CNN","N":len(preds),**lofo_met,
        "Accuracy_Mean":cv_rep["Accuracy"].mean(),
        "BalancedAccuracy_Mean":cv_rep["BalancedAccuracy"].mean(),
        "MacroF1_Mean":cv_rep["MacroF1"].mean(),
        "QWK_Mean":cv_rep["QWK"].mean(),
        "CV_minus_LOFO_Accuracy":cv_rep["Accuracy"].mean()-lofo_met["Accuracy"],
        "NFeatures":20,"Features":";".join(PROFILE_COLS),
        "Role":"fixed 1-D CNN benchmark on the same 20-bin full vertical return profile",
        "Architecture":"Conv1D(16)-Conv1D(32)-Pool-Conv1D(32)-AdaptivePool-Dropout-Linear",
        "NoHyperparameterSearch":True,
    }
    return pd.DataFrame([summary]),cv_rep,fields_df,preds


def run_controls(l1, out, workers, cnn=False):
    from .io_utils import require_fields, write_run_record
    dat = l1[(l1['Source'] == 'L1') & (l1['Altitude'] == 60)].copy()
    # V9D sorted these columns before the manuscript/V9G profile benchmarks.
    dat = dat.sort_values(['FieldID','ID','SensorToken']).reset_index(drop=True)
    require_fields(dat, ['1','2','3','4'], 'Profile20 controls')
    dest = out / 'profiles'
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in ['LR','RF'] + (['CNN'] if cnn else []):
        print(f'Running fixed Profile20_{kind}', flush=True)
        result = run_profile20_cnn(dat) if kind == 'CNN' else run_profile20_lr(dat, kind, workers)
        summary, cv, fields, predictions = result
        for name, frame in zip(['audit','cv','field_metrics','predictions'], result):
            frame.to_csv(dest / f'profile20_{kind.lower()}_{name}.csv', index=False)
        row = summary.iloc[0].to_dict()
        row.update(Scheme='five_class', Representation='20-bin vertical return profile',
                   Classifier=kind, ResultSource='Recomputed profile control')
        for metric in ['Accuracy','BalancedAccuracy','MacroF1','QWK']:
            row['CV_' + metric] = row.pop(metric + '_Mean')
        row['CV_MAE'] = row['CV_Within1'] = np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(dest / 'profile_benchmarks.csv', index=False)
    write_run_record(dest, 'profiles', details={'cnn':cnn,'seed':RANDOM_SEED,'cv_repeats':CV_REPEATS})
