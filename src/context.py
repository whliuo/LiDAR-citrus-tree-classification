"""Optional unlinked visual-repeatability and qPCR summaries from manuscript V2."""
from __future__ import annotations
import re
import math
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, chi2
from .settings import EXPECTED_CLASSES, CONTEXT_EPS as EPS

def _read_table(path: Path, sheet_name=0) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path)


def _detect_rating_cols(df: pd.DataFrame, explicit: str | None) -> List[str]:
    if explicit:
        cols=[x.strip() for x in explicit.split(",") if x.strip()]
        missing=[c for c in cols if c not in df.columns]
        if missing: raise ValueError(f"Visual rating columns not found: {missing}")
        return cols
    # Prefer names containing rating/grade/HLB/score and then numeric content.
    cand=[]
    for c in df.columns:
        lc=str(c).lower()
        if any(k in lc for k in ["rating","grade","score","hlb"]):
            x=pd.to_numeric(df[c],errors="coerce")
            if x.notna().sum() >= max(5, int(0.5*len(df))): cand.append(c)
    if len(cand)>=3:
        return cand[:4]
    numeric=[]
    for c in df.columns:
        x=pd.to_numeric(df[c],errors="coerce")
        if x.notna().sum() >= max(5, int(0.8*len(df))): numeric.append(c)
    if len(numeric) < 3:
        raise ValueError("Could not autodetect >=3 repeated visual-rating columns; use --visual-cols")
    return numeric[:4]


def icc_a1_ak(X: np.ndarray) -> Tuple[float,float]:
    """Two-way random-effects absolute-agreement ICC(A,1) and ICC(A,k)."""
    X=np.asarray(X,float)
    X=X[np.all(np.isfinite(X),axis=1)]
    n,k=X.shape
    if n<3 or k<2: return np.nan,np.nan
    grand=X.mean(); rowm=X.mean(axis=1); colm=X.mean(axis=0)
    ssr=k*np.sum((rowm-grand)**2)
    ssc=n*np.sum((colm-grand)**2)
    resid=X-rowm[:,None]-colm[None,:]+grand
    sse=np.sum(resid**2)
    msr=ssr/(n-1); msc=ssc/(k-1); mse=sse/((n-1)*(k-1))
    den1=msr+(k-1)*mse+k*(msc-mse)/n
    icc1=(msr-mse)/den1 if abs(den1)>EPS else np.nan
    denk=msr+(msc-mse)/n
    icck=(msr-mse)/denk if abs(denk)>EPS else np.nan
    return float(icc1),float(icck)


def summarize_visual_repeats(path: Path, cols_arg: str | None, sheet_name=0):
    df=_read_table(path,sheet_name=sheet_name)
    cols=_detect_rating_cols(df,cols_arg)
    X=df[cols].apply(pd.to_numeric,errors="coerce").dropna().to_numpy(float)
    n,k=X.shape
    icc1,icck=icc_a1_ak(X)
    ranges=np.ptp(X,axis=1)
    exact=float(np.mean(ranges==0))
    within1=float(np.mean(ranges<=1))
    # Pairwise exact/within-one across all rater-round pairs.
    ex=[]; w1=[]
    for i in range(k):
        for j in range(i+1,k):
            d=np.abs(X[:,i]-X[:,j])
            ex.append(np.mean(d==0)); w1.append(np.mean(d<=1))
    summary=pd.DataFrame([{
        "NCompleteTrees":n,"NRepeatedRatings":k,"RatingColumns":";".join(cols),
        "ICC_A1":icc1,"ICC_Ak":icck,
        "AllRatingsExactFraction":exact,"AllRatingsRangeLE1Fraction":within1,
        "MeanPairwiseExactAgreement":float(np.mean(ex)),
        "MeanPairwiseWithin1Agreement":float(np.mean(w1)),
        "MedianWithinTreeRange":float(np.median(ranges)),
        "MeanWithinTreeSD":float(np.mean(np.std(X,axis=1,ddof=1))),
    }])
    dist=(pd.Series(ranges,name="RatingRange").value_counts().sort_index().rename("N").reset_index())
    dist["Fraction"]=dist["N"]/n
    # Plot-ready consensus distribution; no field/tree linkage is claimed.
    consensus=X.mean(axis=1)
    detail=pd.DataFrame({"ConsensusMean":consensus,"RatingRange":ranges,"RatingSD":np.std(X,axis=1,ddof=1)})
    return summary,dist,detail


def fisher_ci(rho: float, n: int, alpha=0.05):
    if n <= 3 or not np.isfinite(rho): return (np.nan,np.nan)
    r=np.clip(rho,-0.999999,0.999999)
    z=np.arctanh(r); se=1/math.sqrt(n-3); zcrit=1.959963984540054
    return float(np.tanh(z-zcrit*se)),float(np.tanh(z+zcrit*se))


def random_effects_fisher(rows: pd.DataFrame):
    """DerSimonian-Laird random effects on Fisher-z transformed Spearman rho."""
    g=rows[(rows["N"]>3)&rows["SpearmanRho"].notna()].copy()
    if len(g)<2:
        return pd.DataFrame()
    z=np.arctanh(np.clip(g["SpearmanRho"].to_numpy(float),-0.999999,0.999999))
    v=1/(g["N"].to_numpy(float)-3)
    w=1/v; zfix=np.sum(w*z)/np.sum(w)
    Q=np.sum(w*(z-zfix)**2); df=len(z)-1
    C=np.sum(w)-np.sum(w*w)/np.sum(w)
    tau2=max(0,(Q-df)/C) if C>0 else 0
    wr=1/(v+tau2); zr=np.sum(wr*z)/np.sum(wr); se=math.sqrt(1/np.sum(wr))
    lo,hi=zr-1.959963984540054*se,zr+1.959963984540054*se
    I2=max(0,(Q-df)/Q)*100 if Q>0 else 0
    return pd.DataFrame([{
        "KDatasets":len(g),"RandomEffectsRho":float(np.tanh(zr)),
        "CI95_low":float(np.tanh(lo)),"CI95_high":float(np.tanh(hi)),
        "Tau2_FisherZ":tau2,"Q":Q,"df":df,"I2_percent":I2,
    }])


def summarize_qpcr(path: Path, dataset_col: str, hlb_col: str, ct_col: str):
    df=_read_table(path)
    for c in [dataset_col,hlb_col,ct_col]:
        if c not in df.columns: raise ValueError(f"qPCR column not found: {c}")
    df=df[[dataset_col,hlb_col,ct_col]].copy()
    df[hlb_col]=pd.to_numeric(df[hlb_col],errors="coerce")
    df[ct_col]=pd.to_numeric(df[ct_col],errors="coerce")
    df=df.dropna()
    rows=[]
    for ds,g in df.groupby(dataset_col):
        rho,p=spearmanr(g[hlb_col],g[ct_col]) if g[hlb_col].nunique()>1 and g[ct_col].nunique()>1 else (np.nan,np.nan)
        lo,hi=fisher_ci(float(rho),len(g)) if np.isfinite(rho) else (np.nan,np.nan)
        rows.append({"Dataset":str(ds),"N":len(g),"SpearmanRho":rho,"p_value":p,"CI95_low":lo,"CI95_high":hi})
    per=pd.DataFrame(rows)
    pooled=random_effects_fisher(per)
    # Shared-domain sensitivity is dataset-name based and does not require tree ID.
    shared=df[df[dataset_col].astype(str).str.upper().str.contains("BTMS|BTMN",regex=True)]
    if len(shared) and shared[hlb_col].nunique()>1 and shared[ct_col].nunique()>1:
        rr,pp=spearmanr(shared[hlb_col],shared[ct_col]); lo,hi=fisher_ci(float(rr),len(shared))
        shared_row=pd.DataFrame([{"Scope":"BTMS+BTMN shared-domain","N":len(shared),"SpearmanRho":rr,"p_value":pp,"CI95_low":lo,"CI95_high":hi}])
    else:
        shared_row=pd.DataFrame()
    counts=df[hlb_col].value_counts().to_dict()
    # Explicitly retain class 0 with N=0 when absent. In this study class 0 denotes
    # dead trees, which cannot provide viable leaf tissue for qPCR sampling.
    levels=sorted(set(EXPECTED_CLASSES) | set(int(x) for x in counts if float(x).is_integer()))
    level_counts=pd.DataFrame({"VisualHLBLevel":levels})
    level_counts["N"]=level_counts["VisualHLBLevel"].map(lambda x:int(counts.get(x,0)))
    level_counts["Fraction"]=level_counts["N"]/len(df)
    level_counts["SamplingInterpretation"]=np.where(
        level_counts["VisualHLBLevel"]==0,
        "Class 0 denotes dead trees; viable leaf tissue is unavailable for qPCR sampling, so N=0 is expected by design",
        "Living-tree visual HLB level represented in qPCR cohort"
    )
    return per,pooled,shared_row,level_counts
