"""Repeated CV, field validation, altitude and sensor audits from t03."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from .settings import (RANDOM_SEED, CV_SPLITS, CV_REPEATS, BOOTSTRAP_REPS, PAIRWISE,
    ALTITUDE_METHODS, SENSOR_METHODS, COMMON_FIELDS, GPF, CPF, PREV, TAIL, EPS)
from .models import label_transform, predict_model, metrics, make_lr

def _run_lofo_fold(dat, yall, scheme, held_i, held, mi, name, spec, rf_jobs):
    te = dat["FieldID"].astype(str).to_numpy() == held
    tr = ~te
    train, test = dat.loc[tr], dat.loc[te]
    ytr, yte = yall[tr], yall[te]
    seed = RANDOM_SEED + 100 * held_i + mi
    p = predict_model(train, test, ytr, spec, seed, rf_jobs)
    field_row = {
        "Scheme": scheme, "HeldField": held, "Method": name,
        "N": int(te.sum()), **metrics(yte, p),
    }
    pred_rows = []
    for j, (_, row) in enumerate(test.iterrows()):
        pred_rows.append({
            "Scheme": scheme, "HeldField": held, "Method": name,
            "TreeKey": row["TreeKey"],
            "OriginalClass": int(row["Class"]),
            "True": int(yte[j]), "Pred": int(p[j]),
        })
    return pred_rows, field_row


def run_lofo(
    dat: pd.DataFrame, scheme: str, methods: Dict[str, Dict], max_workers: int
):
    yall = label_transform(dat["Class"], scheme)
    fields = sorted(dat["FieldID"].astype(str).unique())
    preds = []
    field_rows = []
    # At most one RF task per held field runs concurrently. Allocate each RF
    # task a fair CPU share so nested random-forest parallelism totals all CPUs.
    rf_jobs = max(1, max_workers // max(1, min(len(fields), max_workers)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                _run_lofo_fold, dat, yall, scheme, held_i, held,
                mi, name, spec, rf_jobs,
            )
            for held_i, held in enumerate(fields)
            for mi, (name, spec) in enumerate(methods.items())
        ]
        for fut in as_completed(futures):
            pred_rows, field_row = fut.result()
            preds.extend(pred_rows)
            field_rows.append(field_row)

    pred = pd.DataFrame(preds).sort_values(
        ["Scheme", "Method", "HeldField", "TreeKey"]
    ).reset_index(drop=True)
    fields_df = pd.DataFrame(field_rows).sort_values(
        ["Scheme", "Method", "HeldField"]
    ).reset_index(drop=True)
    summary = []

    for (scheme0, name), g in pred.groupby(["Scheme", "Method"]):
        met = metrics(g["True"], g["Pred"])
        fg = fields_df[
            (fields_df["Scheme"] == scheme0) &
            (fields_df["Method"] == name)
        ]
        spec = methods[name]
        summary.append({
            "Scheme": scheme0,
            "Method": name,
            "Representation": spec["representation"],
            "Classifier": spec["classifier"],
            "Role": spec["role"],
            "NFeatures": len(spec["features"]),
            "N": len(g),
            **met,
            "FieldUnweightedAccuracy": float(fg["Accuracy"].mean()),
            "WorstFieldAccuracy": float(fg["Accuracy"].min()),
            "BestFieldAccuracy": float(fg["Accuracy"].max()),
            "FieldAccuracySD": float(fg["Accuracy"].std(ddof=0)),
        })

    return pred, fields_df, pd.DataFrame(summary)


def _run_cv_method(dat, y, groups, scheme, rep, mi, name, spec, rf_jobs):
    seed = RANDOM_SEED + 1009 * rep
    splitter = StratifiedGroupKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=seed
    )
    oof = np.full(len(dat), -999, int)
    for fold, (tr, te) in enumerate(splitter.split(dat, y, groups), 1):
        oof[te] = predict_model(
            dat.iloc[tr], dat.iloc[te], y[tr],
            spec, seed + 100 * mi + fold, rf_jobs,
        )
    return {
        "Scheme": scheme, "Repeat": rep, "Method": name,
        "N": len(dat), **metrics(y, oof),
    }


def run_repeated_cv(
    dat: pd.DataFrame, scheme: str, methods: Dict[str, Dict], max_workers: int
):
    y = label_transform(dat["Class"], scheme)
    groups = dat["TreeKey"].astype(str).to_numpy()
    rows = []
    # Ten RF repeat jobs can overlap; give each roughly one tenth of the CPUs.
    rf_jobs = max(1, max_workers // max(1, min(CV_REPEATS, max_workers)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                _run_cv_method, dat, y, groups, scheme,
                rep, mi, name, spec, rf_jobs,
            )
            for rep in range(CV_REPEATS)
            for mi, (name, spec) in enumerate(methods.items())
        ]
        for fut in as_completed(futures):
            rows.append(fut.result())

    repdf = pd.DataFrame(rows).sort_values(
        ["Scheme", "Method", "Repeat"]
    ).reset_index(drop=True)
    summary = (
        repdf.groupby(["Scheme", "Method"])
        .agg(
            CV_Accuracy=("Accuracy", "mean"),
            CV_BalancedAccuracy=("BalancedAccuracy", "mean"),
            CV_MacroF1=("MacroF1", "mean"),
            CV_QWK=("QWK", "mean"),
            CV_MAE=("MAE", "mean"),
            CV_Within1=("Within1", "mean"),
        )
        .reset_index()
    )
    return repdf, summary


def _bootstrap_score(name, y, p):
    if name == "Accuracy":
        return accuracy_score(y, p)
    if name == "BalancedAccuracy":
        return balanced_accuracy_score(y, p)
    if name == "MacroF1":
        return f1_score(y, p, average="macro", zero_division=0)
    if name == "QWK":
        return cohen_kappa_score(y, p, weights="quadratic")
    if name == "Within1":
        return np.mean(np.abs(y - p) <= 1)
    raise ValueError(name)


def _bootstrap_one(scheme, a, b, question, metric, y, pa, pb, strata, seed):
    rng = np.random.default_rng(seed)
    point = _bootstrap_score(metric, y, pa) - _bootstrap_score(metric, y, pb)
    vals = np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        idx = np.concatenate([
            rng.choice(ix, size=len(ix), replace=True) for ix in strata
        ])
        vals[i] = (
            _bootstrap_score(metric, y[idx], pa[idx]) -
            _bootstrap_score(metric, y[idx], pb[idx])
        )
    return {
        "Scheme": scheme,
        "MethodA": a, "MethodB": b,
        "Question": question,
        "Metric": metric,
        "A_minus_B": float(point),
        "CI95_low": float(np.percentile(vals, 2.5)),
        "CI95_high": float(np.percentile(vals, 97.5)),
    }


def paired_bootstrap(preds: pd.DataFrame, max_workers: int):
    rows = []
    metric_names = ["Accuracy", "BalancedAccuracy", "MacroF1", "QWK", "Within1"]
    jobs = []

    for scheme_i, scheme in enumerate(["five_class", "merge01"]):
        s = preds[preds["Scheme"] == scheme]
        for pair_i, (a, b, question) in enumerate(PAIRWISE):
            aa = s[s["Method"] == a][
                ["TreeKey", "HeldField", "True", "Pred"]
            ].rename(columns={"Pred": "PredA"})
            bb = s[s["Method"] == b][
                ["TreeKey", "HeldField", "True", "Pred"]
            ].rename(columns={"Pred": "PredB"})
            m = aa.merge(bb, on=["TreeKey", "HeldField", "True"])
            if m.empty:
                continue
            m = m.reset_index(drop=True)
            y = m["True"].to_numpy(int)
            pa = m["PredA"].to_numpy(int)
            pb = m["PredB"].to_numpy(int)
            strata = [
                g.index.to_numpy()
                for _, g in m.groupby(["HeldField", "True"])
            ]

            for metric_i, metric in enumerate(metric_names):
                seed = RANDOM_SEED + 888 + 1000 * scheme_i + 100 * pair_i + metric_i
                jobs.append((scheme, a, b, question, metric, y, pa, pb, strata, seed))

    with ProcessPoolExecutor(max_workers=min(max_workers, len(jobs))) as ex:
        futures = [ex.submit(_bootstrap_one, *job) for job in jobs]
        for fut in as_completed(futures):
            rows.append(fut.result())
    return pd.DataFrame(rows).sort_values(
        ["Scheme", "MethodA", "MethodB", "Metric"]
    ).reset_index(drop=True)


def altitude_audit(l1: pd.DataFrame, max_workers: int):
    rows = []
    field_rows = []

    for alt in [20, 40, 60, 80, 100, 120]:
        d = l1[
            (l1["Altitude"] == alt) &
            l1["FieldID"].astype(str).isin(COMMON_FIELDS)
        ].copy()
        if d.empty:
            continue

        pred, fld, summ = run_lofo(d, "five_class", ALTITUDE_METHODS, max_workers)
        summ["Altitude"] = alt
        summ["NCommonTrees"] = len(d)
        fld["Altitude"] = alt
        rows.append(summ)
        field_rows.append(fld)

    return (
        pd.concat(rows, ignore_index=True),
        pd.concat(field_rows, ignore_index=True),
    )


def match_sensor(l1: pd.DataFrame, h300: pd.DataFrame):
    a = l1[
        (l1["Altitude"] == 120) &
        l1["FieldID"].isin(COMMON_FIELDS)
    ].copy()
    b = h300[
        (h300["Altitude"] == 120) &
        h300["FieldID"].isin(COMMON_FIELDS)
    ].copy()

    cols = ["TreeKey", "FieldID", "Class", GPF, CPF, PREV, TAIL]
    a = a[cols].drop_duplicates("TreeKey")
    b = b[cols].drop_duplicates("TreeKey")
    m = a.merge(b, on="TreeKey", suffixes=("_L1", "_H300"))
    m = m[m["Class_L1"] == m["Class_H300"]].copy()
    return m


def icc_a1(x, y):
    X = np.column_stack([x, y]).astype(float)
    X = X[np.all(np.isfinite(X), axis=1)]
    if len(X) < 3:
        return np.nan
    n, k = X.shape
    grand = X.mean()
    rm = X.mean(axis=1)
    cm = X.mean(axis=0)
    ssr = k * np.sum((rm - grand) ** 2)
    ssc = n * np.sum((cm - grand) ** 2)
    resid = X - rm[:, None] - cm[None, :] + grand
    sse = np.sum(resid ** 2)
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if abs(den) > EPS else np.nan


def sensor_feature_agreement(m: pd.DataFrame):
    rows = []
    for f, name in [
        (GPF, "Global penetration fraction"),
        (CPF, "Core penetration fraction"),
        (PREV, "Penetration prevalence"),
        (TAIL, "Local deep-tail depth"),
    ]:
        x = pd.to_numeric(m[f + "_L1"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(m[f + "_H300"], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        rho = spearmanr(x[ok], y[ok]).statistic
        rows.append({
            "Representation": name,
            "NPaired": int(ok.sum()),
            "Spearman": float(rho),
            "ICC_A1": icc_a1(x[ok], y[ok]),
            "Bias_H300_minus_L1": float(np.mean(y[ok] - x[ok])),
            "MAE": float(np.mean(np.abs(y[ok] - x[ok]))),
        })
    return pd.DataFrame(rows)


def _sensor_transfer_one(m, fields, method, spec, strategy):
    feats = spec["features"]
    yy, pp, ff = [], [], []

    for held_i, held in enumerate(fields):
        tr = m[m["FieldID_L1"].astype(str) != held].copy()
        te = m[m["FieldID_L1"].astype(str) == held].copy()
        ytr = tr["Class_L1"].astype(int).to_numpy()
        yte = te["Class_L1"].astype(int).to_numpy()

        XL1 = tr[[f + "_L1" for f in feats]].copy()
        XH = tr[[f + "_H300" for f in feats]].copy()
        TL1 = te[[f + "_L1" for f in feats]].copy()
        TH = te[[f + "_H300" for f in feats]].copy()
        XL1.columns = feats
        XH.columns = feats
        TL1.columns = feats
        TH.columns = feats

        seed = RANDOM_SEED + held_i

        if strategy == "L1_WITHIN":
            model = make_lr(seed)
            model.fit(XL1, ytr)
            pred = model.predict(TL1)

        elif strategy == "H300_WITHIN":
            model = make_lr(seed)
            model.fit(XH, ytr)
            pred = model.predict(TH)

        elif strategy == "L1_TO_H300_RAW":
            model = make_lr(seed)
            model.fit(XL1, ytr)
            pred = model.predict(TH)

        elif strategy == "L1_TO_H300_BIAS":
            THc = TH.copy()
            for f in feats:
                shift = np.nanmedian(
                    pd.to_numeric(XH[f], errors="coerce") -
                    pd.to_numeric(XL1[f], errors="coerce")
                )
                THc[f] = pd.to_numeric(THc[f], errors="coerce") - shift
            model = make_lr(seed)
            model.fit(XL1, ytr)
            pred = model.predict(THc)

        else:
            Xp = pd.concat([
                XL1.assign(SensorFlag=0.0),
                XH.assign(SensorFlag=1.0),
            ], ignore_index=True)
            yp = np.concatenate([ytr, ytr])
            T = TH.assign(SensorFlag=1.0)
            model = make_lr(seed)
            model.fit(Xp, yp)
            pred = model.predict(T)

        yy.extend(yte)
        pp.extend(pred.astype(int))
        ff.extend([held] * len(te))

    yy = np.asarray(yy, int)
    pp = np.asarray(pp, int)
    ff = np.asarray(ff, str)
    met = metrics(yy, pp)
    facc = [
        accuracy_score(yy[ff == f], pp[ff == f])
        for f in sorted(set(ff))
    ]
    return {
        "Method": method,
        "Representation": spec["representation"],
        "Strategy": strategy,
        "N": len(yy),
        **met,
        "FieldUnweightedAccuracy": float(np.mean(facc)),
        "WorstFieldAccuracy": float(np.min(facc)),
    }


def sensor_transfer(m: pd.DataFrame, max_workers: int):
    fields = sorted(m["FieldID_L1"].astype(str).unique())
    strategies = [
        "L1_WITHIN",
        "H300_WITHIN",
        "L1_TO_H300_RAW",
        "L1_TO_H300_BIAS",
        "POOLED_SENSOR_AWARE_H300",
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_sensor_transfer_one, m, fields, method, spec, strategy)
            for method, spec in SENSOR_METHODS.items()
            for strategy in strategies
        ]
        for fut in as_completed(futures):
            rows.append(fut.result())
    return pd.DataFrame(rows).sort_values(["Method", "Strategy"]).reset_index(drop=True)
