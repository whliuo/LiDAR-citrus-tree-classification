"""One implementation of the locked classifiers and evaluation metrics."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .settings import RF_TREES

def label_transform(y: Sequence[int], scheme: str) -> np.ndarray:
    y = np.asarray(y, int)
    if scheme == "five_class":
        return y.copy()
    if scheme == "merge01":
        z = y.copy()
        z[(y == 0) | (y == 1)] = 0
        z[y == 2] = 1
        z[y == 3] = 2
        z[y == 4] = 3
        return z
    raise ValueError(scheme)


def make_lr(seed: int):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def make_rf(seed: int, n_jobs: int):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=RF_TREES,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=seed,
        )),
    ])


def fit_threshold_rule(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    y = np.asarray(y, int)
    med = float(np.nanmedian(x))
    xx = np.where(np.isfinite(x), x, med)
    rho = spearmanr(xx, y).statistic if len(np.unique(y)) > 1 else -1.0
    orient = 1.0 if np.isfinite(rho) and rho >= 0 else -1.0
    z = orient * xx

    classes = np.array(sorted(np.unique(y)), int)
    centers = []
    weights = []
    for c in classes:
        vals = z[y == c]
        centers.append(float(np.median(vals)))
        weights.append(float(len(vals)))

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    centers = np.asarray(
        iso.fit_transform(classes.astype(float), centers, sample_weight=weights)
    )
    thresholds = (centers[:-1] + centers[1:]) / 2
    return {
        "median": med, "orient": orient,
        "classes": classes, "thresholds": thresholds,
    }


def predict_threshold(rule, x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    xx = np.where(np.isfinite(x), x, rule["median"])
    z = rule["orient"] * xx
    pos = np.searchsorted(rule["thresholds"], z, side="right")
    pos = np.clip(pos, 0, len(rule["classes"]) - 1)
    return rule["classes"][pos].astype(int)


def predict_model(train, test, y_train, spec, seed, rf_jobs=1):
    feats = spec["features"]
    if spec["kind"] == "THRESHOLD":
        rule = fit_threshold_rule(train[feats[0]], y_train)
        return predict_threshold(rule, test[feats[0]])
    if spec["kind"] == "LR":
        model = make_lr(seed)
    elif spec["kind"] == "RF":
        model = make_rf(seed, rf_jobs)
    else:
        raise ValueError(spec["kind"])
    model.fit(train[feats], y_train)
    return model.predict(test[feats]).astype(int)


def metrics(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, int)
    return {
        "Accuracy": float(accuracy_score(y, p)),
        "BalancedAccuracy": float(balanced_accuracy_score(y, p)),
        "MacroF1": float(f1_score(y, p, average="macro", zero_division=0)),
        "QWK": float(cohen_kappa_score(y, p, weights="quadratic")),
        "MAE": float(mean_absolute_error(y, p)),
        "Within1": float(np.mean(np.abs(y - p) <= 1)),
        "SevereErrorRate": float(np.mean(np.abs(y - p) > 1)),
    }
