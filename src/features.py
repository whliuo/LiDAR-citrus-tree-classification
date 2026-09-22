"""Midpoint features from t02; depth-profile bins from corrected V9D."""
from __future__ import annotations
import re
from pathlib import Path
import laspy
import numpy as np
from .settings import (THRESHOLDS, LOW_Z_PERCENTILE, HIGH_Z_PERCENTILE,
    XY_LOW_PERCENTILE, XY_HIGH_PERCENTILE, GRID_N, GRID_MIN_POINTS,
    GRID_MIN_VALID_CELLS, PGF_LOCAL_THRESHOLD, CIRCLE_CORE_C, CORE_MIN_POINTS,
    COLUMN_RADIAL_BANDS, COLUMN_ANGULAR_SECTORS, COLUMN_MIN_POINTS, COLUMN_MIN_VALID, EPS)

def profile20(d: np.ndarray) -> Dict[str, float]:
    # p1 is top layer (d 0-.05), p20 is deepest layer (.95-1.0).
    hist, _ = np.histogram(d, bins=20, range=(0.0, 1.0))
    total = hist.sum()
    if total <= 0:
        return {f"p{i}": np.nan for i in range(1, 21)}
    frac = hist / total
    return {f"p{i+1}": float(frac[i]) for i in range(20)}

def canon_field(x) -> str:
    s = str(x).strip()
    s = re.sub(r"(?i)^field", "", s).strip()
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    return s


def parse_filename(path: Path):
    parts = path.stem.split("_")
    if len(parts) < 6:
        raise ValueError(f"Unexpected filename: {path.name}")
    cls = int(parts[0])
    tree_id = str(parts[1])
    field = canon_field(parts[2])
    tree_type = "_".join(parts[3:-2])
    alt = int(parts[-2])
    sensor = parts[-1]
    if cls not in range(5) or field not in {'1', '2', '3', '4'} or not tree_type:
        raise ValueError(f'Unexpected finalized class, field or type: {path.name}')
    return cls, tree_id, field, tree_type, alt, sensor


def robust_pca_xy(x: np.ndarray, y: np.ndarray):
    xy = np.column_stack([x, y]).astype(float)
    center0 = np.nanmedian(xy, axis=0)
    centered0 = xy - center0

    r0 = np.sqrt(np.sum(centered0 ** 2, axis=1))
    cut = np.nanpercentile(r0, 99.0) if len(r0) >= 20 else np.nanmax(r0)
    keep = np.isfinite(r0) & (r0 <= cut)
    if keep.sum() < 10:
        keep = np.isfinite(r0)

    cov = np.cov(centered0[keep].T)
    try:
        vals, vecs = np.linalg.eigh(cov)
        vecs = vecs[:, np.argsort(vals)[::-1]]
    except Exception:
        vecs = np.eye(2)

    uv = centered0 @ vecs
    qlo = np.nanpercentile(uv, XY_LOW_PERCENTILE, axis=0)
    qhi = np.nanpercentile(uv, XY_HIGH_PERCENTILE, axis=0)
    shift = (qlo + qhi) / 2.0
    uv2 = uv - shift
    half = np.maximum((qhi - qlo) / 2.0, 1e-6)

    if half[1] > half[0]:
        uv2 = uv2[:, ::-1]
        half = half[::-1]

    a, b = float(half[0]), float(half[1])
    rho = np.sqrt((uv2[:, 0] / a) ** 2 + (uv2[:, 1] / b) ** 2)
    theta = np.mod(np.arctan2(uv2[:, 1] / b, uv2[:, 0] / a), 2*np.pi)
    return {
        "u": uv2[:,0], "v": uv2[:,1], "rho": rho, "theta": theta,
        "length": 2*a, "width": 2*b,
    }


def pf(d, t):
    return float(np.mean(d >= t)) if len(d) else np.nan


def grid_features(u, v, d):
    out = {}
    for t in THRESHOLDS:
        out[f"TA{int(t*100):02d}"] = np.nan
        out[f"PGF{int(t*100):02d}"] = np.nan
    out["GridValidCells"] = 0

    if len(d) < GRID_MIN_POINTS * GRID_MIN_VALID_CELLS:
        return out

    ulo, uhi = np.nanpercentile(u, [XY_LOW_PERCENTILE, XY_HIGH_PERCENTILE])
    vlo, vhi = np.nanpercentile(v, [XY_LOW_PERCENTILE, XY_HIGH_PERCENTILE])
    if not (uhi > ulo and vhi > vlo):
        return out

    m = (u >= ulo) & (u <= uhi) & (v >= vlo) & (v <= vhi)
    uu, vv, dd = u[m], v[m], d[m]
    if len(dd) < GRID_MIN_POINTS * GRID_MIN_VALID_CELLS:
        return out

    iu = np.floor((uu - ulo)/(uhi-ulo+EPS)*GRID_N).astype(int)
    iv = np.floor((vv - vlo)/(vhi-vlo+EPS)*GRID_N).astype(int)
    iu = np.clip(iu, 0, GRID_N-1)
    iv = np.clip(iv, 0, GRID_N-1)

    local = {t: [] for t in THRESHOLDS}
    valid = 0
    for a in range(GRID_N):
        for b in range(GRID_N):
            mm = (iu == a) & (iv == b)
            if mm.sum() >= GRID_MIN_POINTS:
                valid += 1
                for t in THRESHOLDS:
                    local[t].append(float(np.mean(dd[mm] >= t)))

    out["GridValidCells"] = valid
    if valid < GRID_MIN_VALID_CELLS:
        return out

    for t in THRESHOLDS:
        arr = np.asarray(local[t], float)
        tag = int(t*100)
        out[f"TA{tag:02d}"] = float(np.mean(arr))
        out[f"PGF{tag:02d}"] = float(np.mean(arr >= PGF_LOCAL_THRESHOLD))
    return out


def cpc90_feature(rho, theta, d):
    m = np.isfinite(rho) & np.isfinite(theta) & np.isfinite(d) & (rho <= 1.0)
    rr, tt, dd = rho[m], theta[m], d[m]
    if len(dd) < COLUMN_MIN_POINTS * COLUMN_MIN_VALID:
        return np.nan

    bounds = np.sqrt(np.linspace(0.0, 1.0, COLUMN_RADIAL_BANDS + 1))
    rb = np.searchsorted(bounds, rr, side="right") - 1
    rb = np.clip(rb, 0, COLUMN_RADIAL_BANDS - 1)
    ab = np.floor(tt/(2*np.pi)*COLUMN_ANGULAR_SECTORS).astype(int)
    ab = np.clip(ab, 0, COLUMN_ANGULAR_SECTORS - 1)

    q90 = []
    for i in range(COLUMN_RADIAL_BANDS):
        for j in range(COLUMN_ANGULAR_SECTORS):
            mm = (rb == i) & (ab == j)
            if mm.sum() >= COLUMN_MIN_POINTS:
                q90.append(float(np.percentile(dd[mm], 90)))

    if len(q90) < COLUMN_MIN_VALID:
        return np.nan
    return float(np.median(q90))


def extract_one(task):
    path_str, folder_alt, source = task
    path = Path(path_str)
    cls, tree_id, field, tree_type, alt_token, sensor_name = parse_filename(path)
    if alt_token != folder_alt:
        raise ValueError(f'Filename/folder altitude mismatch: {path.name}')
    accepted = {'L1': {'1', 'L1'}, 'H300': {'2', 'H300'}}
    if sensor_name.upper() not in accepted[source]:
        raise ValueError(f'Sensor token does not agree with folder: {path.name}')

    las = laspy.read(path)
    x = np.asarray(las.x, float)
    y = np.asarray(las.y, float)
    z = np.asarray(las.z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]

    if len(z) < 50:
        raise ValueError(f"Too few finite points: {len(z)}")

    zlo = float(np.percentile(z, LOW_Z_PERCENTILE))
    zhi = float(np.percentile(z, HIGH_Z_PERCENTILE))
    if not zhi > zlo:
        raise ValueError("Degenerate vertical range")

    keep = (z >= zlo) & (z <= zhi)
    x, y, z = x[keep], y[keep], z[keep]
    h = (z-zlo)/(zhi-zlo)
    d = 1.0 - h

    geom = robust_pca_xy(x,y)
    u, v = np.asarray(geom["u"]), np.asarray(geom["v"])
    rho, theta = np.asarray(geom["rho"]), np.asarray(geom["theta"])

    Dmean = (float(geom["length"]) + float(geom["width"])) / 2.0
    r = np.sqrt(u*u + v*v)
    mc = r <= CIRCLE_CORE_C * Dmean

    row = {
        "Source": source,
        "File": str(path),
        "Class": int(cls),
        "ID": str(tree_id),
        "FieldID": field,
        "Type": str(tree_type),
        "Altitude": int(folder_alt),
        "SensorToken": sensor_name,
        "TreeKey": f"{field}|{tree_id}|{tree_type}",
        "PointCount": int(len(d)),
        "TreeHeight": float(zhi-zlo),
        "CorePointCount": int(mc.sum()),
    }

    # Legacy VRI.
    upper = float(np.mean(h >= 0.70))
    lower = float(np.mean(h <= 0.30))
    row["VRI"] = (lower + 1e-6)/(upper + 1e-6)

    # Whole/core penetration fractions.
    for t in THRESHOLDS:
        tag = int(t*100)
        row[f"GlobalPF{tag:02d}"] = pf(d,t)
        row[f"CorePF{tag:02d}"] = pf(d[mc],t) if mc.sum() >= CORE_MIN_POINTS else np.nan

    row.update(grid_features(u,v,d))
    row["CPC90"] = cpc90_feature(rho,theta,d)

    row.update(profile20(d))
    return row
