"""20-m tree-height statistics and manuscript panels; original v2 calculations."""
from __future__ import annotations

import argparse

import math

import os

from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path

from typing import Dict, List, Tuple

import numpy as np

import pandas as pd

try:
    import laspy
except Exception as e:
    laspy = None
    LASPY_IMPORT_ERROR = repr(e)
else:
    LASPY_IMPORT_ERROR = ""

try:
    from scipy.stats import gaussian_kde
except Exception as e:
    gaussian_kde = None
    SCIPY_IMPORT_ERROR = repr(e)
else:
    SCIPY_IMPORT_ERROR = ""

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

FIELD_MAP: Dict[int, str] = {1: "DKSB", 2: "PBOV", 3: "BTMS", 4: "BTMN"}

EXPECTED_RAW_COUNTS: Dict[int, int] = {1: 440, 2: 126, 3: 212, 4: 104}

HEIGHT_FOLDER_TOKEN = 20

LOW_PERCENTILE = 1.0

HIGH_PERCENTILE = 99.0

MIN_TREE_HEIGHT_M = 0.50

DEFAULT_BIN_WIDTH_M = 0.15

COLOR_HIST = "#CAD6DC"

COLOR_HIST_EDGE = "#AAB7BF"

COLOR_KDE = "#346E9E"

COLOR_MEAN = "#E69F00"

COLOR_MEDIAN = "#009E73"

COLOR_MAX = "#7B3294"

COLOR_MIN = "#D55E00"

COLOR_TEXT = "#222222"

COLOR_BOX = "white"

def set_nature_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.6,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

def parse_filename(path: Path) -> Dict:
    parts = path.stem.split("_")
    if len(parts) < 6:
        raise ValueError(f"Expected at least 6 underscore-delimited tokens, got {len(parts)}: {path.name}")

    cls = int(parts[0])
    tree_id = str(parts[1])
    field_id = int(parts[2])
    tree_type = "_".join(parts[3:-2])
    height_token_raw = parts[-2]
    sensor = str(parts[-1])

    if tree_type == "":
        raise ValueError(f"TreeType token is empty: {path.name}")
    try:
        height_token = int(height_token_raw)
    except Exception:
        raise ValueError(f"Cannot parse height token '{height_token_raw}' in {path.name}")
    if field_id not in FIELD_MAP:
        raise ValueError(f"Unexpected FieldID={field_id}; expected 1-4: {path.name}")

    return {
        "Class": cls,
        "TreeID": tree_id,
        "FieldID": field_id,
        "FieldName": FIELD_MAP[field_id],
        "TreeType": tree_type,
        "HeightToken": height_token,
        "Sensor": sensor,
    }

def extract_one(path: Path, min_height_m: float) -> Dict:
    meta = parse_filename(path)
    height_warning = "" if meta["HeightToken"] == HEIGHT_FOLDER_TOKEN else (
        f"Filename height token={meta['HeightToken']} but folder analysis expects {HEIGHT_FOLDER_TOKEN}"
    )

    las = laspy.read(path)
    z = np.asarray(las.z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 10:
        raise ValueError(f"Too few finite Z points ({len(z)}): {path.name}")

    z01 = float(np.percentile(z, LOW_PERCENTILE))
    z99 = float(np.percentile(z, HIGH_PERCENTILE))
    height = float(z99 - z01)
    if not np.isfinite(height) or height <= 0:
        raise ValueError(f"Invalid robust tree height={height}: {path.name}")

    status = "OK" if height >= min_height_m else "BELOW_MIN_HEIGHT"
    return {
        **meta,
        "FileName": path.name,
        "FilePath": str(path),
        "PointCount": int(len(z)),
        "Z01_m": z01,
        "Z99_m": z99,
        "TreeHeight_1_99_m": height,
        "Status": status,
        "HeightTokenWarning": height_warning,
    }

def summarize_group(name: str, field_id, x: pd.Series, n_raw: int, n_removed: int) -> Dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return {
            "Group": name, "FieldID": field_id, "N_raw_files": n_raw,
            "N_removed_below_0p5m": n_removed, "N_valid": 0,
            "Mean_m": np.nan, "Median_m": np.nan, "Min_m": np.nan, "Max_m": np.nan,
            "SD_m": np.nan, "Q1_m": np.nan, "Q3_m": np.nan,
        }
    return {
        "Group": name, "FieldID": field_id, "N_raw_files": n_raw,
        "N_removed_below_0p5m": n_removed, "N_valid": int(len(x)),
        "Mean_m": float(x.mean()), "Median_m": float(x.median()),
        "Min_m": float(x.min()), "Max_m": float(x.max()),
        "SD_m": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "Q1_m": float(x.quantile(0.25)), "Q3_m": float(x.quantile(0.75)),
    }

def build_summary(all_rows: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    rows.append(summarize_group(
        "Overall", "ALL", valid["TreeHeight_1_99_m"], len(all_rows),
        int((all_rows["Status"] == "BELOW_MIN_HEIGHT").sum())
    ))
    for fid in [1, 2, 3, 4]:
        raw_f = all_rows[all_rows["FieldID"] == fid]
        valid_f = valid[valid["FieldID"] == fid]
        rows.append(summarize_group(FIELD_MAP[fid], fid, valid_f["TreeHeight_1_99_m"], len(raw_f),
                                    int((raw_f["Status"] == "BELOW_MIN_HEIGHT").sum())))
    return pd.DataFrame(rows)

def common_bin_edges(values: np.ndarray, bin_width: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lo = max(MIN_TREE_HEIGHT_M, math.floor(values.min() / bin_width) * bin_width)
    hi = math.ceil(values.max() / bin_width) * bin_width
    if hi <= lo:
        hi = lo + bin_width
    edges = np.arange(lo, hi + bin_width * 1.001, bin_width)
    if len(edges) < 6:
        edges = np.linspace(lo, hi, 7)
    return edges

def scaled_kde(x: np.ndarray, xgrid: np.ndarray, bin_width: float) -> np.ndarray | None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if gaussian_kde is None or len(x) < 5 or np.nanstd(x) <= 1e-9:
        return None
    try:
        kde = gaussian_kde(x, bw_method="scott")
        return kde(xgrid) * len(x) * bin_width
    except Exception:
        return None

def save_all(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

def panel_label(ax, letter: str) -> None:
    ax.text(-0.13, 1.04, f"{letter})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=COLOR_TEXT)

def choose_stats_anchor(x: np.ndarray, counts: np.ndarray, edges: np.ndarray) -> Tuple[Tuple[float, float], str, str]:
    """Heuristic placement using mass in four corners of the plot region."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    mids = 0.5 * (edges[:-1] + edges[1:])
    ymax = max(counts.max() if len(counts) else 1, 1)
    left_mask = mids <= np.median(x)
    right_mask = ~left_mask
    upper_thresh = 0.55 * ymax

    ul = counts[left_mask][counts[left_mask] >= upper_thresh].sum() if left_mask.any() else 0
    ur = counts[right_mask][counts[right_mask] >= upper_thresh].sum() if right_mask.any() else 0
    ll = counts[left_mask][counts[left_mask] < upper_thresh].sum() if left_mask.any() else 0
    lr = counts[right_mask][counts[right_mask] < upper_thresh].sum() if right_mask.any() else 0
    candidates = [
        ((0.98, 0.98), "upper right", "right", ur),
        ((0.02, 0.98), "upper left", "left", ul),
        ((0.98, 0.42), "lower right", "right", lr),
        ((0.02, 0.42), "lower left", "left", ll),
    ]
    best = min(candidates, key=lambda t: t[3])
    return best[0], best[1], best[2]

def build_stat_handles(mean, median, xmax, xmin):
    return [
        Line2D([0], [0], color=COLOR_MEAN, ls="--", lw=1.0, label=f"Mean: {mean:.2f} m"),
        Line2D([0], [0], color=COLOR_MEDIAN, ls="--", lw=1.0, label=f"Median: {median:.2f} m"),
        Line2D([0], [0], color=COLOR_MAX, ls="--", lw=1.0, label=f"Max: {xmax:.2f} m"),
        Line2D([0], [0], color=COLOR_MIN, ls="--", lw=1.0, label=f"Min: {xmin:.2f} m"),
    ]

def draw_single_panel(x: np.ndarray, title: str, letter: str, edges: np.ndarray,
                      xlim: Tuple[float, float], out_stem: Path) -> None:
    set_nature_style()
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    fig, ax = plt.subplots(figsize=(3.0, 2.45))
    counts, _, _ = ax.hist(x, bins=edges, color=COLOR_HIST, edgecolor=COLOR_HIST_EDGE,
                           linewidth=0.45, alpha=0.82)

    bin_width = float(np.median(np.diff(edges)))
    xgrid = np.linspace(xlim[0], xlim[1], 350)
    ykde = scaled_kde(x, xgrid, bin_width)
    if ykde is not None:
        ax.plot(xgrid, ykde, color=COLOR_KDE, lw=1.45, zorder=4)

    mean = float(np.mean(x))
    median = float(np.median(x))
    xmin = float(np.min(x))
    xmax = float(np.max(x))

    for v, c in [(mean, COLOR_MEAN), (median, COLOR_MEDIAN), (xmax, COLOR_MAX), (xmin, COLOR_MIN)]:
        ax.axvline(v, color=c, ls="--", lw=1.0, zorder=5)

    ymax = max(counts.max() if len(counts) else 0, np.nanmax(ykde) if ykde is not None else 0)
    ax.set_ylim(0, max(1, ymax * 1.10))
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"Tree height, $H_{1-99}$ (m)")
    ax.set_ylabel("Frequency")
    ax.tick_params(direction="out")

    panel_label(ax, letter)
    ax.text(0.02, 1.04, f"{title}", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.8, fontweight="bold", color=COLOR_TEXT)

    anchor, loc, align = choose_stats_anchor(x, counts, edges)
    handles = build_stat_handles(mean, median, xmax, xmin)
    leg = ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor,
                    fontsize=6.3, handlelength=2.2, labelspacing=0.26,
                    borderaxespad=0.0, frameon=True)
    frame = leg.get_frame()
    frame.set_facecolor(COLOR_BOX)
    frame.set_edgecolor("none")
    frame.set_alpha(0.85)

    # Consistent x ticks across all panels.
    major_start = math.ceil(xlim[0] * 2) / 2
    major_end = math.floor(xlim[1] * 2) / 2
    xticks = np.arange(major_start, major_end + 0.001, 0.5)
    ax.set_xticks(xticks)

    save_all(fig, out_stem)

def make_figures(valid: pd.DataFrame, output_dir: Path, bin_width: float) -> None:
    all_x = valid["TreeHeight_1_99_m"].to_numpy(float)
    edges = common_bin_edges(all_x, bin_width)
    xpad = max(0.08, 0.03 * (all_x.max() - all_x.min()))
    xlim = (max(0.45, all_x.min() - xpad), all_x.max() + xpad)

    fig_dir = output_dir / "Figure2_panels"
    fig_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        ("a", "Overall", valid),
        ("b", "DKSB", valid[valid["FieldID"] == 1]),
        ("c", "PBOV", valid[valid["FieldID"] == 2]),
        ("d", "BTMS", valid[valid["FieldID"] == 3]),
        ("e", "BTMN", valid[valid["FieldID"] == 4]),
    ]

    for letter, title, g in groups:
        x = g["TreeHeight_1_99_m"].to_numpy(float)
        stem = fig_dir / f"Figure2{letter}_{title}_tree_height_distribution"
        draw_single_panel(x, title, letter, edges, xlim, stem)
