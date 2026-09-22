"""Raw LAS footprint density; original chunked convex-hull calculation."""
from __future__ import annotations

import argparse

import math

import os

from pathlib import Path

from typing import Dict, List, Optional, Tuple

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
    from scipy.spatial import ConvexHull, QhullError
except Exception as e:
    ConvexHull = None
    QhullError = Exception
    SCIPY_IMPORT_ERROR = repr(e)
else:
    SCIPY_IMPORT_ERROR = ""

DEFAULT_CHUNK_SIZE = 500_000

UNIT_FACTORS_TO_M = {
    "meters": 1.0,
    "international_feet": 0.3048,
    "us_survey_feet": 1200.0 / 3937.0,
}

def determine_xy_factor_to_m(header, xy_unit: str) -> Tuple[float, str, str, bool]:
    """Return (factor_to_m, unit_name, crs_string, assumed_meters)."""
    if xy_unit != "auto":
        factor = UNIT_FACTORS_TO_M[xy_unit]
        return factor, xy_unit, "manual override", False

    # Try the CRS stored in the LAS header. laspy returns a pyproj CRS when
    # pyproj is installed and the file contains usable CRS metadata.
    try:
        crs = header.parse_crs()
    except Exception:
        crs = None

    if crs is not None:
        try:
            crs_string = crs.to_string()
        except Exception:
            crs_string = str(crs)
        try:
            axes = crs.axis_info
            if axes and len(axes) >= 1:
                unit_name = str(axes[0].unit_name or "unknown")
                factor = float(axes[0].unit_conversion_factor)
                if np.isfinite(factor) and factor > 0:
                    return factor, unit_name, crs_string, False
        except Exception:
            pass
    else:
        crs_string = ""

    # Project data have generally been processed in metric coordinates. Do not
    # silently hide the assumption: make it explicit in outputs/audit.
    return 1.0, "assumed metres", crs_string, True

def _extreme_candidates(points: np.ndarray) -> np.ndarray:
    """Fallback candidate set for degenerate chunks."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    idx = {
        int(np.argmin(points[:, 0])), int(np.argmax(points[:, 0])),
        int(np.argmin(points[:, 1])), int(np.argmax(points[:, 1])),
    }
    return points[sorted(idx)]

def chunk_hull_vertices(points: np.ndarray) -> np.ndarray:
    """Return vertices of a 2-D chunk convex hull, robust to duplicates."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points

    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except QhullError:
        # Duplicates or a degenerate local geometry may cause Qhull failure.
        # Retry with unique XY coordinates before falling back to extrema.
        try:
            u = np.unique(points, axis=0)
            if len(u) >= 3:
                hull = ConvexHull(u)
                return u[hull.vertices]
            return u
        except Exception:
            return _extreme_candidates(points)

def final_hull_area(candidate_vertices: List[np.ndarray]) -> float:
    if not candidate_vertices:
        return float("nan")
    pts = np.vstack(candidate_vertices)
    if len(pts) < 3:
        return float("nan")
    try:
        hull = ConvexHull(pts)
        # In scipy.spatial.ConvexHull for 2-D data, volume is the enclosed area;
        # area is the perimeter.
        return float(hull.volume)
    except QhullError:
        u = np.unique(pts, axis=0)
        if len(u) < 3:
            return float("nan")
        try:
            return float(ConvexHull(u).volume)
        except Exception:
            return float("nan")

def analyze_las(path: Path, input_dir: Path, chunk_size: int, xy_unit: str) -> Dict:
    with laspy.open(path) as reader:
        header = reader.header
        factor, unit_name, crs_string, assumed_meters = determine_xy_factor_to_m(header, xy_unit)

        n_valid = 0
        hull_candidates: List[np.ndarray] = []
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")

        for points in reader.chunk_iterator(chunk_size):
            x = np.asarray(points.x, dtype=float)
            y = np.asarray(points.y, dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if not finite.any():
                continue

            x = x[finite] * factor
            y = y[finite] * factor
            n_valid += int(len(x))

            xmin = min(xmin, float(x.min()))
            xmax = max(xmax, float(x.max()))
            ymin = min(ymin, float(y.min()))
            ymax = max(ymax, float(y.max()))

            xy = np.column_stack([x, y])
            hull_candidates.append(chunk_hull_vertices(xy))

        if n_valid < 3:
            raise ValueError(f"Too few valid XY points ({n_valid})")

        width_m = xmax - xmin
        height_m = ymax - ymin
        bbox_area_m2 = width_m * height_m
        hull_area_m2 = final_hull_area(hull_candidates)

        if not np.isfinite(hull_area_m2) or hull_area_m2 <= 0:
            raise ValueError(f"Invalid convex-hull area: {hull_area_m2}")
        if not np.isfinite(bbox_area_m2) or bbox_area_m2 <= 0:
            raise ValueError(f"Invalid bounding-box area: {bbox_area_m2}")

        density_hull = n_valid / hull_area_m2
        density_bbox = n_valid / bbox_area_m2

        try:
            relative = str(path.relative_to(input_dir))
        except Exception:
            relative = path.name

        return {
            "FileName": path.name,
            "RelativePath": relative,
            "PointCount_validXY": int(n_valid),
            "HeaderPointCount": int(header.point_count),
            "CRS": crs_string,
            "XY_unit_detected": unit_name,
            "XY_to_meter_factor": factor,
            "AssumedMetersBecauseUnitMissing": bool(assumed_meters),
            "FootprintWidth_m": width_m,
            "FootprintHeight_m": height_m,
            "BBoxArea_m2": bbox_area_m2,
            "ConvexHullArea_m2": hull_area_m2,
            "Hull_to_BBox_area_ratio": hull_area_m2 / bbox_area_m2,
            "PointDensity_points_m2": density_hull,
            "BBoxDensity_points_m2": density_bbox,
        }

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    density = pd.to_numeric(df["PointDensity_points_m2"], errors="coerce").dropna()
    total_points = int(pd.to_numeric(df["PointCount_validXY"], errors="coerce").sum())
    sum_hull_area = float(pd.to_numeric(df["ConvexHullArea_m2"], errors="coerce").sum())

    return pd.DataFrame([{
        "N_files": int(len(df)),
        "Total_valid_XY_points": total_points,
        "Sum_file_convex_hull_area_m2": sum_hull_area,
        "Mean_file_density_points_m2": float(density.mean()),
        "Median_file_density_points_m2": float(density.median()),
        "Min_file_density_points_m2": float(density.min()),
        "Max_file_density_points_m2": float(density.max()),
        "SD_file_density_points_m2": float(density.std(ddof=1)) if len(density) > 1 else np.nan,
        # Area-weighted pooled density across files. If raw LAS files spatially
        # overlap, this should be interpreted as pooled file-footprint density,
        # not density over the geometric union of all footprints.
        "Area_weighted_pooled_density_points_m2": (
            total_points / sum_hull_area if sum_hull_area > 0 else np.nan
        ),
    }])
