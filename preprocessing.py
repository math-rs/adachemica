# SPDX-License-Identifier: MIT
# Copyright (c) 2025
# Matheus Santos (ORCID: 0000-0002-1604-381X)

"""
AdaCHEMICA — Preprocessing (and Visualization)
=====================================================================

Overview
--------
Prepares compositional mineral datasets (filtering, balancing, engineered features) and
produces spiral plots of class size distributions. Supports single-step modes and a combined
"all" mode that runs plotting and preprocessing in sequence.

What this script provides
-------------------------
- Spiral Plot (--mode spiral)
  Generates high-resolution spiral plots of sample counts by class (PNG/TIFF/PDF).
  Optionally highlights series/groups/polymorphs when a DOCX with bolded names is provided.

- Dataset Preparation (--mode preprocess)
  Performs robust, iterative Mahalanobis outlier filtering (MinCovDet) per class with
  gap-based adaptive thresholds and diagnostics (CSV + plots), KMeans undersampling for
  large classes, SMOTE oversampling for minorities (to a common target size where feasible),
  and computes engineered features (e.g., value of highest constituent, number of major
  constituents >1% wt). Outputs a balanced, traceable dataset to Excel.

- Combined Workflow (--mode all)
  Runs the spiral plot and then the preprocessing pipeline in a single invocation, writing
  all artifacts under the specified --outdir.

Quick start
-----------
# Spiral plot only
python preprocessing.py \
  --mode spiral \
  --input data/mineral_data.xlsx \
  --sheet data \
  --label label \
  --docx docs/classes.docx \
  --outdir outputs/run_spiral

# Preprocess (filter + balance + engineered features)
python preprocessing.py \
  --mode preprocess \
  --input data/mineral_data.xlsx \
  --sheet data \
  --label label \
  --outdir outputs/run_prep \
  --output outputs/run_prep/datasets/mineral_data_balanced.xlsx

# Run both (spiral → preprocess) in one command
python preprocessing.py \
  --mode all \
  --input data/mineral_data.xlsx \
  --sheet data \
  --label label \
  --docx docs/classes.docx \
  --outdir outputs/run_all \
  --output outputs/run_all/datasets/mineral_data_balanced.xlsx

Inputs
------
- Excel worksheet with chemistry columns (oxides/halogens) and a target 'label'.
- Optional DOCX with bolded names to drive hierarchical highlighting in spiral plots.

Outputs (under --outdir)
------------------------
- plots/
    - spiral/        : spiral distributions (PNG/TIFF/PDF)
    - mahalanobis/   : Mahalanobis diagnostics per class and round (PNG)
- diagnostics/       : CSV tables supporting diagnostics
- datasets/          : balanced Excel datasets (XLSX)
- logs/
    - run.log        : execution log (INFO)
- ENVIRONMENT.txt    : Python/OS/package versions snapshot

Reproducibility
---------------
- Fixed random seeds for Python/NumPy.
- Threaded BLAS limited via OMP_NUM_THREADS and threadpoolctl.
- High-DPI, white-background figures for consistent rendering.

Dependencies
------------
Python 3.9+; major libraries: numpy, pandas, scikit-learn, imbalanced-learn, matplotlib, scipy, openpyxl, threadpoolctl.

License & Citation
------------------
- MIT License (see SPDX header / LICENSE file).
- Please cite the associated paper when available.

References
----------
- Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator. Technometrics, 41(3), 212-223.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In L. M. Le Cam & J. Neyman (Eds.), Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability (Vol. 1, pp. 281-297). University of California Press.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import random
import re
import shlex
import sys
import time
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

# Suppress scikit-learn FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Plotting backend for headless environments / CI
import matplotlib

matplotlib.use("Agg")

# Core + needed for dump_environment
import scipy
import sklearn
import joblib
import imblearn
import threadpoolctl

# Scientific and plotting
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.linalg import pinvh
from scipy.special import expit
from sklearn.covariance import MinCovDet
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from threadpoolctl import threadpool_limits

# -----------------------------
# Reproducibility & Environment
# -----------------------------
GLOBAL_SEED = 42
os.environ["OMP_NUM_THREADS"] = "1"  # helps Windows stability
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# -----------------------------
# Hyperparameters
# -----------------------------
KMEANS_CLUSTERS = 10               # number of clusters to keep diversity in undersampling
TARGET_PER_CLASS = 50              # target size per class after balancing
MAX_EXCLUSION_FRAC = 0.05          # max fraction of outliers per class (5%)
GAP_RATIO_THRESH = 1.25            # "expressive" gap must be ≥ 1.25× next-largest
TAIL_START_FRAC = 0.10             # ignore lowest 10% of distances (more stable)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------
# Utilities
# -----------------------------
def truncate_colormap(cmap, min_val=0.05, max_val=0.85, n_colors=10):
    """Return a truncated copy of a Matplotlib colormap.

    Parameters
    ----------
    cmap : Colormap
        Base colormap to truncate.
    min_val, max_val : float
        Lower/upper bounds in [0, 1] to slice the colormap.
    n_colors : int
        Number of discrete colors to sample.

    Returns
    -------
    ListedColormap
        Truncated, discrete colormap with 'n_colors' entries.
    """
    new_colors = cmap(np.linspace(min_val, max_val, n_colors))
    return ListedColormap(new_colors)


def safe_logspace_ticks(vmin: float, vmax: float, candidates=(1, 10, 100, 1000, 10000)):
    """Compute readable log-scale tick candidates within [vmin, vmax].

    Notes
    -----
    Ensures 'vmin'≥1 to avoid log(0); pads ends if missing.
    """
    vmin = max(vmin, 1)  # avoid 0 in log scale
    ticks = [v for v in candidates if vmin <= v <= vmax]
    if vmin not in ticks:
        ticks.insert(0, vmin)
    if vmax not in ticks:
        ticks.append(vmax)
    ticks = sorted(set(ticks))
    return ticks


def format_subscripts(text: str) -> str:
    """
    Convert digits after letters/parentheses into Unicode subscripts.
    E.g., 'SiO2'→'SiO₂', 'TiO2'→'TiO₂'.
    """
    subs_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return re.sub(
        r"(?<=[A-Za-z\)])(\d+)", lambda m: m.group(1).translate(subs_map), text
    )


def validate_columns(
    df: pd.DataFrame, required: list[str], kind: str = "required"
) -> None:
    """Raise a clear error if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {kind} columns: {missing}")


def dump_environment(out_dir: Path) -> None:
    """Persist environment metadata for reproducibility (packages, OS, Python)."""
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
    }

    # Engines and extra libraries used in preprocessing
    try:
        import openpyxl  # used by pandas to read .xlsx

        info["openpyxl"] = openpyxl.__version__
    except Exception:
        info["openpyxl"] = "not installed"

    try:
        import imblearn  # SMOTE

        info["imbalanced_learn"] = imblearn.__version__
    except Exception:
        info["imbalanced_learn"] = "not installed"

    try:
        import threadpoolctl

        info["threadpoolctl"] = threadpoolctl.__version__
    except Exception:
        info["threadpoolctl"] = "not installed"

    (out_dir / "ENVIRONMENT.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in info.items()), encoding="utf-8"
    )
    logging.info("Environment written to %s", out_dir / "ENVIRONMENT.txt")


def log_run_header(args, outdir: Path, element_columns: list[str], metadata_cols: list[str]) -> None:
    """Echo the CLI, resolved params, and save a JSON snapshot to diagnostics/."""
    try:
        cmd = "python " + shlex.join(sys.argv)
    except Exception:
        cmd = " ".join(["python"] + sys.argv)
        
    input_path = Path(args.input)

    default_output = (outdir / "datasets" / f"{input_path.stem}_balanced.xlsx") \
        if args.mode in ("preprocess", "all") else None
    effective_output = Path(args.output) if args.output else default_output

    spiral_prefix = (outdir / "plots" / "spiral" / "spiral_distribution_highres") \
        if args.mode in ("spiral", "all") else None

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": cmd,
        "cwd": os.getcwd(),
        "python": sys.executable,
        "mode": args.mode,
        "input": str(args.input),
        "sheet": args.sheet,
        "label": args.label,
        "docx": args.docx,
        "output_cli": args.output,
        "output_effective": str(effective_output) if effective_output else None,
        "spiral_out_prefix": str(spiral_prefix) if spiral_prefix else None,
        "dpi": args.dpi,
        "elements": element_columns,
        "metadata": metadata_cols,
        "outdir": str(outdir),
        "seed": GLOBAL_SEED,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
    }

    logging.info("CLI: %s", cmd)
    logging.info("Resolved parameters:\n%s", json.dumps(payload, indent=2))

    diag_dir = outdir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    (diag_dir / "run_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Saved parameters snapshot to %s", diag_dir / "run_params.json")


# -----------------------------
# Spiral Plot of Class Sizes (raw input)
# -----------------------------
def parse_bold_classes_from_docx(docx_path: str | Path) -> set[str]:
    """Return a set of class names that appear bold in the DOCX (optional)."""
    docx_path = Path(docx_path)
    if not docx_path or not docx_path.exists():
        return set()

    bold_classes = set()
    with zipfile.ZipFile(docx_path) as docx:
        xml_content = docx.read("word/document.xml")
        root = ET.fromstring(xml_content)

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    current_class, inside_bold = [], False

    def clean_class_name(name):
        name = re.sub(r"\s*\(\d+\)\s*$", "", name)
        name = re.sub(r"\s*-\s*", "-", name)
        name = re.sub(r"\(\s*([^)]+?)\s*\)", r"(\1)", name)
        name = re.sub(r"(?<=\w)\s+(?=\d)", "", name)
        name = re.sub(r"(?<=\d)\s+(?=\w)", "", name)
        name = re.sub(r"\s*\)\s*", ")", name)
        name = re.sub(r"\s*\(\s*", "(", name)
        name = re.sub(r"(?<=[0-9A-Za-z)])(?=polymorphs\b)", " ", name)
        return name.strip(",; ")

    for run in root.findall(".//w:r", ns):
        text_el = run.find("w:t", ns)
        if text_el is None:
            continue
        text = (text_el.text or "").strip()
        if not text:
            continue

        bold = run.find("w:rPr/w:b", ns) is not None

        if bold:
            current_class.append(text)
            inside_bold = True
        else:
            if inside_bold:
                class_name = clean_class_name(" ".join(current_class))
                if class_name:
                    bold_classes.add(class_name)
                current_class, inside_bold = [], False

    if inside_bold and current_class:
        class_name = clean_class_name(" ".join(current_class))
        if class_name:
            bold_classes.add(class_name)

    return bold_classes


def normalize_label(name: str) -> str:
    """Lowercase + whitespace/hyphen normalization for class labels."""
    name = name.lower()
    name = re.sub(r"\s*-\s*", "-", name)
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\d+$", "", name)
    return name.strip()


def smart_abbreviation(mineral: str) -> str:
    """Append 'poly.' to 'polymorphs' for clarity."""
    if "polymorph" in mineral.lower():
        return mineral.split()[0] + " poly."
    return mineral


def wrap_hyphenated(text: str, threshold: int = 20) -> str:
    """Break long hyphenated mineral names for readability."""
    if "-" in text and len(text) > threshold:
        return text.replace("-", "-\n")
    return text


def plot_spiral_distribution(
    input_path: Path,
    sheet: str,
    label_col: str,
    docx_path: str | Path | None,
    out_prefix: Path,
    dpi: int = 600,
) -> None:
    """Spiral plots of per-class sample counts with optional DOCX-driven highlighting.

    Notes:
        - Colors reflect class size on a log scale; donut legend shows ticks.
        - 4×3 grid with 11 spiral panels; last cell reserved for the donut legend.
    """
    logging.info("\n\nGenerating Spiral Plot of Class Sizes from raw input data...\n")
    df = pd.read_excel(input_path, sheet_name=sheet)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in input data.")

    class_counts = df[label_col].value_counts()
    classes = class_counts.index.to_numpy()
    sizes = class_counts.values

    # Sort ascending for grouping
    sorted_idx = np.argsort(sizes)
    classes, sizes = classes[sorted_idx], sizes[sorted_idx]

    # Optional bold groups/series/polymorphs from DOCX
    bold_raw = parse_bold_classes_from_docx(docx_path) if docx_path else set()
    normalized_bold = {normalize_label(x) for x in bold_raw}

    # Split into groups for 4x3 grid (last cell used as donut legend)
    n_groups = 11
    groups = np.array_split(np.arange(len(classes)), n_groups)

    fig = plt.figure(figsize=(20, 18), dpi=400)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        4, 3, figure=fig, width_ratios=[1, 1, 1], wspace=-0.45, hspace=0.4
    )

    vmin, vmax = max(1, sizes.min()), sizes.max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    axes = []
    for i in range(n_groups):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, col], polar=True)
        axes.append(ax)

    donut_ax = fig.add_subplot(gs[3, 2], polar=True)
    donut_ax.set_position([0.69, 0.18, 0.05, 0.05])

    # Draw spirals
    for g, ax in zip(groups, axes):
        cls_group, size_group = classes[g], sizes[g]
        n_classes = len(cls_group)

        theta = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
        base_radii = np.linspace(0.45, 1.25, n_classes)

        # Point size ~ log(count)
        point_sizes = np.interp(
            np.log1p(size_group),
            (np.log1p(size_group).min(), np.log1p(size_group).max()),
            (100, 400),
        )

        ax.scatter(
            theta,
            base_radii,
            c=size_group,
            cmap=cmap,
            norm=norm,
            s=point_sizes,
            alpha=0.9,
            edgecolors="none",
            zorder=3,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        for ang, rad, cls in zip(theta, base_radii, cls_group):
            deg = np.degrees(ang)
            ha = "right" if 90 < deg < 270 else "left"
            if 90 < deg < 270:
                deg += 180
            formatted_label = wrap_hyphenated(
                format_subscripts(smart_abbreviation(cls))
            )
            weight = "bold" if normalize_label(cls) in normalized_bold else "normal"
            ax.text(
                ang,
                rad + 0.15,
                formatted_label,
                ha=ha,
                va="center",
                rotation=deg,
                rotation_mode="anchor",
                fontsize=9,
                fontweight=weight,
                zorder=4,
            )

        ax.text(
            0.55,
            0.4,
            f"{size_group.min()}–{size_group.max()} samples",
            fontsize=11,
            ha="center",
            va="center",
            transform=ax.transAxes,
            zorder=5,
        )

        ax.set_rlim(0, 1.45)

    # Donut legend (log scale)
    donut_ax.set_xticks([])
    donut_ax.set_yticks([])
    donut_ax.set_frame_on(False)
    donut_ax.set_facecolor("white")

    inner_r, outer_r = 0.15, 0.25
    donut_ax.set_ylim(0, outer_r)

    theta = np.linspace(0, 2 * np.pi, 720)
    r = np.linspace(inner_r, outer_r, 2)
    T, R = np.meshgrid(theta, r)

    log_vals = np.logspace(np.log10(vmin), np.log10(vmax), len(theta))
    C = np.tile(log_vals, (2, 1))

    donut_ax.pcolormesh(T, R, C, cmap=cmap, norm=norm, shading="auto")
    donut_ax.fill_between(theta, 0, inner_r, color="white", zorder=5)

    log_ticks = safe_logspace_ticks(vmin, vmax)
    for val in log_ticks:
        angle = (np.log(val) - np.log(vmin)) / (np.log(vmax) - np.log(vmin)) * 2 * np.pi
        donut_ax.plot(
            [angle, angle], [inner_r, outer_r], color="white", lw=2.0, zorder=10
        )
        if val not in (vmin, vmax):
            label_radius = outer_r + 0.1
            donut_ax.text(
                angle,
                label_radius,
                f"$10^{{{int(np.log10(val))}}}$",
                ha="center",
                va="center",
                fontsize=11,
            )

    donut_ax.text(
        0.5,
        -0.5,
        "Number of samples per class\n(log scale for color)",
        ha="center",
        va="center",
        transform=donut_ax.transAxes,
        fontsize=12,
    )

    # Figure title and subtitle
    plt.suptitle("Spiral Distribution of Classes by Sample Count", fontsize=18, y=0.93)
    plt.figtext(
        0.5,
        0.908,
        "Regular font indicates species; bold font indicates series, groups, and polymorphs",
        fontsize=12,
        ha="center",
    )

    # Save
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "tiff", "pdf"):
        plt.savefig(
            out_prefix.with_suffix(f".{ext}"),
            dpi=dpi,
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(fig)
    logging.info("Spiral plot saved to %s.png/.tiff/.pdf", out_prefix)


# -----------------------------
# Mahalanobis Outlier Filtering & Balancing
# -----------------------------
def mahalanobis_distance_mcd(
    df: pd.DataFrame, eps=1e-3, class_name=None, verbose=False, weight_by="mean"
) -> pd.Series:
    """Robust Mahalanobis distances per row using MCD + ridge + pseudoinverse.

    Args:
        df: Feature matrix for a single class (rows=samples).
        eps: Base ridge term added to Σ as Σ + λI.
        class_name: Class name (used in logs).
        verbose: If True, emit MCD warnings to logs.
        weight_by: 'mean' (default), 'max' or None — multiplicative column weighting
            before MCD to emphasize higher-contribution components.

    Returns:
        pd.Series with non-negative Mahalanobis distances.

    Notes:
        - Uses MinCovDet for robust location/scatter.
        - Regularizes Σ with λ = max(eps, 1e-6 * trace(Σ)/p) and uses pinvh(Σ) (Moore–Penrose)
          to handle potential rank deficiency (compositional/collinearity).
        - Vectorized computation via einsum; Σ⁺ is symmetrized for numeric safety.
    """
    df = df.loc[:, df.var() > 0]
    if len(df) <= 2:
        return pd.Series([0.0] * len(df), index=df.index)

    if weight_by == "mean":
        weights = df.mean().replace(0, 1)
    elif weight_by == "max":
        weights = df.max().replace(0, 1)
    else:
        weights = pd.Series(1.0, index=df.columns)

    df_weighted = df * weights
    support = max(0.5, min(0.75, (len(df_weighted) - 1) / len(df_weighted)))

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            warnings.simplefilter("always", category=UserWarning)
            warnings.simplefilter("always", category=RuntimeWarning)

            mcd = MinCovDet(support_fraction=support, random_state=GLOBAL_SEED).fit(
                df_weighted
            )

            if verbose and w and class_name:
                logging.warning(
                    "Mahalanobis warning in '%s' (%d warning(s))", class_name, len(w)
                )
                for warn in w:
                    logging.warning("   → %s", str(warn.message))

        mean = mcd.location_
        p = df_weighted.shape[1]
        ridge = max(eps, 1e-6 * (np.trace(mcd.covariance_) / p))
        cov = mcd.covariance_ + np.eye(p) * ridge
        inv_covmat = pinvh(cov, rtol=1e-12)
        inv_covmat = 0.5 * (inv_covmat + inv_covmat.T)

        Xc = df_weighted.to_numpy(dtype=float) - mean
        d2 = np.einsum("ij,jk,ik->i", Xc, inv_covmat, Xc)
        distances = pd.Series(np.sqrt(np.clip(d2, 0, None)), index=df_weighted.index)
        return distances

    except Exception as e:
        if verbose and class_name:
            logging.error("Error in Mahalanobis for '%s': %s", class_name, e)
        raise e


def run_mahalanobis_filtering(
    X_input: pd.DataFrame,
    y_input: pd.Series,
    metadata_input: pd.DataFrame,
    excluded_input: pd.DataFrame,
    source_colors: dict,
    round_n: int,
    maha_plot_dir: Path,
    classes_to_process: list[str] | None = None,
    final_retention_dict: dict | None = None,
):
    """Per-class robust Mahalanobis filtering with adaptive gap thresholding.

    - Applies MCD + pseudoinverse per class and finds a cutoff in the tail via gap Z-scores,
      with a conservative rule (≥1.25× next-largest gap; ≤5% exclusion).
    - Iterates over rounds until no further changes.
    - Emits diagnostics (distances, Z-scores, retention) and plots per class/round.
    """
    filtered_X, filtered_y, filtered_meta, filtered_excluded = [], [], [], []
    diagnostic_tables = []

    all_classes = y_input.unique()

    for cls in all_classes:
        mask = y_input == cls
        X_cls = X_input.loc[mask]
        y_cls = y_input.loc[mask]
        meta_cls = metadata_input.loc[mask]
        excl_cls = excluded_input.loc[mask]
        before = len(X_cls)

        if classes_to_process is not None and cls not in classes_to_process:
            filtered_X.append(X_cls)
            filtered_y.append(y_cls)
            filtered_meta.append(meta_cls)
            filtered_excluded.append(excl_cls)
            continue

        if before > 10:
            try:
                dist = mahalanobis_distance_mcd(X_cls, class_name=cls, weight_by="mean")
                sorted_dist = np.sort(dist.to_numpy(dtype=float))

                p10 = int(TAIL_START_FRAC * len(sorted_dist))
                tail_region = sorted_dist[p10:]

                gaps = np.diff(tail_region)
                threshold = None
                z_scores = np.array([])
                z_thresh = None
                z_scale = 1.0

                if len(gaps) > 1:
                    z_scores = (gaps - np.mean(gaps)) / (np.std(gaps) + 1e-8)
                    z_thresh = 2 - expit((len(X_cls) - 100) / 30)
                    expressive_gaps = set(np.where(z_scores > z_thresh)[0])

                    def find_threshold_from(idx, candidate):
                        lower_indices = list(range(idx - 1, max(-1, idx - 6), -1))
                        lower_expressive = [
                            i for i in lower_indices if i in expressive_gaps
                        ]
                        if lower_expressive:
                            prev_idx = min(lower_expressive)
                            earlier_candidate = tail_region[prev_idx]
                            earlier_outliers = np.sum(dist > earlier_candidate)
                            if earlier_outliers > MAX_EXCLUSION_FRAC * len(dist):
                                return candidate
                            return find_threshold_from(prev_idx, earlier_candidate)
                        return candidate

                    for idx in sorted(expressive_gaps):
                        current_gap = gaps[idx]
                        next_largest = max(
                            [
                                g
                                for j, g in enumerate(gaps)
                                if j != idx and g < current_gap
                            ],
                            default=None,
                        )
                        if (
                            next_largest is None
                            or current_gap < GAP_RATIO_THRESH * next_largest
                        ):
                            continue

                        candidate = tail_region[idx]
                        num_outliers = np.sum(dist > candidate)
                        if num_outliers > MAX_EXCLUSION_FRAC * len(X_cls):
                            continue

                        threshold = find_threshold_from(idx, candidate)
                        break

                retained = (
                    dist <= threshold
                    if threshold is not None
                    else np.full(len(X_cls), True, dtype=bool)
                )

                # Retention tracker
                if final_retention_dict is not None:
                    sample_ids = excluded_input.loc[mask, "global_id"]
                    for sid, is_retained in zip(sample_ids, retained):
                        key = f"{cls}_{sid}"
                        prev = final_retention_dict.get(key, True)
                        final_retention_dict[key] = prev and bool(is_retained)

                # Diagnostics table
                diagnostic_df = X_cls.copy()
                diagnostic_df["mahalanobis"] = dist
                z_score_full = np.full(len(X_cls), np.nan)

                if len(gaps) > 1:
                    sorted_idx = dist.sort_values().index
                    tail_indices = sorted_idx[p10 + 1 : p10 + 1 + len(z_scores)]
                    z_score_full[X_cls.index.get_indexer(tail_indices)] = z_scores

                diagnostic_df["z_score"] = z_score_full
                diagnostic_df["retained"] = retained
                diagnostic_df["mineral_class"] = cls
                diagnostic_df["sample_index"] = excluded_input.loc[
                    mask, "global_id"
                ].values
                diagnostic_tables.append(diagnostic_df)

                # Plot
                dist_df = pd.DataFrame(
                    {
                        "distance": dist,
                        "source": excl_cls["database"].fillna("Unknown").astype(str),
                    }
                )
                dist_df_sorted = dist_df.sort_values("distance").reset_index(drop=True)

                plt.style.use("default")
                fig, ax1 = plt.subplots(figsize=(12, 5), dpi=400, facecolor="white")

                z_line = None
                if len(gaps) > 0 and len(z_scores) > 0:
                    bar_x = np.arange(p10 + 1, p10 + 1 + len(z_scores))
                    max_dist = dist.max()
                    max_z = max(z_scores.max(), z_thresh if z_thresh is not None else 0)
                    z_scale = max_dist / max_z if max_z > 0 else 1.0
                    scaled_z = z_scores * z_scale
                    scaled_thresh = (z_thresh or 0) * z_scale

                    ax1.bar(bar_x, scaled_z, width=1.0, color="orange", zorder=1)
                    z_line = ax1.axhline(
                        scaled_thresh,
                        color="gray",
                        linestyle="--",
                        linewidth=1.8,
                        zorder=4,
                    )

                scatter_handles = []
                for src in dist_df_sorted["source"].unique():
                    mask_src = dist_df_sorted["source"] == src
                    h = ax1.scatter(
                        dist_df_sorted[mask_src].index,
                        dist_df_sorted[mask_src]["distance"],
                        color=source_colors.get(src, "C0"),
                        s=18,
                        alpha=0.7,
                        zorder=3,
                    )
                    scatter_handles.append((h, src))

                outlier_line = None
                if threshold is not None:
                    outlier_line = ax1.axhline(
                        threshold, color="red", linestyle="--", linewidth=1.8, zorder=4
                    )

                ax1.set_title(
                    f"Iterative Mahalanobis Filtering (Round {round_n}) – {cls}",
                    fontsize=14,
                    weight="bold",
                )
                ax1.set_xlabel("Sample Index (sorted)", fontsize=12)
                ax1.set_ylabel("Mahalanobis Distance", fontsize=12)
                ax1.set_ylim(bottom=0)
                ax1.tick_params(labelsize=10)
                ax1.grid(alpha=0.2, linestyle="--")

                ax2 = ax1.twinx()
                ax2.set_ylabel("Gap Z-score", color="orange", fontsize=12)
                if len(gaps) > 0 and len(z_scores) > 0:
                    ax2.set_ylim([y / z_scale for y in ax1.get_ylim()])
                ax2.tick_params(axis="y", labelcolor="orange", labelsize=10)
                ax1.spines["top"].set_visible(False)
                ax2.spines["top"].set_visible(False)

                legend_elements = [Line2D([0], [0], color="none", label="Samples:")]
                for _, src in scatter_handles:
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color=source_colors.get(src, "C0"),
                            label="  " + src,
                            markerfacecolor=source_colors.get(src, "C0"),
                            markersize=6,
                            linewidth=0,
                        )
                    )
                legend_elements.append(
                    Line2D([0], [0], color="none", label="\nThresholds:")
                )
                if z_line is not None and z_thresh is not None:
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            color="gray",
                            linestyle="--",
                            linewidth=1.5,
                            label=f"  Expressive Gaps Z-score = {z_thresh:.2f}",
                        )
                    )
                if outlier_line is not None:
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            color="red",
                            linestyle="--",
                            linewidth=1.5,
                            label=f"  Outlier Threshold = {threshold:.2f}",
                        )
                    )

                ax1.legend(
                    legend_elements,
                    [l.get_label() for l in legend_elements],
                    loc="upper left",
                    fontsize=10,
                    frameon=False,
                )

                maha_plot_dir.mkdir(parents=True, exist_ok=True)
                fig_path = (
                    maha_plot_dir
                    / f"mahalanobis_{round_n}_{re.sub(r'[^A-Za-z0-9_-]+','_', str(cls))}.png"
                )
                plt.tight_layout()
                plt.savefig(fig_path, dpi=400, facecolor="white", bbox_inches="tight")
                plt.close(fig)

                filtered_X.append(X_cls[retained])
                filtered_y.append(y_cls[retained])
                filtered_meta.append(meta_cls.loc[retained])
                filtered_excluded.append(excl_cls.loc[retained])

            except np.linalg.LinAlgError as e:
                logging.warning("Skipping MCD for class '%s' due to LinAlgError: %s", cls, e)
                retained = np.ones(len(X_cls), dtype=bool)
                filtered_X.append(X_cls)
                filtered_y.append(y_cls)
                filtered_meta.append(meta_cls)
                filtered_excluded.append(excl_cls)
        else:
            filtered_X.append(X_cls)
            filtered_y.append(y_cls)
            filtered_meta.append(meta_cls)
            filtered_excluded.append(excl_cls)

        after = retained.sum() if before > 10 else before
        logging.info(
            "%-25s | Original: %5d → Retained: %5d (%.1f%%)",
            str(cls),
            before,
            after,
            100 * after / max(1, before),
        )

    if len(filtered_X) == 0:
        logging.warning("No samples retained in filtering. Aborting...")
        return (
            X_input.copy(),
            y_input.copy(),
            metadata_input.copy(),
            excluded_input.copy(),
            pd.DataFrame(),
        )

    X_clean = pd.concat(filtered_X, ignore_index=True)
    y_clean = pd.concat(filtered_y, ignore_index=True)
    metadata_clean = pd.concat(filtered_meta, ignore_index=True)
    excluded_clean = pd.concat(filtered_excluded, ignore_index=True)
    diagnostics_df = (
        pd.concat(diagnostic_tables, ignore_index=True)
        if diagnostic_tables
        else pd.DataFrame()
    )

    return X_clean, y_clean, metadata_clean, excluded_clean, diagnostics_df


def run_preprocessing_pipeline(
    input_path: Path,
    sheet: str,
    label_col: str,
    output_path: Path | None,
    element_columns: list[str],
    metadata_cols: list[str],
    diag_dir: Path,
    maha_plot_dir: Path,
    data_dir: Path,
) -> Path:
    """Full preprocessing workflow (filter → balance → engineer → export).

    Parameters
    ----------
    input_path : Path
        Excel with raw data.
    sheet : str
        Sheet name with the table.
    label_col : str
        Target column name.
    output_path : Path | None
        Where to write the balanced dataset (XLSX). If None, uses outdir default.
    element_columns : list[str]
        Chemistry feature columns.
    metadata_cols : list[str]
        Metadata columns to carry along into the output.
    diag_dir, maha_plot_dir, data_dir : Path
        Output directories for diagnostics, plots, and datasets.

    Returns
    -------
    Path
        Path to the written balanced XLSX.
    """
    t0 = time.time()
    # Load & basic cleaning
    df = pd.read_excel(input_path, sheet_name=sheet)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in data.")
    # Ensure required columns are present
    validate_columns(df, element_columns, kind="feature")
    if metadata_cols:
        validate_columns(df, metadata_cols, kind="metadata")

    df = df.dropna(subset=[label_col]).copy()
    df["global_id"] = np.arange(len(df))

    # Feature / metadata split
    X_full = df[element_columns].copy()
    y_full = df[label_col].copy()
    metadata = (
        df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)
    )
    
    # Initial class count table + preview
    try:
        cc0 = y_full.value_counts().sort_values(ascending=False)
        cc0_df = cc0.rename_axis("class").reset_index(name="n")
        cc0_csv = diag_dir / "class_counts_initial.csv"
        cc0_df.to_csv(cc0_csv, index=False)
        logging.info("[preprocess] Initial class counts:\n%s",
                     cc0_df.to_string(index=False))
        logging.info("[preprocess] Saved full initial class counts to %s", cc0_csv)
    except Exception as e:
        logging.warning("[preprocess] Could not write initial class counts: %s", e)

    # Preserve non-feature, non-metadata columns for traceability (not replicated in SMOTE)
    excluded_cols = [
        c for c in df.columns if c not in element_columns + metadata_cols + [label_col]
    ]
    excluded_data = df[excluded_cols].copy()

    # Filter invalid rows: drop all-zero (or NaN) rows, fill NaNs with 0
    non_zero_rows = ~(X_full.isna() | (X_full == 0)).all(axis=1)
    X_full = X_full[non_zero_rows].fillna(0)
    y_full = y_full[non_zero_rows]
    metadata = metadata[non_zero_rows]
    excluded_data = excluded_data[non_zero_rows]

    # Color mapping for "database" (if present) — used in diagnostics plots
    df_color_ref = excluded_data.copy()
    df_color_ref["database"] = (
        df_color_ref.get(
            "database", pd.Series(index=df_color_ref.index, dtype="string")
        )
        .astype("string")
        .fillna("Unknown")
    )
    all_sources = df_color_ref["database"].astype(object).unique().tolist()
    random.seed(GLOBAL_SEED)
    random.shuffle(all_sources)
    viridis_trunc = truncate_colormap(
        cm.get_cmap("viridis"), 0.05, 0.85, len(all_sources)
    )
    source_colors = {src: viridis_trunc(i) for i, src in enumerate(all_sources)}

    # Iterative Mahalanobis filtering until convergence
    logging.info("\n\nStarting iterative Mahalanobis filtering until convergence...\n")
    X_iter, y_iter, meta_iter, excl_iter = X_full, y_full, metadata, excluded_data
    diagnostics_list = []
    final_retention_dict: dict[str, bool] = {}

    classes_remaining = y_full.unique().tolist()
    round_n = 1
    while True:
        logging.info("\n\nRound %d of Mahalanobis filtering...\n", round_n)
        X_new, y_new, meta_new, excl_new, diag = run_mahalanobis_filtering(
            X_iter,
            y_iter,
            meta_iter,
            excl_iter,
            source_colors=source_colors,
            round_n=round_n,
            maha_plot_dir=maha_plot_dir,
            classes_to_process=classes_remaining,
            final_retention_dict=final_retention_dict,
        )
        diagnostics_list.append(diag.assign(round=f"R{round_n}"))

        # Detect classes that changed
        class_counts_old = y_iter.value_counts()
        class_counts_new = y_new.value_counts()
        classes_remaining = [
            cls
            for cls in class_counts_old.index
            if class_counts_new.get(cls, 0) < class_counts_old[cls]
        ]

        if not classes_remaining:
            break

        X_iter, y_iter, meta_iter, excl_iter = X_new, y_new, meta_new, excl_new
        round_n += 1

    X, y, metadata, excluded_data = X_iter, y_iter, meta_iter, excl_iter
    
    # Class counts after Mahalanobis filtering
    try:
        cc_f = y.value_counts().sort_values(ascending=False)
        cc_f_df = cc_f.rename_axis("class").reset_index(name="n")
        cc_f_csv = diag_dir / "class_counts_after_filtering.csv"
        cc_f_df.to_csv(cc_f_csv, index=False)
        logging.info("[preprocess] Post-filtering class counts:\n%s",
                     cc_f_df.to_string(index=False))
        logging.info("[preprocess] Saved post-filtering class counts to %s", cc_f_csv)
    except Exception as e:
        logging.warning("[preprocess] Could not write post-filtering counts: %s", e)

    # Save diagnostics
    if diagnostics_list and any(len(d) for d in diagnostics_list):
        mahalanobis_diagnostics = pd.concat(diagnostics_list, ignore_index=True)
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag_csv = diag_dir / "mahalanobis_diagnostic_table_iterative.csv"
        mahalanobis_diagnostics.to_csv(diag_csv, index=False)
        logging.info("[preprocess] Saved to %s", diag_csv)
    else:
        logging.info(
            "[preprocess] No Mahalanobis diagnostics to save (all classes ≤10?)."
        )

    # Lightweight final summary
    first_round_keys = set(
        diagnostics_list[0]["mineral_class"].astype(str)
        + "_"
        + diagnostics_list[0]["sample_index"].astype(str)
    )
    final_df = pd.DataFrame(
        [
            {"sample_key": k, "mineral_class": k.split("_", 1)[0], "retained": v}
            for k, v in final_retention_dict.items()
        ]
    )
    final_df = final_df[final_df["sample_key"].isin(first_round_keys)]

    summary_df = (
        final_df.groupby("mineral_class")
        .agg(
            original_samples=("sample_key", "count"),
            retained_samples=("retained", "sum"),
        )
        .reset_index()
    )
    summary_df["excluded_samples"] = (
        summary_df["original_samples"] - summary_df["retained_samples"]
    )
    summary_df["retention_rate"] = (
        summary_df["retained_samples"] / summary_df["original_samples"] * 100
    ).round(1)
    summary_df = summary_df.sort_values("excluded_samples", ascending=False)
    summary_csv = diag_dir / "mahalanobis_summary_lightweight.csv"
    summary_df.to_csv(summary_csv, index=False)
    logging.info("[preprocess] Saved to %s", summary_csv)

    # KMeans undersampling (>TARGET_PER_CLASS → retain ~TARGET_PER_CLASS with diversity)
    class_counts = y.value_counts()
    overrepresented = class_counts[class_counts > TARGET_PER_CLASS].index
    subsampled_idx = []

    with threadpool_limits(limits=1):
        for cls in overrepresented:
            cls_idx = np.where(y.values == cls)[0]
            X_cls = X.iloc[cls_idx].reset_index(drop=True)
            kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=GLOBAL_SEED, n_init=20)
            clusters = kmeans.fit_predict(X_cls)

            cluster_indices = {
                cl: np.where(clusters == cl)[0].tolist() for cl in np.unique(clusters)
            }
            retained_local_idx = []

            for cl, idxs in cluster_indices.items():
                if len(idxs) <= 5:
                    retained_local_idx.extend(idxs)
                else:
                    np.random.seed(GLOBAL_SEED)
                    retained_local_idx.extend(
                        np.random.choice(idxs, 5, replace=False).tolist()
                    )

            retained_local_idx = list(set(retained_local_idx))

            if len(retained_local_idx) < TARGET_PER_CLASS:
                remaining = TARGET_PER_CLASS - len(retained_local_idx)
                eligible_clusters = {}
                for cl, idxs in cluster_indices.items():
                    available = list(set(idxs) - set(retained_local_idx))
                    if available:
                        eligible_clusters[cl] = available
                total_eligible = sum(len(idxs) for idxs in eligible_clusters.values())

                for cl, idxs in eligible_clusters.items():
                    n_av = len(idxs)
                    n_extra = (
                        int(round((n_av / total_eligible) * remaining))
                        if total_eligible
                        else 0
                    )
                    chosen = np.random.choice(
                        idxs, min(n_extra, len(idxs)), replace=False
                    ).tolist()
                    retained_local_idx.extend(chosen)

                while len(retained_local_idx) < TARGET_PER_CLASS and any(eligible_clusters.values()):
                    for idxs in eligible_clusters.values():
                        available = list(set(idxs) - set(retained_local_idx))
                        if available:
                            retained_local_idx.append(np.random.choice(available))
                        if len(retained_local_idx) == TARGET_PER_CLASS:
                            break

            retained_local_idx = list(dict.fromkeys(retained_local_idx))[:TARGET_PER_CLASS]
            retained_global_idx = cls_idx[retained_local_idx]
            subsampled_idx.extend(retained_global_idx.tolist())

    X_sub = X.iloc[subsampled_idx]
    y_sub = y.iloc[subsampled_idx]
    meta_sub = metadata.iloc[subsampled_idx]
    ex_sub = excluded_data.iloc[subsampled_idx]

    # Retain minority classes (<= TARGET_PER_CLASS) as-is
    minority_classes = class_counts[class_counts <= TARGET_PER_CLASS].index
    X_min = X[y.isin(minority_classes)]
    y_min = y[y.isin(minority_classes)]
    meta_min = metadata[y.isin(minority_classes)]
    ex_min = excluded_data[y.isin(minority_classes)]

    # Engineered features (computed after balancing)
    def calculate_engineered_features(
        df_in: pd.DataFrame, cols: list[str]
    ) -> pd.DataFrame:
        df2 = df_in.copy()
        df2["sum of constituents"] = df2[cols].sum(axis=1)
        df2["value of highest constituent"] = df2[cols].max(axis=1)
        df2["value of second highest constituent"] = df2[cols].apply(
            lambda r: r.nlargest(2).iloc[1] if (r > 0).sum() >= 2 else 0, axis=1
        )
        df2["highest + second highest constituents"] = (
            df2["value of highest constituent"]
            + df2["value of second highest constituent"]
        )
        df2["sum of major constituents (>1% wt)"] = df2[cols].apply(
            lambda r: r[r > 1].sum(), axis=1
        )
        df2["number of major constituents (>1% wt)"] = df2[cols].apply(
            lambda r: (r > 1).sum(), axis=1
        )
        return df2

    # Combine overrepresented subset + minority as base for SMOTE
    X_comb_full = pd.concat([X_sub, X_min], ignore_index=True)
    y_comb = pd.concat([y_sub, y_min], ignore_index=True)
    meta_comb = pd.concat([meta_sub, meta_min], ignore_index=True).reset_index(
        drop=True
    )
    ex_comb = pd.concat([ex_sub, ex_min], ignore_index=True).reset_index(drop=True)

    # SMOTE (target each minority class to TARGET_PER_CLASS where feasible)
    vc = y_comb.value_counts()
    smote_target = {cls: TARGET_PER_CLASS for cls in vc[(vc < TARGET_PER_CLASS) & (vc >= 2)].index}
    k_val = max(1, min(5, int(vc.min()) - 1)) if len(vc) else 1
    
    # SMOTE configuration echo
    try:
        logging.info(
            "[preprocess] SMOTE k_neighbors=%d; target minorities→%d (eligible classes=%d).",
            k_val, TARGET_PER_CLASS, len(smote_target)
        )
        if smote_target:
            preview = dict(list(smote_target.items())[:10])
            logging.info("[preprocess] SMOTE sampling preview (up to 10): %s", preview)
    except Exception as e:
        logging.warning("[preprocess] Could not log SMOTE settings: %s", e)
    
    smote = SMOTE(
        sampling_strategy=smote_target, k_neighbors=k_val, random_state=GLOBAL_SEED
    )
    X_resampled, y_resampled = smote.fit_resample(X_comb_full, y_comb)
    
    # Class counts after balancing (final)
    try:
        cc_bal = y_resampled.value_counts().sort_values(ascending=False)
        cc_bal_df = cc_bal.rename_axis("class").reset_index(name="n")
        cc_bal_csv = diag_dir / "class_counts_balanced.csv"
        cc_bal_df.to_csv(cc_bal_csv, index=False)
        logging.info("[preprocess] Balanced class counts:\n%s",
                     cc_bal_df.to_string(index=False))
        logging.info("[preprocess] Saved balanced class counts to %s", cc_bal_csv)
    except Exception as e:
        logging.warning("[preprocess] Could not write balanced class counts: %s", e)

    # Reconstruct metadata for synthetic samples
    num_new = len(X_resampled) - len(X_comb_full)
    synthetic_mask = [False] * len(X_comb_full) + [True] * num_new
    synthetic_labels = (
        y_resampled.iloc[-num_new:]
        if num_new > 0
        else pd.Series([], dtype=y_resampled.dtype)
    )

    meta_synthetic, ex_synthetic = [], []
    for label in synthetic_labels:
        real_indices = y_comb[y_comb == label].index
        chosen = np.random.choice(real_indices)
        meta_synthetic.append(meta_comb.loc[chosen])
        ex_synthetic.append(
            pd.Series([np.nan] * len(excluded_cols), index=excluded_cols)
        )

    meta_resampled = pd.concat(
        [meta_comb, pd.DataFrame(meta_synthetic)], ignore_index=True
    )
    ex_resampled = pd.concat([ex_comb, pd.DataFrame(ex_synthetic)], ignore_index=True)

    # Final assembly
    df_resampled = pd.DataFrame(X_resampled, columns=X_comb_full.columns)
    df_resampled[label_col] = y_resampled.reset_index(drop=True)

    for col in metadata_cols:
        df_resampled[col] = meta_resampled[col].reset_index(drop=True)
    for col in excluded_cols:
        df_resampled[col] = ex_resampled[col].reset_index(drop=True)

    df_resampled["synthetic_data"] = pd.Series(synthetic_mask).reset_index(drop=True)
    df_resampled = calculate_engineered_features(df_resampled, element_columns)

    # Export
    data_dir.mkdir(parents=True, exist_ok=True)
    if not output_path:
        output_path = data_dir / f"{input_path.stem}_balanced.xlsx"
    else:
        output_path = Path(output_path)

    df_resampled = df_resampled.round(2)
    df_resampled.to_excel(output_path, index=False, sheet_name="data")

    logging.info("Balanced dataset saved to %s", output_path)
    logging.info(
        "→ Total: %d | Synthetic: %d | Original: %d",
        len(df_resampled),
        int(df_resampled["synthetic_data"].sum()),
        len(df_resampled) - int(df_resampled["synthetic_data"].sum()),
    )
    logging.info("Total runtime: %.2f s", time.time() - t0)

    return output_path


# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="Mineral dataset visualization and preparation pipeline."
    )
    p.add_argument(
        "--mode",
        choices=["spiral", "preprocess", "all"],
        default="all",
        help="Which workflow to run.",
    )
    p.add_argument("--input", required=True, help="Path to input Excel file.")
    p.add_argument("--sheet", default="data", help="Sheet name in Excel.")
    p.add_argument("--label", default="label", help="Label column name.")
    p.add_argument(
        "--output", default=None, help="Output Excel path for processed dataset."
    )
    p.add_argument(
        "--docx",
        default=None,
        help="Optional DOCX path with bold class names for spiral plot.",
    )
    p.add_argument("--dpi", type=int, default=600, help="DPI for saved figures.")
    p.add_argument(
        "--elements",
        default=None,
        help="JSON list of element columns to use as features. "
        "Default uses common oxides/halogens used in the paper.",
    )
    p.add_argument(
        "--metadata",
        default=None,
        help="JSON list of metadata columns to preserve & replicate in synthetic samples.",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Base output directory. Defaults to <input_stem>_preprocessing_results next to the input file.",
    )
    return p


# Default element & metadata sets (edit if needed)
DEFAULT_ELEMENT_COLUMNS = [
    "SiO2",
    "TiO2",
    "Al2O3",
    "Fe2O3t",
    "MnOt",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "Cr2O3t",
    "BaO",
    "Nb2O5",
    "P2O5",
    "F",
    "Cl",
]
DEFAULT_METADATA_COLS = [
    "systematic classification",
    "higher-order group/species",
    "species/series or lower-order group",
]


def main(args=None):
    if args is None:
        args = build_argparser().parse_args()

    # Parse element/metadata lists if provided
    if args.elements:
        try:
            element_columns = json.loads(args.elements)
            assert isinstance(element_columns, list)
        except Exception:
            raise ValueError("--elements must be a JSON list of column names.")
    else:
        element_columns = DEFAULT_ELEMENT_COLUMNS

    if args.metadata:
        try:
            metadata_cols = json.loads(args.metadata)
            assert isinstance(metadata_cols, list)
        except Exception:
            raise ValueError("--metadata must be a JSON list of column names.")
    else:
        metadata_cols = DEFAULT_METADATA_COLS

    # Paths
    input_path = Path(args.input)
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = input_path.with_name(f"{input_path.stem}_preprocessing_results")

    plots_dir = outdir / "plots"
    spiral_dir = plots_dir / "spiral"
    maha_plot_dir = plots_dir / "mahalanobis"
    diag_dir = outdir / "diagnostics"
    data_dir = outdir / "datasets"
    log_dir = outdir / "logs"

    for d in [spiral_dir, maha_plot_dir, diag_dir, data_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # File logging in addition to console
    fh = logging.FileHandler(log_dir / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.info("Logging initialized. Writing to %s", log_dir / "run.log")

    # === Dump environment snapshot ===
    dump_environment(outdir)
    
    # Header echo (CLI + params JSON)
    log_run_header(args, outdir, element_columns, metadata_cols)

    # Run modes
    if args.mode in ("spiral", "all"):
        plot_spiral_distribution(
            input_path=input_path,
            sheet=args.sheet,
            label_col=args.label,
            docx_path=args.docx,
            out_prefix=spiral_dir / "spiral_distribution_highres",
            dpi=args.dpi,
        )

    if args.mode in ("preprocess", "all"):
        run_preprocessing_pipeline(
            input_path=input_path,
            sheet=args.sheet,
            label_col=args.label,
            output_path=Path(args.output) if args.output else None,
            element_columns=element_columns,
            metadata_cols=metadata_cols,
            diag_dir=diag_dir,
            maha_plot_dir=maha_plot_dir,
            data_dir=data_dir,
        )

    gc.collect()


# -----------------------------
# Standard CLI entry-point
# -----------------------------
# Keep argument parsing separate via `build_argparser()` and wrap execution
# in a try/except block so failures are logged and CI receives a non-zero exit code.
if __name__ == "__main__":
    try:
        try:
            parser = build_argparser()
        except NameError:
            parser = None

        if parser is None:
            main()
        else:
            args = parser.parse_args()
            main(args)
    except SystemExit:
        raise
    except Exception:
        logging.exception("Fatal error")
        sys.exit(1)
