# SPDX-License-Identifier: MIT
# Copyright (c) 2025
# Matheus Santos (ORCID: 0000-0002-1604-381X)

"""
AdaCHEMICA — Hybrid Learning
=========================================================================

Overview
--------
Supervised classification and open-set evaluation for compositional mineral data.
The method combines class-specific confidence thresholds with robust Mahalanobis
distance limits estimated via Minimum Covariance Determinant (MCD).
Mahalanobis per-class limits support three modes:
- "max": maximum in-class distance
- "quantile": in-class q-quantile (q=0.95)
- "inflated_quantile": 1.05 × in-class q-quantile
  
What this script provides
-------------------------
- Robust per-class profiles (MCD center and inverse covariance) and associated
  Mahalanobis distance thresholds.
- Class-specific confidence thresholds from stratified CV out-of-fold (OOF)
  predictions with explicit guardrails.
- Hybrid open-set decision rule (confidence ∧ Mahalanobis distance).
- LOCO (Leave-One-Class-Out) evaluation to simulate unseen classes and quantify
  abstention (unknown) and misclassification behavior.
- Coverage-aware reporting (confident-only and confident+uncertain acceptance).
- Publication-ready diagnostics: confidence-threshold curves; coverage–precision,
  coverage–recall, and coverage–F1 curves; class-wise mosaic; LOCO stacked counts;
  confidence distributions (step histogram, ECDF, and per-status facet histograms);
  Mahalanobis distance profiles (per class); and feature-importance export.

Quick start
-----------
python hybrid_learning.py \
  --input_xlsx outputs/run_prep/datasets/mineral_data_balanced.xlsx \
  --sheet data \
  --out_dir outputs/run_model \
  --plot

Inputs
------
- Excel dataset with chemistry features and target column 'label' (same schema as
  produced by preprocessing).

Outputs (under --out_dir)
-------------------------
- models/
    - learning_model.pkl           : final RandomForest fitted on the full dataset
    - hybrid_thresholds.json       : confident/uncertain thresholds + MCD parameters
- diagnostics/                     : metrics & tables (feature importances, LOCO rates & mappings,
                                     classwise coverage, coverage reports)
    - feature_importance_full.csv
    - loco_openset_full_rates.csv
    - loco_detailed_mappings.csv
    - classwise_coverage.csv
    - open_set_metrics.txt
- plots/
    - feature_importance/          : feature-importance bar plots
    - loco/                        : LOCO stacked bar plots
    - mahalanobis_profiles/        : per-class Mahalanobis distance plots
    - classwise_coverage/          : class-wise mosaic of outcomes
    - coverage_precision_recall/   : coverage–precision/recall/F1 curves
    - confidence_distributions/    : step histogram, ECDF, and facet histograms
- logs/
    - run.log                      : execution log (INFO)
- ENVIRONMENT.txt                  : Python, package versions, OS info

Reproducibility
---------------
- Fixed random seeds and limited threading (OMP_NUM_THREADS, threadpoolctl).
- Headless, high-DPI figures suitable for CI environments.
- Full environment snapshot written to 'ENVIRONMENT.txt'.

Default settings
------------------
- Confident fallback threshold: 0.8
- Minimum uncertainty gap: 0.2
- Mahalanobis thresholding (default): inflated_quantile @ q=1.0 with inflate=1.05
- Covariance diagonal shrink (cov_shrink_alpha): 0.10

Dependencies
------------
Python 3.9+; major libraries: numpy, pandas, scikit-learn, matplotlib, scipy, joblib.
Optional: seaborn (for selected plots).

License & citation
------------------
MIT License (see SPDX header / LICENSE file). Please cite the associated paper when available.

References
----------
- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.
- Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator.
  Technometrics, 41(3), 212–223.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique.
  Journal of Artificial Intelligence Research, 16, 321–357.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
  In Proc. Fifth Berkeley Symposium on Mathematical Statistics and Probability.
- Scheirer, W. J., de Rezende Rocha, A., Sapkota, A., & Boult, T. E. (2013). Toward open set recognition.
  IEEE TPAMI, 35(7), 1757–1772.
"""

from __future__ import annotations

import argparse
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
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Plotting backend for headless environments / CI
import matplotlib

matplotlib.use("Agg")

# Core + needed for dump_environment
import scipy
import sklearn
import joblib

# Scientific and plotting
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import cm, ticker as mticker
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.linalg import pinvh
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import MinCovDet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from itertools import cycle

# Silence matplotlib verbosity
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("matplotlib.colorbar").setLevel(logging.WARNING)

# Try seaborn, but make it optional for environments without it
try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:  # pragma: no cover
    HAS_SEABORN = False

# -----------------------------
# Defaults / Config
# -----------------------------
DEFAULT_FEATURES = [
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
    "value of highest constituent",
    "value of second highest constituent",
    "highest + second highest constituents",
    "sum of constituents",
    "sum of major constituents (>1% wt)",
    "number of major constituents (>1% wt)",
]
DEFAULT_TARGET = "label"

# -----------------------------
# Hyperparameters
# -----------------------------
GLOBAL_SEED = 42
RF_N_ESTIMATORS = 300
CV_FOLDS = 5

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------
# Data Classes
# -----------------------------
@dataclass
class MahalanobisInfo:
    """Container for robust class mean vector and inverse covariance matrix."""

    mean: np.ndarray
    inv_cov: np.ndarray


# -----------------------------
# Utilities
# -----------------------------
def set_determinism(seed: int = GLOBAL_SEED) -> None:
    """Set seeds and environment for reproducibility across runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"  # avoid thread nondeterminism on some platforms
    random.seed(seed)
    np.random.seed(seed)


def dump_environment(out_dir: Path) -> None:
    """Write an environment snapshot to 'ENVIRONMENT.txt'.

    Captures Python/OS details and installed library versions to support
    exact reproducibility.

    Parameters
    ----------
    out_dir : Path
        Root output directory where 'ENVIRONMENT.txt' will be written.

    Notes
    -----
    Optional dependencies (e.g., 'seaborn', 'openpyxl') are recorded only if
    present and otherwise marked as “not installed”.
    """
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
    # Implicit Excel engine (if available)
    try:
        import openpyxl

        info["openpyxl"] = openpyxl.__version__
    except Exception:
        info["openpyxl"] = "not installed"
    # Optional seaborn
    if "HAS_SEABORN" in globals() and HAS_SEABORN:
        import seaborn as sns

        info["seaborn"] = sns.__version__
    else:
        info["seaborn"] = "not installed"

    (out_dir / "ENVIRONMENT.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in info.items()), encoding="utf-8"
    )
    logging.info("Environment written to %s", out_dir / "ENVIRONMENT.txt")


def validate_columns(df: pd.DataFrame, cols: Iterable[str], kind: str) -> None:
    """Ensure required columns are present; raise descriptive error if not."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {kind} columns: {missing}")


def log_run_params(
    args: argparse.Namespace,
    out_dir: Path,
    features: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    """Log and persist all parameters/metadata used in the run."""
    params = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command_line": " ".join(map(shlex.quote, sys.argv)),
        "script": (Path(__file__).name if "__file__" in globals() else "hybrid_learning.py"),
        "random_seed": GLOBAL_SEED,
        "input_xlsx": str(args.input_xlsx),
        "sheet": args.sheet,
        "out_dir": str(out_dir),
        "target": args.target,
        "features_json": args.features_json,
        "n_features": int(X.shape[1]),
        "n_samples": int(len(X)),
        "n_classes": int(y.nunique()),
        "classes": sorted(map(str, y.unique())),
        # Hyperparameters & flags
        "rf_n_estimators": int(args.n_estimators),
        "cv_folds": int(args.cv_folds),
        "fallback_conf": float(args.fallback_conf),
        "min_uncertainty_gap": float(args.min_uncertainty_gap),
        "cov_shrink_alpha": float(args.cov_shrink_alpha),
        "plot": bool(args.plot),
        # Mahalanobis thresholding
        "threshold_mode": str(getattr(args, "threshold_mode", "inflated_quantile")),
        "mahal_quantile": float(getattr(args, "mahal_quantile", 0.95)),
        "mahal_inflate": float(getattr(args, "mahal_inflate", 1.05)),
        # Fixed settings for reference
        "mcd_support_min": 0.5,
        "mcd_support_max": 0.75,
        "mcd_ridge": 1e-3,
        "openset_loose_factor": 2.0,
        "features": list(features),
    }
    if extra:
        params.update(extra)

    text = json.dumps(params, indent=2, ensure_ascii=False)
    logging.info("Run parameters:\n%s", text)
    (out_dir / "logs" / "RUN_PARAMS.json").write_text(text, encoding="utf-8")


# -----------------------------
# Core Computations
# -----------------------------
def compute_mcd_mahalanobis_profiles(
    X: pd.DataFrame,
    y: pd.Series,
    plot_dir: Optional[Path] = None,
    support_min: float = 0.5,
    support_max: float = 0.75,
    ridge: float = 1e-3,
    seed: int = GLOBAL_SEED,
    threshold_mode: str = "inflated_quantile",
    mahal_quantile: float = 0.95,
    mahal_inflate: float = 1.05,
    cov_shrink_alpha: float = 0.10,
) -> Tuple[Dict[str, Optional[MahalanobisInfo]], Dict[str, float]]:
    """
    Compute robust per-class Mahalanobis profiles using MinCovDet (MCD).

    Threshold modes (selected via `threshold_mode`):
      - "inflated_quantile" (default):  threshold = (mahal_inflate) × in-class q-quantile
      - "quantile":                     threshold = in-class q-quantile
      - "max":                          threshold = max in-class distance
    Optionally saves diagnostic plots of per-class distance distributions.
    """
    logging.info(
        "\n\nComputing MCD Mahalanobis profiles (support in [%.2f, %.2f])\n",
        support_min,
        support_max,
    )
    class_info: Dict[str, Optional[MahalanobisInfo]] = {}
    class_thresholds: Dict[str, float] = {}
    plot_dir = (plot_dir / "mahalanobis_profiles") if plot_dir else None
    if plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)

    labels = np.unique(y.values)
    for cls in labels:
        X_cls = X[y == cls].to_numpy()
        if X_cls.shape[0] <= 2:
            logging.warning("Skipping class '%s' (n<=2).", cls)
            class_info[cls] = None
            class_thresholds[cls] = float("inf")
            continue

        # Adaptive support fraction bounded by [support_min, support_max]
        support = max(support_min, min(support_max, (len(X_cls) - 1) / len(X_cls)))
        try:
            mcd = MinCovDet(support_fraction=support, random_state=seed).fit(X_cls)
            cov_raw = mcd.covariance_
            # Ridge + optional diagonal shrinkage for numerical stability
            cov = cov_raw + np.eye(X_cls.shape[1]) * ridge
            if cov_shrink_alpha > 0.0:
                diag_cov = np.diag(np.diag(cov))
                cov = (1.0 - cov_shrink_alpha) * cov + cov_shrink_alpha * diag_cov

            inv_cov = pinvh(cov)  # symmetric pseudo-inverse
            center = mcd.location_

            distances = np.asarray(
                [mahalanobis(x, center, inv_cov) for x in X_cls], dtype=float
            )

            # --- threshold rule ---
            mode = (threshold_mode or "inflated_quantile").lower()
            if mode not in {"max", "quantile", "inflated_quantile"}:
                logging.warning("Invalid threshold_mode='%s'; falling back to 'inflated_quantile'.", mode)
                mode = "inflated_quantile"

            if mode == "max":
                threshold = float(np.max(distances))
                thr_label = f"max = {threshold:.2f}"
            elif mode == "quantile":
                qv = float(np.clip(mahal_quantile, 0.5, 0.9999))
                qthr = float(np.quantile(distances, qv))
                threshold = qthr
                thr_label = f"q{qv:.3f} = {qthr:.2f}"
            else:  # inflated_quantile
                qv = float(np.clip(mahal_quantile, 0.5, 1.0))
                qthr = float(np.quantile(distances, qv))
                thr = float(max(0.0, mahal_inflate) * qthr)
                threshold = thr
                thr_label = f"{mahal_inflate:.2f}×q{qv:.3f} = {thr:.2f}"

            class_info[cls] = MahalanobisInfo(mean=center, inv_cov=inv_cov)
            class_thresholds[cls] = threshold

            logging.info("Class '%s' → threshold (%s)", cls, thr_label)

            # Optional diagnostics: sorted distances with threshold line
            if plot_dir:
                fig, ax = plt.subplots(figsize=(8, 4))
                sdist = np.sort(distances)
                ax.scatter(
                    np.arange(len(sdist)),
                    sdist,
                    s=18,
                    color="steelblue",
                    label="Samples",
                )
                ax.axhline(
                    threshold,
                    color="red",
                    linestyle="--",
                    label=f"Threshold: {thr_label}",
                )
                ax.set_title(f"Mahalanobis Distances – {cls}")
                ax.set_xlabel("Sample index (sorted)")
                ax.set_ylabel("Mahalanobis distance")
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0.01, 0.90),
                    frameon=True,
                    framealpha=0.9,
                    facecolor="white",
                    borderpad=0.4,
                )
                fig.tight_layout()
                fig.savefig(plot_dir / f"mahal_{sanitize(cls)}.png", dpi=300)
                plt.close(fig)

        except Exception as e:
            logging.exception("Skipping class '%s' due to MCD error: %s", cls, e)
            class_info[cls] = None
            class_thresholds[cls] = float("inf")

    return class_info, class_thresholds


def compute_confidence_thresholds(
    X: pd.DataFrame,
    y: np.ndarray,
    rf_n_estimators: int = RF_N_ESTIMATORS,
    cv_folds: int = CV_FOLDS,
    seed: int = GLOBAL_SEED,
    plot_dir: Optional[Path] = None,
    fallback_conf: float = 0.8,
    min_uncertainty_gap: float = 0.2,
) -> Tuple[Dict[str, float], Dict[str, float], RandomForestClassifier, np.ndarray]:
    """
    Compute class-specific 'confident' and 'uncertain' thresholds using CV OOF predictions.

    Heuristic per class:
      - 'uncertain' = first correct prediction by increasing confidence
      - 'confident' = correct at position i, next 3 also correct, and ≥90% of remaining correct

    Guardrails and defaults:
      - Fallback confident threshold ('fallback_conf'): used if the derived confident threshold
        is missing or exceeds `fallback_conf` (default: 0.8). Lower bound at 0.3.
      - Minimum gap between confident and uncertain ('min_uncertainty_gap'): default 0.2,
        i.e., uncertain ≤ (confident − 0.2), with floor at 0.1.

    Returns:
      dict_conf (class → confident threshold),
      dict_unc (class → uncertain threshold),
      trained_rf (last fit from OOF loop),
      OOF_probs (out-of-fold probabilities)
    """
    logging.info(
        "\n\nCross-validating RF (n_estimators=%d, folds=%d) to compute confidence thresholds\n",
        rf_n_estimators,
        cv_folds,
    )

    # Flatten labels to numpy array for sklearn
    y_np = np.asarray(y).ravel()

    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators, random_state=seed, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Allocate OOF probability matrix (rows=samples, cols=all_classes)
    all_classes = np.unique(y_np)
    probs = np.zeros((len(X), len(all_classes)))

    # Build once: global mapping class → column index
    col_map = {c: i for i, c in enumerate(all_classes)}

    # OOF loop
    for train_idx, test_idx in cv.split(X, y_np):
        rf.fit(X.iloc[train_idx], y_np[train_idx])
        fold_classes = rf.classes_
        fold_probs = rf.predict_proba(X.iloc[test_idx])

        dest_cols = [col_map[c] for c in fold_classes]
        probs[np.ix_(test_idx, dest_cols)] = fold_probs

    pred_labels = all_classes[np.argmax(probs, axis=1)]
    conf_scores = np.max(probs, axis=1)

    dict_conf: Dict[str, float] = {}
    dict_unc: Dict[str, float] = {}
    plot_out = (plot_dir / "confidence_thresholds") if plot_dir else None
    if plot_out:
        plot_out.mkdir(parents=True, exist_ok=True)

    for cls in sorted(all_classes):
        # Mask for this class
        mask = y_np == cls
        if int(np.sum(mask)) == 0:
            continue

        conf_cls = pd.Series(conf_scores[mask])
        true_cls = pd.Series(y_np[mask])
        pred_cls = pd.Series(pred_labels[mask])

        # Sort by ascending confidence so early points are the least confident
        sorted_idx = np.argsort(conf_cls.values)
        conf_sorted = conf_cls.iloc[sorted_idx].reset_index(drop=True)
        true_sorted = true_cls.iloc[sorted_idx].reset_index(drop=True)
        pred_sorted = pred_cls.iloc[sorted_idx].reset_index(drop=True)
        correct_sorted = (true_sorted == pred_sorted).astype(int)

        # Uncertain threshold: first correct prediction
        uncertain = None
        first_correct_index = None
        for i in range(len(correct_sorted)):
            if int(correct_sorted.iloc[i]) == 1:
                uncertain = float(conf_sorted.iloc[i])
                first_correct_index = i
                break

        # Confident threshold: stricter rule as described above
        confident = None
        start = first_correct_index if first_correct_index is not None else 0
        for i in range(start, len(correct_sorted)):
            if int(correct_sorted.iloc[i]) == 1:
                if i + 3 < len(correct_sorted):
                    if int(correct_sorted.iloc[i + 1 : i + 4].sum()) < 3:
                        continue
                else:
                    continue
                remaining_correct = int(correct_sorted.iloc[i:].sum())
                remaining_total = len(correct_sorted) - i
                if remaining_total == 0:
                    continue
                acc_remaining = remaining_correct / remaining_total
                if acc_remaining >= 0.90:
                    confident = float(conf_sorted.iloc[i])
                    break

        # Keep originals (for arrows/lines if fallbacks kick in)
        conf_before_adjustment = confident
        unc_before_adjustment = uncertain

        # Guardrails / fallbacks
        used_conf = False
        used_unc = False
        if (confident is None) or (confident > fallback_conf):
            used_conf = True
            confident = fallback_conf
        if confident < 0.3:
            used_conf = True
            confident = 0.3

        if (uncertain is None) or (uncertain >= confident - min_uncertainty_gap):
            used_unc = True
            uncertain = max(0.0, confident - min_uncertainty_gap)
        if uncertain < 0.1:
            used_unc = True
            uncertain = 0.1

        dict_conf[cls] = confident
        dict_unc[cls] = uncertain

        logging.info(
            "Class '%s' → Confident = %.4f | Uncertain = %.4f%s%s",
            cls,
            confident,
            uncertain,
            " (fallback_conf)" if used_conf else "",
            " (fallback_unc)" if used_unc else "",
        )

        # Optional per-class diagnostic plot: cumulative accuracy vs confidence
        if plot_out is not None:
            # Cumulative accuracy as we move from low to high confidence
            n = len(correct_sorted)
            cum_correct = correct_sorted.to_numpy().cumsum()
            cum_total = np.arange(1, n + 1, dtype=float)
            cum_acc = cum_correct / cum_total

            plot_df = pd.DataFrame(
                {
                    "confidence": conf_sorted.values,
                    "accuracy": cum_acc,
                    "true_label": true_sorted.values,
                    "pred_label": pred_sorted.values,
                }
            )
            plot_df["correct"] = plot_df["true_label"] == plot_df["pred_label"]

            # Collapse coincident points to get a density-like mark size
            density_df = (
                plot_df.groupby(
                    ["confidence", "accuracy", "correct", "pred_label"], as_index=False
                )
                .size()
                .rename(columns={"size": "count"})
            )

            all_same = density_df["count"].min() == density_df["count"].max()
            if not all_same:
                norm = mcolors.Normalize(
                    vmin=density_df["count"].min(), vmax=density_df["count"].max()
                )
                cmap = plt.cm.viridis
                density_df["color"] = density_df["count"].apply(lambda c: cmap(norm(c)))
            max_count = max(1, density_df["count"].max())
            density_df["edgecolor"] = density_df["correct"].map(
                {True: "black", False: "red"}
            )
            density_df["edgewidth"] = density_df["correct"].map({True: 0.6, False: 1.6})
            density_df["size"] = 24 + 28 * (density_df["count"] / max_count)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(
                plot_df["confidence"],
                plot_df["accuracy"],
                color="orange",
                label="Cumulative Accuracy",
                zorder=1,
            )

            # Scatter points sized by local density; red edge = wrong predictions
            for _, r in density_df.iterrows():
                ax.scatter(
                    r["confidence"],
                    r["accuracy"],
                    s=r["size"],
                    color=(r["color"] if not all_same else "tab:blue"),
                    edgecolors=r["edgecolor"],
                    linewidths=r["edgewidth"],
                    zorder=2,
                )

            # --- Show counts for points with density > 1 ---
            offset_magnitude = 0.015
            offset_dir = cycle([-1, 1])  # alternate up/down
            for _, row in density_df[density_df["count"] > 1].iterrows():
                dy = next(offset_dir) * offset_magnitude
                x, y_val = row["confidence"], row["accuracy"]
                y_label = y_val + dy
                ax.plot([x, x], [y_val, y_label], color="gray", linewidth=0.5, zorder=1)
                ax.text(
                    x,
                    y_label,
                    str(row["count"]),
                    fontsize=9,
                    color="black",
                    ha="center",
                    va="bottom" if dy > 0 else "top",
                    zorder=4,
                )

            # ---- Place labels for wrong predictions with collision avoidance ----
            wrong = plot_df.loc[~plot_df["correct"]].copy()
            wrong = wrong.sort_values("confidence").reset_index(drop=True)

            def data_delta(ax, dx_px=1.0, dy_px=1.0):
                """Convert a pixel offset to data units at the plot center."""
                inv = ax.transData.inverted()
                cx = 0.5 * (ax.get_xlim()[0] + ax.get_xlim()[1])
                cy = 0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1])
                x2, y2 = inv.transform(
                    ax.transData.transform((cx, cy)) + (dx_px, dy_px)
                )
                return abs(x2 - cx), abs(y2 - cy)

            # Approximate marker radius in data units (assumes ~8 px marker)
            r_x, r_y = data_delta(ax, 8, 8)

            # Estimate a text bounding box in data units from a string and font size
            def approx_bbox(ax, x, y, txt, fs=9):
                w_px = max(28, 6.2 * len(str(txt)) * (fs / 10))
                h_px = 18 * (fs / 10)
                inv = ax.transData.inverted()
                x2, y2 = inv.transform(ax.transData.transform((x, y)) + (w_px, h_px))
                return (min(x, x2), min(y, y2), max(x, x2), max(y, y2))

            def boxes_overlap(a, b, pad=0.002):
                # a,b = (x0,y0,x1,y1) in data units
                return not (
                    a[2] + pad < b[0]
                    or b[2] + pad < a[0]
                    or a[3] + pad < b[1]
                    or b[3] + pad < a[1]
                )

            def box_hits_point(box, px, py, rx, ry, pad=0.0):
                # Expand a point to a small rectangle using marker radii
                pbox = (px - rx - pad, py - ry - pad, px + rx + pad, py + ry + pad)
                return boxes_overlap(box, pbox, pad=0.0)

            # Cache point locations to prevent label overlap with markers
            all_points = plot_df[["confidence", "accuracy"]].to_numpy(dtype=float)

            fs = 9
            base_dx, base_dy = 0.012, 0.028  # base offsets (x,y) per ring
            # 8 candidate directions (R, UR, U, UL, L, DL, D, DR)
            dirs = np.array(
                [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]],
                dtype=float,
            )

            placed = []  # list of (label_x, label_y, text, (point_x, point_y))
            used_boxes = []

            for k, row in wrong.iterrows():
                x = float(row["confidence"])
                y = float(row["accuracy"])
                label = str(row["pred_label"])

                # Rotate starting direction for variety
                dirs_order = np.roll(dirs, shift=(k % len(dirs)), axis=0)

                found = False
                for ring in range(1, 28):  # expand search radius progressively
                    for d in dirs_order:
                        dx = base_dx * ring * d[0]
                        dy = base_dy * ring * d[1]
                        lx, ly = x + dx, y + dy
                        lb = approx_bbox(ax, lx, ly, label, fs)

                        # 1) avoid overlapping other labels
                        if any(boxes_overlap(lb, ub, pad=0.003) for ub in used_boxes):
                            continue

                        # 2) avoid colliding with plotted markers
                        hit_point = False
                        for px, py in all_points:
                            if abs(px - lx) > (lb[2] - lb[0]) + r_x + 0.01:
                                continue
                            if box_hits_point(lb, px, py, r_x, r_y, pad=0.002):
                                hit_point = True
                                break
                        if hit_point:
                            continue

                        used_boxes.append(lb)
                        placed.append((lx, ly, label, (x, y)))
                        found = True
                        break
                    if found:
                        break

                if not found:
                    # Fallback: small offset down-right
                    lx, ly = x + 0.01, y - 0.02
                    used_boxes.append(approx_bbox(ax, lx, ly, label, fs))
                    placed.append((lx, ly, label, (x, y)))

            # Final small relaxation pass to separate any residual overlaps
            for _ in range(3):
                moved = False
                for i in range(len(placed)):
                    li = list(placed[i])
                    bi = approx_bbox(ax, li[0], li[1], li[2], fs)
                    for j in range(i + 1, len(placed)):
                        lj = list(placed[j])
                        bj = approx_bbox(ax, lj[0], lj[1], lj[2], fs)
                        if boxes_overlap(bi, bj, pad=0.002):
                            li[1] += 0.012  # nudge up
                            lj[1] -= 0.012  # nudge down
                            placed[i] = tuple(li)
                            placed[j] = tuple(lj)
                            moved = True
                if not moved:
                    break

            # Draw labels with solid connectors; add white halo behind the line
            for lx, ly, label, (x, y) in placed:
                ann = ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(lx, ly),
                    fontsize=fs,
                    ha="left",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        fc="white",
                        ec="red",
                        lw=0.9,
                        alpha=0.97,
                    ),
                    arrowprops=dict(
                        arrowstyle="-",
                        lw=1.4,
                        color="red",
                        shrinkA=0,
                        shrinkB=2,
                        connectionstyle="arc3,rad=0",
                    ),
                    zorder=4,
                    annotation_clip=False,
                )
                ann.arrow_patch.set_path_effects(
                    [pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()]
                )

            # Final and original thresholds (show arrows if fallback adjusted them)
            ax.axvline(
                confident,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Confident = {confident:.4f}",
            )
            ax.axvline(
                uncertain,
                color="cyan",
                linestyle="--",
                linewidth=2,
                label=f"Uncertain = {uncertain:.4f}",
            )

            # Absolute positions on the Y axis (accuracy ∈ [0,1])
            conf_y = 0.8
            unc_y = 0.6

            if (
                (conf_before_adjustment is not None)
                and not np.isnan(conf_before_adjustment)
                and confident != conf_before_adjustment
            ):
                ax.axvline(
                    conf_before_adjustment,
                    color="green",
                    linestyle=(0, (6, 3)),
                    linewidth=1.2,
                    label=f"Original Confident = {conf_before_adjustment:.4f}",
                )
                ax.annotate(
                    "",
                    xy=(confident, conf_y),
                    xytext=(conf_before_adjustment, conf_y),
                    arrowprops=dict(arrowstyle="->", color="green", linewidth=1.4),
                )

            if (
                (unc_before_adjustment is not None)
                and not np.isnan(unc_before_adjustment)
                and uncertain != unc_before_adjustment
            ):
                ax.axvline(
                    unc_before_adjustment,
                    color="cyan",
                    linestyle=(0, (2, 3)),
                    linewidth=1.2,
                    label=f"Original Uncertain = {unc_before_adjustment:.4f}",
                )
                ax.annotate(
                    "",
                    xy=(uncertain, unc_y),
                    xytext=(unc_before_adjustment, unc_y),
                    arrowprops=dict(arrowstyle="->", color="cyan", linewidth=1.4),
                )

            # Legend and colorbar
            handles, labels_ = ax.get_legend_handles_labels()
            if not all_same:
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                fig.colorbar(sm, label="Sample Density", shrink=0.8, ax=ax)

            sample_color = (
                "tab:blue" if all_same else cmap(norm(density_df["count"].median()))
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=sample_color,
                    markeredgecolor="black",
                )
            )
            labels_.append("Samples")
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor="none",
                    markeredgecolor="red",
                    markeredgewidth=1.6,
                )
            )
            labels_.append("Wrong prediction (edge)")
            ax.legend(handles, labels_, loc="lower right")

            # Axes cosmetics
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Cumulative Accuracy")
            ax.set_title(f"Class: {cls}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

            fig.tight_layout()
            fig.savefig(plot_out / f"confthr_{sanitize(cls)}.png", dpi=300)
            plt.close(fig)

    return dict_conf, dict_unc, rf, probs


def evaluate_sample(
    sample: pd.Series,
    pred_class: str,
    conf: float,
    conf_thresholds: Dict[str, float],
    uncertain_thresholds: Dict[str, float],
    mahal_thresholds: Dict[str, float],
    mahal_info_dict: Dict[str, Optional[MahalanobisInfo]],
    loose_factor: float = 2.0,
) -> str:
    """
    Apply the hybrid open-set rule (confidence ∧ Mahalanobis).

    Decision:
      - "confident": conf ≥ conf_threshold AND M(x) ≤ mahal_threshold
      - "uncertain": conf ≥ uncertain_threshold AND M(x) ≤ (loose_factor × mahal_threshold)
      - "unknown": otherwise

    The 'loose_factor' (>1) relaxes the Mahalanobis gate for the 'uncertain' band to avoid
    rejecting near-boundary true samples that still deserve manual review or lower-risk acceptance.
    """
    conf_thresh = conf_thresholds[pred_class]
    unc_thresh = uncertain_thresholds[pred_class]
    mahal_thresh = mahal_thresholds[pred_class]
    loose_mahal_thresh = mahal_thresh * loose_factor

    mahal_dist = np.inf
    m_info = mahal_info_dict.get(pred_class)
    if m_info is not None:
        x = np.asarray(sample, dtype=float)
        mahal_dist = mahalanobis(x, m_info.mean, m_info.inv_cov)

    if (conf >= conf_thresh) and (mahal_dist <= mahal_thresh):
        return "confident"
    elif (conf >= unc_thresh) or (mahal_dist <= loose_mahal_thresh):
        return "uncertain"
    else:
        return "unknown"


def loco_openset_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    conf_thresholds: Dict[str, float],
    uncertain_thresholds: Dict[str, float],
    mahal_thresholds: Dict[str, float],
    mahal_info: Dict[str, Optional[MahalanobisInfo]],
    loose_factor: float = 2.0,
    rf_n_estimators: int = RF_N_ESTIMATORS,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """
    Leave-One-Class-Out evaluation under the hybrid open-set rules.

    Trains without the held-out class and classifies held-out samples as
    confident / uncertain / unknown. Returns a DataFrame with counts and rates.
    Also attaches a per-mapped-class detail DataFrame at df.attrs['detailed_mappings'].
    """
    logging.info("\n\nRunning LOCO open-set evaluation\n")
    rows: List[Tuple[str, int, int, int, int]] = []
    detailed_rows: List[
        Tuple[str, str, int, int, int, float]
    ] = []  # (heldout, mapped_cls, n_conf, n_unc, total_mapped, pct_of_heldout)

    for heldout in sorted(set(y)):
        X_train = X[y != heldout]
        y_train = y[y != heldout]
        X_test = X[y == heldout]

        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators, random_state=seed, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        probs = rf.predict_proba(X_test)
        top1 = np.argmax(probs, axis=1)
        labels = rf.classes_[top1]
        confs = probs[np.arange(len(X_test)), top1]

        status: List[str] = []
        detailed_predictions: List[Tuple[str, str]] = []  # (predicted_class, final_tag)

        for i in range(len(X_test)):
            sample = X_test.iloc[i]
            pred_cls = labels[i]
            conf = confs[i]

            tag = evaluate_sample(
                sample=sample,
                pred_class=pred_cls,
                conf=conf,
                conf_thresholds=conf_thresholds,
                uncertain_thresholds=uncertain_thresholds,
                mahal_thresholds=mahal_thresholds,
                mahal_info_dict=mahal_info,
                loose_factor=loose_factor,
            )
            status.append(tag)
            detailed_predictions.append((pred_cls, tag))

        n_total = len(status)
        n_conf = status.count("confident")
        n_unc = status.count("uncertain")
        n_unk = status.count("unknown")
        rows.append((heldout, n_total, n_conf, n_unc, n_unk))

        # safe percentages
        p = lambda k: (k / n_total * 100) if n_total else 0.0
        logging.info(
            "Held-out '%s' (n = %d) | confident = %d (%.1f%%) | uncertain = %d (%.1f%%) | unknown = %d (%.1f%%)",
            heldout,
            n_total,
            n_conf,
            p(n_conf),
            n_unc,
            p(n_unc),
            n_unk,
            p(n_unk),
        )

        # === detailed breakdown of non-unknown mappings by predicted class ===
        mapping_counter = defaultdict(lambda: defaultdict(int))
        for mapped_cls, tag in detailed_predictions:
            if tag != "unknown":
                mapping_counter[mapped_cls][tag] += 1

        if mapping_counter:
            logging.info("    → Detailed mappings for '%s':", heldout)
            for mapped_cls, tag_counts in mapping_counter.items():
                c_conf = tag_counts.get("confident", 0)
                c_unc = tag_counts.get("uncertain", 0)
                total_mapped = c_conf + c_unc
                pct_mapped = (total_mapped / n_total) if n_total else 0.0
                parts = []
                if c_conf:
                    parts.append(f"{c_conf} confident")
                if c_unc:
                    parts.append(f"{c_unc} uncertain")
                detail_str = " + ".join(parts) if parts else "—"
                logging.info(
                    "        - %-25s: %3d (%.1f%%) [%s]",
                    mapped_cls,
                    total_mapped,
                    pct_mapped * 100,
                    detail_str,
                )
                detailed_rows.append(
                    (heldout, mapped_cls, c_conf, c_unc, total_mapped, pct_mapped)
                )

    df = pd.DataFrame(
        rows, columns=["class", "n_samples", "n_confident", "n_uncertain", "n_unknown"]
    )
    df["confident_rate"] = df["n_confident"] / df["n_samples"]
    df["uncertain_rate"] = df["n_uncertain"] / df["n_samples"]
    df["unknown_rate"] = df["n_unknown"] / df["n_samples"]

    # attach the detailed dataframe for the caller to save
    if detailed_rows:
        detailed_df = (
            pd.DataFrame(
                detailed_rows,
                columns=[
                    "heldout_class",
                    "mapped_class",
                    "n_confident",
                    "n_uncertain",
                    "n_total_mapped",
                    "pct_of_heldout",
                ],
            )
            .sort_values(["heldout_class", "n_total_mapped"], ascending=[True, False])
            .reset_index(drop=True)
        )
        df.attrs["detailed_mappings"] = detailed_df

    return df


def coverage_aware_metrics(
    X: pd.DataFrame,
    y: List[str],
    probs: np.ndarray,
    rf_classes: np.ndarray,
    conf_thresholds: Dict[str, float],
    uncertain_thresholds: Dict[str, float],
    mahal_thresholds: Dict[str, float],
    mahal_info: Dict[str, Optional[MahalanobisInfo]],
    loose_factor: float = 2.0,
    plot_dir: Optional[Path] = None,
) -> Tuple[str, str, float, float, pd.DataFrame]:
    """
    Apply the hybrid open-set rule to OOF predictions and produce coverage-aware summaries
    plus diagnostics.

    Summary of outputs
    ------------------
    - Confident-only report (string from sklearn.classification_report):
      accepts only "confident"; "uncertain" and "unknown" contaminate errors (unclassified).
    - Confident+uncertain report (string):
      accepts "confident" and "uncertain"; only "unknown" is unclassified.
    - Coverage metrics: overall coverage for confident-only and confident+uncertain.
    - Class-wise coverage table: per-class counts/rates for {confident, uncertain, unknown}.

    Plots
    -----
    If plotting is enabled (via '--plot'), this function writes, when possible:
      • Class-wise mosaic (classwise_coverage/)
      • Coverage–Precision, –Recall, –F1 curves (coverage_precision_recall/)
      • Confidence distributions (confidence_distributions/): step, ECDF, facet histograms.
      • “Most confused” heatmaps (most_confused/): strict vs lenient acceptance.
    """
    confidences = np.max(probs, axis=1)
    preds = rf_classes[np.argmax(probs, axis=1)]
    y_true = list(y)

    statuses: List[str] = []
    y_pred_conf_only: List[str] = []
    y_pred_known: List[str] = []

    for i in range(len(X)):
        pred_class = preds[i]
        conf = confidences[i]
        sample = X.iloc[i]
        tag = evaluate_sample(
            sample=sample,
            pred_class=pred_class,
            conf=conf,
            conf_thresholds=conf_thresholds,
            uncertain_thresholds=uncertain_thresholds,
            mahal_thresholds=mahal_thresholds,
            mahal_info_dict=mahal_info,
            loose_factor=loose_factor,
        )
        statuses.append(tag)

        if tag == "confident":
            y_pred_conf_only.append(pred_class)
            y_pred_known.append(pred_class)
        elif tag == "uncertain":
            y_pred_conf_only.append("unclassified")
            y_pred_known.append(pred_class)
        else:
            y_pred_conf_only.append("unclassified")
            y_pred_known.append("unclassified")

    # Coverage metrics
    n_conf = statuses.count("confident")
    n_unc = statuses.count("uncertain")
    n_tot = len(statuses)
    cov_conf = n_conf / n_tot if n_tot else 0.0
    cov_conf_plus_unc = (n_conf + n_unc) / n_tot if n_tot else 0.0

    # Reports (only ground-truth labels are evaluated)
    real_labels = sorted(set(y_true))
    rep_conf_only = classification_report(
        y_true,
        y_pred_conf_only,
        labels=real_labels,
        target_names=real_labels,
        digits=3,
        zero_division=0,
    )
    rep_conf_plus_unc = classification_report(
        y_true,
        y_pred_known,
        labels=real_labels,
        target_names=real_labels,
        digits=3,
        zero_division=0,
    )

    # ===== Class-wise coverage table (counts and correctness) =====
    df_all = pd.DataFrame({"true": y_true, "pred": preds, "status": statuses,})

    rows_cw = []
    for cls in real_labels:
        sub = df_all[df_all["true"] == cls]
        n_tot = len(sub)

        # counts by status
        n_conf = int((sub["status"] == "confident").sum())
        n_unc = int((sub["status"] == "uncertain").sum())
        n_unk = int((sub["status"] == "unknown").sum())

        # correct/incorrect within each status
        n_conf_correct = int(
            ((sub["status"] == "confident") & (sub["pred"] == sub["true"])).sum()
        )
        n_conf_incorrect = n_conf - n_conf_correct

        n_unc_correct = int(
            ((sub["status"] == "uncertain") & (sub["pred"] == sub["true"])).sum()
        )
        n_unc_incorrect = n_unc - n_unc_correct

        rows_cw.append(
            {
                "class": cls,
                "confident": n_conf,
                "uncertain": n_unc,
                "unknown": n_unk,
                "total": n_tot,
                "confident_correct": n_conf_correct,
                "confident_incorrect": max(0, n_conf_incorrect),
                "uncertain_correct": n_unc_correct,
                "uncertain_incorrect": max(0, n_unc_incorrect),
            }
        )

    classwise_df = pd.DataFrame(rows_cw).set_index("class")

    # rates (avoid div/0)
    denom = classwise_df["total"].replace(0, np.nan)
    classwise_df["confident_rate"] = classwise_df["confident"] / denom
    classwise_df["uncertain_rate"] = classwise_df["uncertain"] / denom
    classwise_df["unknown_rate"] = classwise_df["unknown"] / denom
    classwise_df = classwise_df.fillna(0.0).sort_values(
        "confident_rate", ascending=False
    )

    # ===================== Optional plots =====================
    if plot_dir is not None:
        # Subdirectories for plots
        cw_dir = plot_dir / "classwise_coverage"
        pr_dir = plot_dir / "coverage_precision_recall"
        cm_dir = plot_dir / "most_confused"
        cdis_dir = plot_dir / "confidence_distributions"
        for d in (cw_dir, pr_dir, cm_dir, cdis_dir):
            d.mkdir(parents=True, exist_ok=True)

        # --- Class-wise Mosaic Plot ---
        try:
            plot_classwise_mosaic(
                classwise_df,
                save_path_png=cw_dir / "classwise_mosaic.png",
                save_path_pdf=cw_dir / "classwise_mosaic.pdf",
            )
            logging.info("Saved class-wise mosaic plot to %s", cw_dir)
        except Exception as e:
            logging.exception("Skipping class-wise mosaic plot: %s", e)

        # --- Coverage–Precision / Coverage–Recall / Coverage–F1 ---
        try:
            # Sort by descending confidence
            sorted_idx = np.argsort(confidences)[::-1]
            y_true_sorted = np.asarray(y_true)[sorted_idx]
            y_pred_sorted = np.asarray(preds)[sorted_idx]

            N = len(y_true_sorted)
            coverage_levels = np.linspace(0.1, 1.0, 50)

            precisions, recalls, f1s = [], [], []
            for cov in coverage_levels:
                n_samp = int(N * cov)
                if n_samp < 1:
                    precisions.append(np.nan)
                    recalls.append(np.nan)
                    f1s.append(np.nan)
                    continue
                y_t = y_true_sorted[:n_samp]
                y_p = y_pred_sorted[:n_samp]
                correct = np.sum(y_t == y_p)
                p = correct / n_samp if n_samp else np.nan
                r = correct / N if N else np.nan
                if not (np.isnan(p) or np.isnan(r)) and (p + r) > 0:
                    f = 2 * p * r / (p + r)
                else:
                    f = np.nan
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)

            def _save_xy(xs, ys, ylabel, title, stem):
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(xs * 100, ys, marker="o")
                ax.set_xlabel("Coverage (% of samples accepted)")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(True)
                fig.savefig(
                    pr_dir / f"{stem}.png",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                )
                fig.savefig(
                    pr_dir / f"{stem}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                )
                plt.close(fig)

            _save_xy(
                coverage_levels,
                precisions,
                "Precision among accepted",
                "Coverage–Precision Curve",
                "coverage_precision_curve",
            )
            _save_xy(
                coverage_levels,
                recalls,
                "Recall over all samples",
                "Coverage–Recall Curve",
                "coverage_recall_curve",
            )
            _save_xy(
                coverage_levels,
                f1s,
                "F1 (P–R harmonic mean)",
                "Coverage–F1 Curve",
                "coverage_f1_curve",
            )
        except Exception as e:
            logging.exception("Skipping coverage metric curves: %s", e)

        # --- Confidence plots (step, ECDF, facets) ---
        try:
            if not HAS_SEABORN:
                raise RuntimeError("seaborn not installed")

            df_scores = pd.DataFrame(
                {"confidence": confidences, "status": statuses}
            ).dropna(subset=["confidence", "status"])
            if df_scores.empty or not np.isfinite(df_scores["confidence"]).any():
                logging.info(
                    "Confidence plots skipped: no finite confidence values to plot."
                )
            else:
                viridis = cm.get_cmap("viridis", 10)
                palette = {
                    "unknown": viridis(9),
                    "uncertain": viridis(6),
                    "confident": viridis(3),
                }
                hue_order = [
                    s
                    for s in ["unknown", "uncertain", "confident"]
                    if s in df_scores["status"].unique()
                ]
                bins_array = np.linspace(0.0, 1.0, 120)

                # Step histogram (density) with hue
                fig, ax = plt.subplots(figsize=(9, 5.5))
                sns.histplot(
                    data=df_scores,
                    x="confidence",
                    hue="status",
                    bins=bins_array,
                    stat="density",
                    common_norm=False,
                    element="step",
                    fill=False,
                    ax=ax,
                    palette=palette,
                    hue_order=hue_order,
                )
                ax.set_title("Confidence Distributions by Final Status (Step Hist)")
                ax.set_xlabel("Confidence")
                ax.set_ylabel("Density")
                ax.grid(True, alpha=0.25)
                fig.savefig(
                    cdis_dir / "confidence_step.png", dpi=300, facecolor="white"
                )
                fig.savefig(
                    cdis_dir / "confidence_step.pdf", dpi=300, facecolor="white"
                )
                plt.close(fig)

                # ECDF
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                sns.ecdfplot(
                    data=df_scores,
                    x="confidence",
                    hue="status",
                    ax=ax2,
                    palette=palette,
                    hue_order=hue_order,
                )
                ax2.set_title("ECDF of Confidence by Final Status")
                ax2.set_xlabel("Confidence")
                ax2.set_ylabel("F(x)")
                ax2.grid(True, alpha=0.3)
                fig2.savefig(
                    cdis_dir / "confidence_ecdf.png",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                )
                fig2.savefig(
                    cdis_dir / "confidence_ecdf.pdf",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                )
                plt.close(fig2)

                # Facets (no hue) → recolor each facet to match palette
                g = sns.displot(
                    data=df_scores,
                    x="confidence",
                    col="status",
                    col_order=hue_order,
                    col_wrap=3,
                    kind="hist",
                    bins=bins_array,
                    stat="density",
                    common_norm=False,
                    element="step",
                    fill=False,
                    facet_kws=dict(sharex=True, sharey=True),
                )
                for ax_f in g.axes.flatten():
                    title = ax_f.get_title() or ""
                    status_name = (
                        title.split("=")[-1].strip() if "=" in title else title.strip()
                    )
                    col = palette.get(status_name, "black")
                    for p in ax_f.patches:
                        try:
                            p.set_edgecolor(col)
                            p.set_linewidth(1.3)
                            p.set_facecolor("none")
                        except Exception:
                            pass
                    for line in getattr(ax_f, "lines", []):
                        line.set_color(col)
                        line.set_linewidth(1.3)

                g.set(xlim=(0, 1))
                g.set_axis_labels("Confidence", "Density")
                g.fig.suptitle("Confidence Distributions by Status (Facets)", y=1.02)
                g.figure.savefig(
                    cdis_dir / "confidence_facets.png",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                )
                g.figure.savefig(
                    cdis_dir / "confidence_facets.pdf",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                )
                plt.close(g.figure)
        except Exception as e:
            logging.warning("Skipping confidence plots: %s", e)

        # --- "Most Confused" heatmaps (strict / lenient) ---
        try:
            if not HAS_SEABORN:
                raise RuntimeError("seaborn not installed")

            real_labels = sorted(set(y_true))
            labels_with_unclassified = list(real_labels) + ["unclassified"]
            n_labels = len(real_labels)
            top_n = min(25, n_labels)

            # Strict: only confident; others → unclassified
            adjusted_preds_strict = [
                p if s == "confident" else "unclassified"
                for p, s in zip(preds, statuses)
            ]
            # Lenient: confident + uncertain; unknown → unclassified
            adjusted_preds_lenient = [
                p if s in ["confident", "uncertain"] else "unclassified"
                for p, s in zip(preds, statuses)
            ]

            # STRICT
            cm_strict = confusion_matrix(
                y_true,
                adjusted_preds_strict,
                labels=labels_with_unclassified,
                normalize="true",
            )
            errors_strict = 1 - np.diag(cm_strict[:n_labels, :n_labels])
            top_classes_strict = np.argsort(errors_strict)[-top_n:]
            uncl_idx = labels_with_unclassified.index("unclassified")
            cm_top_strict = cm_strict[
                np.ix_(top_classes_strict, top_classes_strict.tolist() + [uncl_idx])
            ]
            labels_top = [real_labels[i] for i in top_classes_strict]
            labels_top_plus_uncl = labels_top + ["unclassified"]
            labels_strict = np.vectorize(fmt_cell)(cm_top_strict)

            fig, ax = plt.subplots(figsize=(24, 18))
            sns.heatmap(
                cm_top_strict,
                xticklabels=labels_top_plus_uncl,
                yticklabels=labels_top,
                cmap="YlGnBu",
                annot=labels_strict,
                fmt="",
                annot_kws={"fontsize": 18},
                square=True,
                ax=ax,
                linewidths=0.5,
                linecolor="white",
                cbar=True,
                cbar_kws={"shrink": 0.6, "aspect": 20},
            )
            ax.set_title(
                f"Confusion Matrix – Top {top_n} Most Confused Classes (Strict)\nOnly Confident Predictions; Uncertain + Unknown → unclassified",
                fontsize=24,
            )
            ax.set_xlabel("Predicted Label", fontsize=20, labelpad=10)
            ax.set_ylabel("True Label", fontsize=20, labelpad=10)
            plt.xticks(rotation=45, ha="right", fontsize=18)
            for tick in ax.get_yticklabels():
                tick.set_rotation(45)
                tick.set_rotation_mode("anchor")
                tick.set_horizontalalignment("right")
                tick.set_verticalalignment("center")
                tick.set_fontsize(18)
            ax.tick_params(axis="y", pad=6)
            cbar = ax.collections[0].colorbar
            cbar.formatter = mticker.FuncFormatter(lambda x, pos: fmt_cell(x))
            cbar.ax.tick_params(labelsize=18)
            cbar.update_ticks()
            fig.savefig(
                cm_dir / "most_confused_strict.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            fig.savefig(
                cm_dir / "most_confused_strict.pdf",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)

            # LENIENT
            cm_lenient = confusion_matrix(
                y_true,
                adjusted_preds_lenient,
                labels=labels_with_unclassified,
                normalize="true",
            )
            errors_lenient = 1 - np.diag(cm_lenient[:n_labels, :n_labels])
            top_classes_lenient = np.argsort(errors_lenient)[-top_n:]
            cm_top_lenient = cm_lenient[
                np.ix_(top_classes_lenient, top_classes_lenient.tolist() + [uncl_idx])
            ]
            labels_top_lenient = [real_labels[i] for i in top_classes_lenient]
            labels_top_lenient_plus_uncl = labels_top_lenient + ["unclassified"]
            labels_lenient = np.vectorize(fmt_cell)(cm_top_lenient)

            fig, ax = plt.subplots(figsize=(24, 18))
            sns.heatmap(
                cm_top_lenient,
                xticklabels=labels_top_lenient_plus_uncl,
                yticklabels=labels_top_lenient,
                cmap="YlGnBu",
                annot=labels_lenient,
                fmt="",
                annot_kws={"fontsize": 18},
                square=True,
                ax=ax,
                linewidths=0.5,
                linecolor="white",
                cbar=True,
                cbar_kws={"shrink": 0.6, "aspect": 20},
            )
            ax.set_title(
                "Confusion Matrix – Top {} Most Confused Classes (Lenient)\nConfident + Uncertain Predictions; Unknown → unclassified".format(
                    top_n
                ),
                fontsize=24,
            )
            ax.set_xlabel("Predicted Label", fontsize=20, labelpad=10)
            ax.set_ylabel("True Label", fontsize=20, labelpad=10)
            plt.xticks(rotation=45, ha="right", fontsize=18)
            for tick in ax.get_yticklabels():
                tick.set_rotation(45)
                tick.set_rotation_mode("anchor")
                tick.set_horizontalalignment("right")
                tick.set_verticalalignment("center")
                tick.set_fontsize(18)
            ax.tick_params(axis="y", pad=6)
            cbar = ax.collections[0].colorbar
            cbar.formatter = mticker.FuncFormatter(lambda x, pos: fmt_cell(x))
            cbar.ax.tick_params(labelsize=18)
            cbar.update_ticks()
            fig.savefig(
                cm_dir / "most_confused_lenient.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            fig.savefig(
                cm_dir / "most_confused_lenient.pdf",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)

        except Exception as e:
            logging.warning("Skipping most-confused heatmaps (needs seaborn): %s", e)

    return (
        rep_conf_only,
        rep_conf_plus_unc,
        cov_conf,
        cov_conf_plus_unc,
        classwise_df,
    )


# -----------------------------
# Helpers
# -----------------------------
def sanitize(name: str) -> str:
    """Convert a class name to a filesystem-safe token for plot filenames."""
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^\w.-]", "", s)
    return s


def format_subscripts(text: str) -> str:
    """
    Convert digits after letters/parentheses into Unicode subscripts.
    E.g., 'SiO2'→'SiO₂', 'TiO2'→'TiO₂'.
    """
    subs_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return re.sub(
        r"(?<=[A-Za-z\)])(\d+)", lambda m: m.group(1).translate(subs_map), text
    )


def fmt_cell(v: float) -> str:
    """Format for normalized confusion matrices (0 ≤ v ≤ 1):
    - 0.0 → "0"
    - 1.0 → "1"
    - 0.0 < v < 1.0 → ".xx" (no leading zero)
    """
    r = round(float(v), 2)
    if r == 0.0:
        return "0"
    if r == 1.0:
        return "1"
    return f"{r:.2f}"[1:]  # strip leading zero


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist for the mosaic plot, casting to int."""
    req = [
        "confident_correct",
        "confident_incorrect",
        "uncertain_correct",
        "uncertain_incorrect",
        "unknown",
    ]
    out = df.copy()
    for c in req:
        if c not in out.columns:
            out[c] = 0
    out = out[req].fillna(0)
    for c in req:
        out[c] = out[c].astype(int)
    return out


def split_label_two_lines(s: str, max_len: int = 15) -> str:
    """Split long labels across two lines near a space or punctuation."""
    s = str(s).replace("\u00A0", " ")
    if len(s) <= max_len:
        return s
    cand = [i for i, ch in enumerate(s) if ch in " -/" and 1 < i < len(s) - 1]
    if cand:
        split_idx = min(cand, key=lambda i: abs(i - max_len))
        return s[:split_idx] + "\n" + s[split_idx + 1 :]
    return s[:max_len] + "-\n" + s[max_len:]


def lighten(color, factor=0.6):
    """Lighten an RGBA color by moving it towards white by the given factor."""
    c = np.array(mcolors.to_rgba(color), dtype=float)
    return c + (1 - c) * factor


def plot_classwise_mosaic(
    classwise_df: pd.DataFrame,
    grid_rows: int = 20,
    grid_cols: int = 11,
    waffle_rows: int = 5,
    waffle_cols: int = 10,
    title: str = "Class-Wise Prediction Coverage Mosaic",
    seed: int = 0,
    save_path_png: Optional[Path] = None,
    save_path_pdf: Optional[Path] = None,
):
    """Draw a portrait mosaic of per-class outcomes.

    Each class occupies a 5×10 waffle (50 cells). Incorrect cells reuse the
    corresponding “correct” color and are overlaid with red hatching. Classes
    are sorted so that those with more correct assignments appear first.
    """
    # Colors (lightened so the text label remains prominent)
    viridis = cm.get_cmap("viridis", 10)
    colors = {
        "confident_correct": lighten(viridis(2), 0.6),
        "confident_incorrect": lighten(viridis(2), 0.6),
        "uncertain_correct": lighten(viridis(7), 0.6),
        "uncertain_incorrect": lighten(viridis(7), 0.6),
        "unknown": lighten(viridis(9), 0.6),
    }

    # Prepare data
    classwise_df = ensure_columns(classwise_df)

    # Sort: more "good" first (all descending)
    classwise_df = classwise_df.sort_values(
        by=[
            "confident_correct",
            "confident_incorrect",
            "uncertain_correct",
            "uncertain_incorrect",
            "unknown",
        ],
        ascending=[False, False, False, False, False],
    )

    total_cells = waffle_rows * waffle_cols
    cap = grid_rows * grid_cols
    df_plot = classwise_df.iloc[:cap].copy()
    class_labels = df_plot.index.astype(str).tolist()

    # Rescale to exactly total_cells per class
    rng = np.random.default_rng(seed)
    for idx, row in df_plot.iterrows():
        counts = row.to_numpy(dtype=float)
        tot = counts.sum()
        if tot != total_cells:
            if tot > 0:
                probs = counts / tot
                df_plot.loc[idx] = rng.multinomial(total_cells, probs)
            else:
                df_plot.loc[idx] = [total_cells, 0, 0, 0, 0]

    # Figure (large, print-friendly)
    fig_w, fig_h = 22, 22
    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )

    draw_order = [
        "confident_correct",
        "confident_incorrect",
        "uncertain_correct",
        "uncertain_incorrect",
        "unknown",
    ]

    axes_flat = np.ravel(axes)
    for i, ax in enumerate(axes_flat):
        if i >= len(df_plot):
            ax.axis("off")
            continue

        row = df_plot.iloc[i]
        counts = np.array([row[c] for c in draw_order], dtype=int)

        # Build base sequence (incorrect gets same base color as its "correct" counterpart)
        base_seq = []
        seg_bounds = {}  # [start, end) ranges for each segment
        cursor = 0
        for key, n in zip(draw_order, counts):
            seg_bounds[key] = (cursor, cursor + n)
            if n > 0:
                if key == "confident_incorrect":
                    base_seq.extend([colors["confident_correct"]] * n)
                elif key == "uncertain_incorrect":
                    base_seq.extend([colors["uncertain_correct"]] * n)
                else:
                    base_seq.extend([colors[key]] * n)
                cursor += n

        rgba = np.array(base_seq, dtype=float).reshape(waffle_rows, waffle_cols, 4)

        ax.imshow(rgba, aspect="auto", origin="upper", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        # Red hatching overlay for incorrect segments
        for key in ("confident_incorrect", "uncertain_incorrect"):
            start, end = seg_bounds.get(key, (0, 0))
            for k in range(start, end):
                r, c = divmod(k, waffle_cols)
                ax.add_patch(
                    mpatches.Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor="red",
                        hatch="///",
                        linewidth=0.0,
                        alpha=0.9,
                    )
                )

        # Centered class label
        label = split_label_two_lines(format_subscripts(class_labels[i]))
        ax.text(
            0.5,
            0.5,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            zorder=5,
        )

    # Title
    fig.suptitle(title, fontsize=18, y=0.975)

    # Legend: show incorrect as hatched over the same base color
    legend_handles = [
        mpatches.Patch(
            facecolor=colors["confident_correct"], label="Confident Correct"
        ),
        mpatches.Patch(
            facecolor=colors["confident_correct"],
            hatch="///",
            edgecolor="red",
            label="Confident Incorrect",
        ),
        mpatches.Patch(
            facecolor=colors["uncertain_correct"], label="Uncertain Correct"
        ),
        mpatches.Patch(
            facecolor=colors["uncertain_correct"],
            hatch="///",
            edgecolor="red",
            label="Uncertain Incorrect",
        ),
        mpatches.Patch(facecolor=colors["unknown"], label="Unknown"),
    ]
    fig.legend(
        handles=legend_handles,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        title="Status",
        fontsize=14,
        title_fontsize=16,
    )

    # Tight outer margins
    fig.subplots_adjust(left=0.02, right=0.995, top=0.96, bottom=0.06)

    # Export
    if save_path_png:
        fig.savefig(save_path_png, dpi=300, bbox_inches="tight", facecolor="white")
    if save_path_pdf:
        fig.savefig(save_path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main(args: argparse.Namespace) -> None:
    t0 = time.time()
    set_determinism(GLOBAL_SEED)
    out_dir = Path(args.out_dir).resolve()
    plots_dir = out_dir / "plots"
    diag_dir = out_dir / "diagnostics"
    models_dir = out_dir / "models"
    log_dir = out_dir / "logs"
    for d in (plots_dir, diag_dir, models_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # File logging in addition to console
    fh = logging.FileHandler(log_dir / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.info("Logging initialized. Writing to %s", log_dir / "run.log")

    dump_environment(out_dir)

    # Load data
    logging.info("Reading Excel: %s (sheet=%s)", args.input_xlsx, args.sheet)
    # Note: requires 'openpyxl' for .xlsx
    df = pd.read_excel(args.input_xlsx, sheet_name=args.sheet)

    # Features & target
    features = (
        DEFAULT_FEATURES
        if args.features_json is None
        else json.loads(Path(args.features_json).read_text())
    )
    target = args.target

    validate_columns(df, features + [target], "required")
    df_clean = df.dropna(subset=features + [target]).copy()
    X = df_clean[features]
    y = df_clean[target]
    log_run_params(args=args, out_dir=out_dir, features=features, X=X, y=y)

    # Silence specific warnings from sklearn.covariance
    warnings.filterwarnings(
        "ignore",
        message=re.escape(
            "Determinant has increased; this should not happen: log(det) > log(previous_det)"
        ),
        category=RuntimeWarning,
        module="sklearn.covariance._robust_covariance",
    )
    warnings.filterwarnings(
        "ignore",
        message="The covariance matrix associated to your dataset is not full rank",
        category=UserWarning,
    )

    # 1) Mahalanobis profiles
    m_info, m_thresh = compute_mcd_mahalanobis_profiles(
        X,
        y,
        plot_dir=plots_dir if args.plot else None,
        support_min=0.5,
        support_max=0.75,
        ridge=1e-3,
        seed=GLOBAL_SEED,
        threshold_mode=args.threshold_mode,
        mahal_quantile=args.mahal_quantile,
        mahal_inflate=args.mahal_inflate,
        cov_shrink_alpha=args.cov_shrink_alpha,
    )

    # 2) Confidence thresholds via CV OOF
    conf_thr, unc_thr, rf, oof_probs = compute_confidence_thresholds(
        X,
        y.to_numpy(),
        rf_n_estimators=args.n_estimators,
        cv_folds=args.cv_folds,
        seed=GLOBAL_SEED,
        plot_dir=plots_dir if args.plot else None,
        fallback_conf=args.fallback_conf,
        min_uncertainty_gap=args.min_uncertainty_gap,
    )
    
    # --- Log & persist a per-class threshold summary ---
    try:
        thr_table = (
            pd.DataFrame({"class": sorted(set(y))})
            .assign(
                confident=lambda d: d["class"].map(conf_thr),
                uncertain=lambda d: d["class"].map(unc_thr),
                mahalanobis=lambda d: d["class"].map(m_thresh),
            )
            .sort_values("class")
        )
        logging.info(
            "Per-class thresholds:\n%s",
            thr_table.to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )
        # Use the diagnostics directory you already created
        thr_csv = diag_dir / "thresholds_per_class.csv"
        thr_table.to_csv(thr_csv, index=False)
        logging.info("Saved full thresholds table to %s", thr_csv)
    except Exception as e:
        logging.warning("Could not log threshold summary: %s", e)

    # Refit on full data for the final artifact
    rf_final = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=GLOBAL_SEED, n_jobs=-1
    )
    rf_final.fit(X, y)

    # Save model
    model_path = models_dir / "learning_model.pkl"
    joblib.dump(rf_final, model_path)
    logging.info("Model saved to %s", model_path)

    # Save thresholds JSON
    export = {
        "conf_thresholds": conf_thr,
        "uncertain_thresholds": unc_thr,
        "mahal_thresholds": m_thresh,
        "mahal_thresholding": {
            "mode": args.threshold_mode,
            "quantile": float(args.mahal_quantile),
            "inflate": float(args.mahal_inflate),
            "cov_shrink_alpha": float(args.cov_shrink_alpha),
        },
        "mahal_info": {
            cls: (
                {"mean": info.mean.tolist(), "inv_cov": info.inv_cov.tolist()}
                if info is not None else None
            )
            for cls, info in m_info.items()
        },
        "classes": list(rf_final.classes_),
    }
    thr_path = models_dir / "hybrid_thresholds.json"
    thr_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    logging.info("Thresholds saved to %s", thr_path)

    # 3) Feature importance
    try:
        fi_full = pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": rf_final.feature_importances_.astype(float),
            }
        ).sort_values("importance", ascending=False)

        # Save CSV
        fi_full_csv = diag_dir / "feature_importance_full.csv"
        fi_full.to_csv(fi_full_csv, index=False)
        logging.info("Feature importances (final model) saved to %s", fi_full_csv)

        # --- Plot barh with viridis + value labels ---
        try:
            fi_plot_dir = (
                (plots_dir / "feature_importance")
                if plots_dir
                else (out_dir / "plots" / "feature_importance")
            )
            fi_plot_dir.mkdir(parents=True, exist_ok=True)

            # Sort ascending so largest ends up at the bottom in barh
            fi_plot = fi_full.sort_values("importance", ascending=True).reset_index(
                drop=True
            )
            importances = fi_plot["importance"].to_numpy()
            features_raw = fi_plot["feature"].tolist()

            # Optional: shorten long feature labels
            def shorten(lbl: str) -> str:
                repl = {
                    "constituents": "const.",
                    "constituent": "const.",
                    "highest": "high.",
                    "second": "2nd",
                    "number of": "#",
                    "sum of": "Σ",
                    "value of": "val.",
                }
                for k, v in repl.items():
                    lbl = lbl.replace(k, v)
                    
                # chem subscripts
                return format_subscripts(lbl)

            features = [shorten(f) for f in features_raw]

            fig_h = max(4, 0.35 * len(fi_plot))
            fig, ax = plt.subplots(figsize=(8.5, fig_h))

            # Color by normalized importance so hue reflects value
            norm = plt.Normalize(
                importances.min(), importances.max() if importances.max() > 0 else 1
            )
            cmap = plt.cm.get_cmap("viridis")
            colors = cmap(norm(importances))

            y_pos = np.arange(len(features))
            ax.barh(y_pos, importances, color=colors)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel("Feature importance (Gini gain)")
            ax.set_ylabel("Feature")
            ax.set_title("Feature Importance – Final Random Forest Model")
            ax.grid(axis="x", linestyle="--", alpha=0.4)

            # Right padding so labels don't clip
            ax.set_xlim(0, importances.max() * 1.08 if importances.size else 1.0)

            # Numeric labels at bar ends
            for yi, v in enumerate(importances):
                ax.text(
                    v + (0.01 * ax.get_xlim()[1]),
                    yi,
                    f"{v:.3f}",
                    va="center",
                    fontsize=9,
                )

            plt.tight_layout()
            fig.savefig(
                fi_plot_dir / "feature_importance_full.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            fig.savefig(
                fi_plot_dir / "feature_importance_full.pdf",
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)
            logging.info("Feature importance plot saved to %s", fi_plot_dir)
        except Exception as e:
            logging.exception("Skipping feature importance plot: %s", e)
    except Exception as e:
        logging.exception("Skipping feature importance computation: %s", e)

    # 4) LOCO open-set evaluation
    loco_df = loco_openset_evaluation(
        X,
        y,
        conf_thr,
        unc_thr,
        m_thresh,
        m_info,
        loose_factor=args.loose_factor,
        rf_n_estimators=args.n_estimators,
        seed=GLOBAL_SEED,
    )
    loco_path = diag_dir / "loco_openset_full_rates.csv"
    loco_df.to_csv(loco_path, index=False)
    logging.info("LOCO results saved to %s", loco_path)

    # Save detailed mappings if present
    dmap = loco_df.attrs.get("detailed_mappings")
    if dmap is not None:
        dmap_path = diag_dir / "loco_detailed_mappings.csv"
        dmap.to_csv(dmap_path, index=False)
        logging.info("Detailed LOCO mappings saved to %s", dmap_path)

    # === LOCO stacked bar plots ===
    try:
        loco_plot_dir = (
            (plots_dir / "loco") if plots_dir else (out_dir / "plots" / "loco")
        )
        loco_plot_dir.mkdir(parents=True, exist_ok=True)

        # --- Sort and totals ---
        df_plot = loco_df.sort_values(
            by=["n_confident", "n_uncertain"], ascending=[True, True]
        ).reset_index(drop=True)
        n_classes = len(df_plot)
        if n_classes == 0:
            logging.info("LOCO: no rows to plot; skipping stacked bar plot.")
        else:
            total_conf = int(df_plot["n_confident"].sum())
            total_unc = int(df_plot["n_uncertain"].sum())
            total_unk = int(df_plot["n_unknown"].sum())
            total_all = total_conf + total_unc + total_unk

            perc_conf = 100 * total_conf / max(1, total_all)
            perc_unc = 100 * total_unc / max(1, total_all)
            perc_unk = 100 * total_unk / max(1, total_all)

            summary_text = (
                f"Unknown: {total_unk:,} ({perc_unk:.1f}%)  |  "
                f"Uncertain: {total_unc:,} ({perc_unc:.1f}%)  |  "
                f"Confident: {total_conf:,} ({perc_conf:.1f}%)"
            )

            # --- Colors (viridis) ---
            viridis = cm.get_cmap("viridis", 10)
            unknown_color, uncertain_color, confident_color = (
                viridis(3),
                viridis(6),
                viridis(9),
            )

            # --- Grid setup ---
            n_cols = 4
            rows_per_col = int(np.ceil(n_classes / n_cols))
            fig_w = 4 * n_cols
            fig_h = 0.28 * rows_per_col + 1.2

            # Left margin scales with the longest label (cap at 0.35 of figure width)
            max_label_len = max(len(s) for s in df_plot["class"])
            left_margin = min(0.35, 0.10 + 0.007 * max_label_len)

            fig, axes = plt.subplots(ncols=n_cols, figsize=(fig_w, fig_h), sharex=True)
            fig.patch.set_facecolor("white")
            if n_cols == 1:
                axes = [axes]

            # Bar height and base font size for internal labels
            bar_h = 0.72
            y_fs = 10 if rows_per_col > 40 else 12 if rows_per_col > 25 else 14

            # --- Draw columns ---
            for i in range(n_cols):
                start = i * rows_per_col
                end = min((i + 1) * rows_per_col, n_classes)
                df_chunk = df_plot.iloc[start:end]
                ax = axes[i]

                if df_chunk.empty:
                    ax.axis("off")
                    continue

                # Stacked bars
                ax.barh(
                    df_chunk["class"],
                    df_chunk["n_unknown"],
                    color=unknown_color,
                    height=bar_h,
                    alpha=0.4,
                )
                ax.barh(
                    df_chunk["class"],
                    df_chunk["n_uncertain"],
                    left=df_chunk["n_unknown"],
                    color=uncertain_color,
                    height=bar_h,
                    alpha=0.4,
                )
                ax.barh(
                    df_chunk["class"],
                    df_chunk["n_confident"],
                    left=df_chunk["n_unknown"] + df_chunk["n_uncertain"],
                    color=confident_color,
                    height=bar_h,
                    alpha=0.4,
                )

                # Category axis
                ax.set_ylim(-0.5, len(df_chunk) - 0.5)
                ax.invert_yaxis()
                ax.set_yticks(np.arange(len(df_chunk)))
                ax.set_yticklabels([])

                # Inline labels with fallbacks
                for row_idx, (cls, unk, unc, conf) in enumerate(
                    zip(
                        df_chunk["class"],
                        df_chunk["n_unknown"],
                        df_chunk["n_uncertain"],
                        df_chunk["n_confident"],
                    )
                ):
                    total = float(unk + unc + conf)
                    if total <= 0:
                        continue

                    label_text = format_subscripts(str(cls))
                    x_center, y_text = 0.5 * total, row_idx

                    fs_try = min(18, y_fs + 2)
                    txt = ax.text(
                        x_center,
                        y_text,
                        label_text,
                        ha="center",
                        va="center",
                        fontsize=fs_try,
                        color="black",
                        zorder=5,
                        clip_on=True,
                    )

                    # quick text-fit helpers
                    def text_width_in_data(ax_, txt_artist):
                        fig = ax_.figure
                        fig.canvas.draw()
                        bbox = txt_artist.get_window_extent(
                            renderer=fig.canvas.get_renderer()
                        )
                        (x0, _), (x1, _) = ax_.transData.inverted().transform(
                            [(bbox.x0, bbox.y0), (bbox.x1, bbox.y0)]
                        )
                        return abs(x1 - x0)

                    def text_height_in_data(ax_, txt_artist):
                        fig = ax_.figure
                        fig.canvas.draw()
                        bbox = txt_artist.get_window_extent(
                            renderer=fig.canvas.get_renderer()
                        )
                        (_, y0), (_, y1) = ax_.transData.inverted().transform(
                            [(bbox.x0, bbox.y0), (bbox.x0, bbox.y1)]
                        )
                        return abs(y1 - y0)

                    fits_w = text_width_in_data(ax, txt) <= total * 0.92
                    fits_h = text_height_in_data(ax, txt) <= bar_h * 0.90

                    while (not (fits_w and fits_h)) and fs_try > 7:
                        fs_try -= 1
                        txt.set_fontsize(fs_try)
                        fits_w = text_width_in_data(ax, txt) <= total * 0.92
                        fits_h = text_height_in_data(ax, txt) <= bar_h * 0.90

                    if not (fits_w and fits_h):
                        txt.set_text(split_label_two_lines(label_text))
                        fits_w = text_width_in_data(ax, txt) <= total * 0.92
                        fits_h = text_height_in_data(ax, txt) <= bar_h * 0.90

                    if not (fits_w and fits_h):
                        txt.remove()
                        x_out = total + max(1.0, 0.03 * total)
                        ax.annotate(
                            label_text,
                            xy=(total, y_text),
                            xytext=(x_out, y_text),
                            ha="left",
                            va="center",
                            fontsize=max(8, fs_try),
                            color="black",
                            arrowprops=dict(
                                arrowstyle="-",
                                lw=1.2,
                                color="black",
                                shrinkA=0,
                                shrinkB=2,
                                connectionstyle="arc3,rad=0",
                            ),
                            zorder=6,
                            annotation_clip=True,
                        )

                ax.set_ylabel("Held-out class" if i == 0 else "", fontsize=11)
                ax.set_xlabel("")

            # --- Layout: margins and titles ---
            axes_arr = np.atleast_1d(axes).ravel()
            last_ax = next((ax for ax in reversed(axes_arr) if ax.has_data()), axes_arr[-1])

            # Tight side margins now; bottom will be set dynamically
            fig.subplots_adjust(
                left=max(0.08, left_margin), right=0.97, top=0.95, bottom=0.02, wspace=0.12
            )

            # Column-centered title + summary
            fig.canvas.draw()
            pos = [ax.get_position() for ax in axes_arr]
            left, right = pos[0].x0, pos[-1].x1
            top = max(p.y1 for p in pos)
            center_x = 0.5 * (left + right)

            fig.suptitle(
                "LOCO Open-Set Counts",
                x=center_x,
                y=min(0.98, top + 0.035),
                fontsize=14,
                ha="center",
            )
            fig.text(
                center_x,
                min(0.96, top + 0.017),
                summary_text,
                ha="center",
                va="center",
                fontsize=11,
                transform=fig.transFigure,
            )

            # --- Dynamic, overlap-safe spacing for legend + global x-label ---
            renderer = fig.canvas.get_renderer()
            tick_heights = [
                lbl.get_window_extent(renderer=renderer).height
                for lbl in last_ax.get_xticklabels()
                if lbl.get_text()
            ]
            tick_h_px = max(tick_heights) if tick_heights else 12

            fig_w_in, fig_h_in = fig.get_size_inches()
            px_per_in = fig.dpi
            tick_h_fig = (tick_h_px / px_per_in) / fig_h_in

            TICK_MULT = 1.6
            EXTRA_PAD_FIG = 0.012
            gap_fig = tick_h_fig * TICK_MULT + EXTRA_PAD_FIG

            # apply final bottom margin
            fig.subplots_adjust(bottom=max(0.02, gap_fig))
            fig.canvas.draw()

            ax_pos = last_ax.get_position()
            gap_axes = gap_fig / ax_pos.height

            # Legend just below the last column
            leg = last_ax.legend(
                ["Unknown", "Uncertain", "Confident"],
                loc="upper right",
                bbox_to_anchor=(1.0, -gap_axes),
                bbox_transform=last_ax.transAxes,
                ncol=3,
                frameon=False,
                fontsize=11,
                borderaxespad=0.0,
            )
            for h in leg.legendHandles:
                h.set_alpha(0.4)
                h.set_edgecolor("none")

            # Global x-axis label
            fig.canvas.draw()
            lb = leg.get_window_extent(renderer=renderer)
            (x0, y0), (x1, y1) = fig.transFigure.inverted().transform(
                [(lb.x0, lb.y0), (lb.x1, lb.y1)]
            )
            x_text = 0.5 * (fig.subplotpars.left + x0)
            y_text = 0.5 * (y0 + y1)
            fig.text(
                x_text, y_text, "Number of samples", ha="center", va="center", fontsize=11
            )

            # --- Save ---
            fig.savefig(
                loco_plot_dir / "LOCO_OpenSet_Counts.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            fig.savefig(
                loco_plot_dir / "LOCO_OpenSet_Counts.pdf",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)
            logging.info("LOCO stacked bar plots saved to %s", loco_plot_dir)

    except Exception as e:
        logging.exception("Skipping LOCO stacked bar plots: %s", e)

    # 5) Coverage-aware evaluation & exports (reports, tables, plots)
    rf_classes_global = np.unique(y.to_numpy())
    (
        rep_conf_only,
        rep_conf_plus_unc,
        cov_conf,
        cov_conf_plus_unc,
        classwise_df,
    ) = coverage_aware_metrics(
        X,
        y.tolist(),
        oof_probs,
        rf_classes_global,
        conf_thr,
        unc_thr,
        m_thresh,
        m_info,
        loose_factor=args.loose_factor,
        plot_dir=(plots_dir if args.plot else None),
    )

    # Persist classification reports
    report_path = diag_dir / "open_set_metrics.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Coverage-aware classification report\n\n")
        f.write("Confident-only predictions (unclassified counted as errors):\n")
        f.write(rep_conf_only + "\n")
        f.write(
            "\nConfident + Uncertain predictions (unclassified counted as errors):\n"
        )
        f.write(rep_conf_plus_unc + "\n")
        f.write("\n--- Final Open-Set Evaluation Summary ---\n")
        f.write(f"Coverage (Confident Only): {cov_conf:.2%}\n")
        f.write(f"Coverage (Confident + Uncertain): {cov_conf_plus_unc:.2%}\n")
    logging.info("Coverage-aware metrics written to %s", report_path)

    # Save class-wise coverage table (counts and rates per class)
    cw_path = diag_dir / "classwise_coverage.csv"
    classwise_df.to_csv(cw_path, index=True)
    logging.info("Class-wise coverage table saved to %s", cw_path)

    logging.info("Total runtime: %.2f s", time.time() - t0)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hybrid Open-Set Mineral Classification Pipeline (publication-ready)."
    )
    p.add_argument(
        "--input_xlsx",
        type=str,
        required=True,
        help="Path to input Excel file containing features and target (requires 'openpyxl').",
    )
    p.add_argument(
        "--sheet", type=str, default="data", help="Worksheet name (default: data)."
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory for outputs (models, thresholds, plots, reports).",
    )
    p.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help=f"Target column name (default: {DEFAULT_TARGET}).",
    )
    p.add_argument(
        "--features_json",
        type=str,
        default=None,
        help="Optional path to a JSON file containing a list of feature names.",
    )
    p.add_argument(
        "--n_estimators",
        type=int,
        default=RF_N_ESTIMATORS,
        help=f"RandomForest n_estimators (default: {RF_N_ESTIMATORS}).",
    )
    p.add_argument(
        "--cv_folds",
        type=int,
        default=CV_FOLDS,
        help=f"Number of CV folds for OOF (default: {CV_FOLDS}).",
    )
    p.add_argument(
        "--fallback_conf",
        type=float,
        default=0.8,
        help="Fallback confident threshold if rule fails or is too high (default: 0.8).",
    )
    p.add_argument(
        "--min_uncertainty_gap",
        type=float,
        default=0.2,
        help="Minimum gap between confident and uncertain thresholds (default: 0.2).",
    )
    p.add_argument(
        "--threshold_mode",
        choices=["max", "quantile", "inflated_quantile"],
        default="inflated_quantile",
        help="How to derive per-class Mahalanobis limit: "
             "'max', 'quantile', or 'inflated_quantile' (default).",
    )
    p.add_argument(
        "--mahal_quantile",
        type=float,
        default=0.95,
        help="Quantile q (0 < q < 1) for quantile-based modes (default: 0.95).",
    )
    p.add_argument(
        "--mahal_inflate",
        type=float,
        default=1.05,
        help="Multiplicative factor for 'inflated_quantile' mode (default: 1.05).",
    )
    p.add_argument(
        "--cov_shrink_alpha",
        type=float,
        default=0.10,
        help="Diagonal shrinkage toward I for MCD covariance (0–1). 0 disables.",
    )
    p.add_argument(
        "--loose_factor",
        type=float, default=2.0,
        help="Mahalanobis relaxation for 'uncertain' band (default: 2.0)."
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save diagnostic plots."
    )
    return p


# -----------------------------
# Standard CLI entry-point
# -----------------------------
# We keep argument parsing separate via 'build_argparser()' and wrap execution
# in a try/except block so failures are logged clearly and CI receives a non-zero exit code.
if __name__ == "__main__":
    try:
        args = build_argparser().parse_args()
        main(args)
    except SystemExit:
        raise
    except Exception:
        logging.exception("Fatal error")
        sys.exit(1)
