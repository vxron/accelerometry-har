#!/usr/bin/env python3 
"""
Train Activity SVMs (static vs dynamic) with feature selection and ONNX export.

This script:
(1) Loads a CSV (columns: tick,x,y,z,active)
(2) Detects contiguous 'active == 1' blocks and assigns labels in fixed order:
    sit -> stand -> walk -> turn -> (repeat)
(3) Builds sliding windows (win_len, hop) ONLY if fully inside an activity block,
    with guard margins trimmed from both block edges to avoid toggle jitter
(4) Extracts features (time + frequency); standardizes inside the model pipeline
(5) Learns an intensity-based pre-classifier (SMA) to route:
        static {sit, stand} vs dynamic {walk, turn}
(6) Trains two SVMs (RBF kernels): activity_svm_static and activity_svm_dynamic
    - Feature selection: SelectKBest(MI) -> correlation filter -> SFS wrapper
(7) Exports both models to ONNX; also saves a meta.json containing feature lists
    and SMA threshold. ZipMap disabled for easy C++ tensor outputs.

CLI:
  python train_activity_svm.py \
      --csv build/accel_calib_data.csv \
      --win 200 --hop 100 --guard 50 \
      --fs 119 --outdir build --use-pca 0

Dependencies:
  pip install numpy pandas scikit-learn scipy skl2onnx onnx
"""

#!/usr/bin/env python3
"""
If --data_dir exists, we auto-load all *.csv inside (each file = one subject).
Fallback: single --csv path.

Run:
  python trainSVM.py --data_dir data --win 200 --hop 100 --guard 50 --fs 119 --outdir models
"""

# ***** imports *****
import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from numpy.linalg import lstsq
from scipy.signal import butter, filtfilt, medfilt, welch
from scipy.stats import skew, kurtosis

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, StratifiedKFold, GridSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Reporting / plots
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import pathlib

# ONNX export (optional)
try:
    import onnx  # noqa: F401
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except Exception:
    convert_sklearn = None

# ***** config (defaults) *****
STATIC_CLASSES = ["stand", "sit"]
DYNAMIC_CLASSES = ["walk", "turn"]
# NOTE: Header comment says sit->stand->walk->turn; the actual order used here is:
ACTIVITY_ORDER  = ["stand", "walk", "sit", "turn"]

FS    = 119.0   # Hz (CLI overridable)
WIN   = 200     # samples (CLI)
HOP   = 100     # samples (CLI)
GUARD = 50      # samples (CLI)

AR_ORDER = 4
RNG      = 13
np.random.seed(RNG)

# feature-selection defaults (inner CV can move these)
K_BEST_STATIC  = 12
K_BEST_DYNAMIC = 18
SFS_K_STATIC   = 2
SFS_K_DYNAMIC  = 3


# ------------------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------------------
def _svm_summary(pipeline: Pipeline) -> Dict:
    svm = pipeline.named_steps["svm"]
    info = {
        "kernel": getattr(svm, "kernel", None),
        "C": getattr(svm, "C", None),
        "gamma": getattr(svm, "gamma", None),
        "degree": getattr(svm, "degree", None),
        "coef0": getattr(svm, "coef0", None),
        "n_support": getattr(svm, "n_support_", None).tolist() if hasattr(svm, "n_support_") else None,
        "n_sv_total": int(getattr(svm, "support_vectors_", np.empty((0,))).shape[0]) if hasattr(svm, "support_vectors_") else None,
    }
    if getattr(svm, "kernel", None) == "linear" and hasattr(svm, "coef_"):
        info["linear_w_shape"] = tuple(svm.coef_.shape)
        info["linear_b_shape"] = tuple(svm.intercept_.shape)
        info["linear_w"] = svm.coef_.tolist()
        info["linear_b"] = svm.intercept_.tolist()
    return info

def _selector_summary(pipeline: Pipeline) -> Dict:
    info = {
        "selected_features": [],
        "sfs_order": [],
        "mi_scores_for_selected": {}
    }
    try:
        sel = pipeline.named_steps["selector"]
        if hasattr(sel, "selected_names_") and sel.selected_names_:
            info["selected_features"] = sel.selected_names_
        if hasattr(sel, "sfs_selected_names_order_") and sel.sfs_selected_names_order_:
            info["sfs_order"] = sel.sfs_selected_names_order_
        if hasattr(sel, "mi_scores_") and sel.mi_scores_:
            # only report MI for the final selected ones (cleaner)
            for n in info["selected_features"]:
                info["mi_scores_for_selected"][n] = sel.mi_scores_.get(n, None)
    except Exception:
        pass
    return info

def _plot_confusion(cm: np.ndarray, labels: List[str], outpath: str, title: str):
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=140)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def _maybe_plot_feature_space_and_svm(
    pipeline_fit: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    outdir: pathlib.Path,
    label: str
) -> List[str]:
    """
    If the final SVM sees 2 or 3 features, plot them properly.
    We transform X using the full pre-SVM stack (scaler -> pca -> selector),
    then call svm on grid points *in that same space*.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    imgs = []

    if "svm" not in pipeline_fit.named_steps:
        return imgs

    # Split pipeline into pre-SVM and the SVM itself
    pre = Pipeline([(n, s) for (n, s) in pipeline_fit.steps if n != "svm"])
    svm = pipeline_fit.named_steps["svm"]

    # Project training data into the exact feature space SVM was trained on
    try:
        Z = pre.transform(X)     # shape: (n_samples, k)
    except Exception:
        return imgs
    if Z.ndim != 2 or Z.shape[1] not in (2, 3):
        return imgs

    k = Z.shape[1]

    # Try to recover human-friendly axis names from the selector
    sel = pipeline_fit.named_steps.get("selector", None)
    if sel is not None and getattr(sel, "selected_names_", None):
        axis_names = sel.selected_names_
    else:
        axis_names = [f"f{i+1}" for i in range(k)]

    # Encode labels for color mapping
    yu = list(np.unique(y))
    y_map = {c: i for i, c in enumerate(yu)}
    yc = np.array([y_map[v] for v in y], dtype=int)

    # Color map (Matplotlib >=3.7 API)
    n_classes = len(yu)
    try:
        # Newer API: get the colormap, then resample to n_classes
        base_cmap = plt.colormaps.get_cmap("tab10").resampled(n_classes)
    except Exception:
        # Older API fallback (has LUT/N parameter)
        base_cmap = plt.get_cmap("tab10", n_classes)
    class_colors = [base_cmap(i) for i in range(n_classes)]
    cm_points = ListedColormap(class_colors)

    def pad_range(a, frac=0.08):
        lo, hi = float(np.min(a)), float(np.max(a))
        pad = (hi - lo) * frac if hi > lo else 1.0
        return lo - pad, hi + pad

    # ---------------- 2D ----------------
    if k == 2:
        z1, z2 = Z[:, 0], Z[:, 1]
        xlim = pad_range(z1); ylim = pad_range(z2)
        xx, yy = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 400),
            np.linspace(ylim[0], ylim[1], 400)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        # For binary SVC: decision_function -> (n,), threshold at 0
        # For multiclass SVC: decision_function -> (n, n_classes) or OvO dims; fall back to predict.
        try:
            df = svm.decision_function(grid)
            if df.ndim == 1:
                Zpred = (df > 0).astype(int)
            else:
                # If OvR style scores exist, take argmax; otherwise just predict
                Zpred = np.argmax(df, axis=1)
        except Exception:
            Zpred = svm.predict(grid)
            Zpred = np.array([y_map.get(lbl, 0) for lbl in Zpred], dtype=int)  # map labels to [0..C-1]
        Zpred = Zpred.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
        ax.contourf(xx, yy, Zpred, alpha=0.15, cmap=cm_points, levels=np.arange(len(yu)+1)-0.5)

        # decision boundary (binary only) where df=0
        try:
            df0 = svm.decision_function(grid)
            if df0.ndim == 1:
                ax.contour(xx, yy, df0.reshape(xx.shape), levels=[0], linewidths=1.2)
        except Exception:
            pass

        sc = ax.scatter(z1, z2, c=yc, s=16, cmap=cm_points, edgecolor='k', linewidths=0.3, alpha=0.9)

        if hasattr(svm, "support_vectors_") and svm.support_vectors_.shape[1] == 2:
            sv = svm.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], s=36, facecolors='none', edgecolors='k', linewidths=1.0, label='SV')

        ax.set_xlabel(axis_names[0]); ax.set_ylabel(axis_names[1])
        ax.set_title(f"{label} – 2D feature space with SVM")

        # Legend
        handles = [plt.Line2D([0],[0], marker='o', color='w',
                              markerfacecolor=class_colors[i], markeredgecolor='k', markersize=6,
                              label=str(c)) for i, c in enumerate(yu)]
        if hasattr(svm, "support_vectors_"):
            handles.append(plt.Line2D([0],[0], marker='o', color='w',
                                      markerfacecolor='none', markeredgecolor='k', markersize=6, label='SV'))
        ax.legend(handles=handles, loc='best', frameon=True)

        fn = f"{label.lower()}_viz2d.png"
        fig.tight_layout(); fig.savefig(outdir / fn, bbox_inches="tight"); plt.close(fig)
        imgs.append(fn)

    # ---------------- 3D ----------------
    if k == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        z1, z2, z3 = Z[:, 0], Z[:, 1], Z[:, 2]
        fig = plt.figure(figsize=(8, 7), dpi=140)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(z1, z2, z3, c=yc, s=14, cmap=cm_points, depthshade=True,
                   edgecolor='k', linewidths=0.2)

        if hasattr(svm, "support_vectors_") and svm.support_vectors_.shape[1] == 3:
            sv = svm.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], sv[:, 2], s=36, facecolors='none', edgecolors='k',
                       linewidths=1.0, label='SV')

        ax.set_xlabel(axis_names[0]); ax.set_ylabel(axis_names[1]); ax.set_zlabel(axis_names[2])
        ax.set_title(f"{label} – 3D feature space")

        handles = [plt.Line2D([0],[0], marker='o', color='w',
                              markerfacecolor=class_colors[i], markeredgecolor='k', markersize=6,
                              label=str(c)) for i, c in enumerate(yu)]
        if hasattr(svm, "support_vectors_"):
            handles.append(plt.Line2D([0],[0], marker='o', color='w',
                                      markerfacecolor='none', markeredgecolor='k', markersize=6, label='SV'))
        ax.legend(handles=handles, loc='best', frameon=True)

        fn3 = f"{label.lower()}_viz3d.png"
        fig.tight_layout(); fig.savefig(outdir / fn3, bbox_inches="tight"); plt.close(fig)
        imgs.append(fn3)

    return imgs


def generate_report(
    X, y, groups, feature_names, pipeline, label, outdir, outer_n_splits=5,
    best_params: dict | None = None,   # NEW: pass final best params here
    extra_notes: list[str] | None = None
):

    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Normalize y as plain 1D array of str
    y = np.asarray(pd.Series(y).astype(str).to_numpy().reshape(-1))
    classes = sorted(pd.Series(y).unique().tolist())
    min_class_count = int(pd.Series(y).value_counts().min())

    # Helper to ensure ≥2 classes per fold in both train & test
    def _validate_cv(splitter, use_groups_flag, X_, y_, groups_):
        for tr_idx, te_idx in splitter.split(X_, y_, groups_ if use_groups_flag else None):
            if np.unique(y_[tr_idx]).size < 2 or np.unique(y_[te_idx]).size < 2:
                return False
        return True

    # Build CV: prefer StratifiedGroupKFold if group coverage allows; else StratifiedKFold
    use_groups = False
    cv = None
    if groups is not None and len(np.unique(groups)) >= 2:
        df_cv = pd.DataFrame({"y": y, "g": groups})
        grp_per_class = df_cv.groupby("y")["g"].nunique().reindex(classes, fill_value=0)
        min_groups_per_class = int(grp_per_class.min())
        if min_groups_per_class >= 2:
            desired = max(2, min(outer_n_splits, min_groups_per_class))
            valid = None
            for k in range(desired, 1, -1):
                cand = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=RNG)
                if _validate_cv(cand, True, X, y, groups):
                    valid = cand; used_k = k; break
            if valid is not None:
                cv = valid; use_groups = True
    if cv is None:
        desired = max(2, min(outer_n_splits, min_class_count))
        for k in range(desired, 1, -1):
            cand = StratifiedKFold(n_splits=k, shuffle=True, random_state=RNG)
            if _validate_cv(cand, False, X, y, None):
                cv = cand; use_groups = False; break
        if cv is None:
            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RNG)
            use_groups = False

    # --- describe the CV splitter (build cv_desc here) ---
    try:
        n_splits = getattr(cv, "n_splits", None) or cv.get_n_splits()
    except Exception:
        n_splits = "?"
    cv_name = cv.__class__.__name__
    cv_desc = f"{cv_name}(n_splits={n_splits})"
    if groups is not None and len(np.unique(groups)) > 1:
        cv_desc += " [group-aware]" if "Group" in cv_name else " [window-stratified]"

    # --- cross-validated predictions & scores ---
    if use_groups:
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, groups=groups, method="predict", n_jobs=-1)
        scores = cross_val_score(pipeline, X, y, cv=cv, groups=groups, n_jobs=-1)
    else:
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict", n_jobs=-1)
        scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)

    # Confusion + classification report
    cm  = confusion_matrix(y, y_pred, labels=classes)
    rep = classification_report(y, y_pred, labels=classes, target_names=classes, digits=3)

    # Save confusion plot (the Markdown later embeds this)
    _plot_confusion(cm, classes, str(pathlib.Path(outdir) / f"{label.lower()}_confusion.png"), f"{label} – Confusion Matrix")

    # FINAL REFIT for reporting with feature names (important for selector)
    pipeline_fit = clone(pipeline).fit(X, y, selector__feature_names=feature_names)
    train_acc = float(pipeline_fit.score(X, y))

    # Summaries
    svm_info = _svm_summary(pipeline_fit)
    selector_info = _selector_summary(pipeline_fit)
    viz_imgs = _maybe_plot_feature_space_and_svm(pipeline_fit, X, y, outdir, label)

    # ----- NEW: human-readable decisions -----
    steps = dict(pipeline_fit.named_steps)
    pca_step = steps.get("pca", "passthrough")
    pca_used = (pca_step != "passthrough")

    # Pretty list of selected features (fallback if missing)
    selected_names = selector_info.get("selected_features") or []
    selected_names_txt = ", ".join(selected_names) if selected_names else "_<none>_"

    # CV splitter description for clarity
    cv_used = cv_desc  # from your existing code

    # Class balance
    cls_counts = pd.Series(y).value_counts().to_dict()

    # ----- Build Markdown -----
    md = []
    md.append(f"# {label} Model Report")
    md.append("")
    md.append("## Dataset Summary")
    md.append(f"- Samples: **{len(y)}** across classes {sorted(set(y.tolist()))}")
    md.append(f"- Class counts: `{cls_counts}`")
    md.append(f"- Grouping for CV: **{cv_used}**")
    md.append("")

    md.append("## Cross-Validation Results")
    md.append(f"- Outer-CV accuracy: **{scores.mean():.3f} ± {scores.std():.3f}**")
    md.append(f"- Train accuracy (refit on all): **{train_acc:.3f}**")
    md.append("")

    # Decisions & rationale: what was compared and what won
    md.append("## Decisions & Rationale")
    # PCA decision
    pca_choice = "ENABLED (PCA(0.95))" if pca_used else "DISABLED (passthrough)"
    md.append(f"- **PCA choice**: {pca_choice}. We compared `passthrough` vs `PCA(0.95)` inside inner-CV and chose the one with higher CV accuracy.")
    # Feature selector decision
    md.append(f"- **Feature selection**: mutual-information filter (k=`selector__k_best`), correlation filter (|r|>`selector__corr_thr`), and SFS wrapper (target k=`selector__sfs_k`).")
    # Final SVM hyper-params
    md.append(f"- **Final SVM (RBF)**: { {k:v for k,v in svm_info.items() if k in ('kernel','C','gamma')} }")
    if best_params is not None:
        md.append(f"- **Best parameter combination (inner-CV winner)**: `{best_params}`")

    # Selected features
    md.append("")
    md.append("## Selected Features (final SVM inputs)")
    md.append(f"- {selected_names_txt}")

    # Optional MI/SFS details if you captured them
    if selector_info.get("sfs_order"):
        md.append(f"- **SFS order**: {', '.join(selector_info['sfs_order'])}")
    if selector_info.get("mi_scores_for_selected"):
        md.append("- **Mutual Information scores (selected)**:")
        for n, s in selector_info["mi_scores_for_selected"].items():
            md.append(f"  - {n}: {('None' if s is None else f'{s:.4f}')}")

    # Confusion matrix (already plotted)
    md.append("")
    md.append("## Confusion Matrix (outer-CV predictions)")
    md.append(f"![]({label.lower()}_confusion.png)")

    # Classification report
    md.append("")
    md.append("## Classification Report (precision/recall/F1)")
    md.append("```")
    md.append(rep.strip())
    md.append("```")

    if viz_imgs:
        md.append("")
        md.append("## Feature-space Visualization")
        for fn in viz_imgs:
            md.append(f"![]({fn})")

    if extra_notes:
        md.append("")
        md.append("## Notes")
        for n in extra_notes:
            md.append(f"- {n}")

    md_txt = "\n".join(md)
    with open(outdir / f"{label.lower()}_report.md", "w", encoding="utf-8") as f:
        f.write(md_txt)

    print(f"[REPORT] Wrote {label} report: {outdir / (label.lower() + '_report.md')}")


# ------------------------------------------------------------------------
# Data & labeling
# ------------------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["x", "y", "z", "active"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {path}")
    return df

def find_active_blocks(active: np.ndarray) -> List[Tuple[int, int]]:
    blocks, in_run, start = [], False, 0
    for idx, val in enumerate(active):
        if val == 1 and not in_run:
            in_run, start = True, idx
        elif val == 0 and in_run:
            blocks.append((start, idx))
            in_run = False
    if in_run:
        blocks.append((start, len(active)))
    return blocks

def windows_from_block(start: int, end: int, win: int, hop: int, guard: int) -> List[Tuple[int, int]]:
    s, e = start + guard, end - guard
    spans, i = [], s
    while i + win <= e:
        spans.append((i, i + win))
        i += hop
    return spans

def assign_labels_to_blocks(n: int, one_blocks: List[Tuple[int, int]], order: List[str]) -> np.ndarray:
    labels = np.array([""] * n, dtype=object)
    for k, (a, b) in enumerate(one_blocks):
        cls = order[k % len(order)]
        labels[a:b] = cls
    return labels

def rotate_activity_order(start_activity: str, base_order: List[str]) -> List[str]:
    if not start_activity or start_activity not in base_order:
        return base_order
    i = base_order.index(start_activity)
    return base_order[i:] + base_order[:i]

def build_windows(df: pd.DataFrame, subject_id: str, start_activity: str = "") -> List[Dict]:
    """
    Returns windows with: s,e,x,y,z,label,subject_id,block_id,block_uid
    """
    x = df["x"].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    z = df["z"].to_numpy(dtype=np.float32)
    active = df["active"].astype(int).to_numpy()

    blocks = find_active_blocks(active)
    order = rotate_activity_order(start_activity, ACTIVITY_ORDER)
    labels_series = assign_labels_to_blocks(len(active), blocks, order=order)

    windows = []
    for block_id, (a, b) in enumerate(blocks):
        lab = labels_series[a] if a < len(labels_series) else ""
        if lab == "":
            continue
        for s, e in windows_from_block(a, b, WIN, HOP, GUARD):
            if (labels_series[s:e] == lab).all():
                windows.append({
                    "s": s, "e": e,
                    "x": x[s:e], "y": y[s:e], "z": z[s:e],
                    "label": lab,
                    "subject_id": subject_id,
                    "block_id": block_id,
                    "block_uid": f"{subject_id}#blk{block_id}"
                })
    return windows

def collect_windows_from_dir(data_dir: str, start_activity: str = "") -> List[Dict]:
    p = Path(data_dir)
    csvs = sorted(p.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")

    all_windows: List[Dict] = []
    for csv_path in csvs:
        subject_id = csv_path.stem
        df = load_csv(str(csv_path))
        ws = build_windows(df, subject_id=subject_id, start_activity=start_activity)
        all_windows.extend(ws)
        print(f"[LOAD] {csv_path.name}: {len(ws)} windows")
    print(f"[LOAD] Total windows from {data_dir}: {len(all_windows)}")
    return all_windows

def collect_windows_from_person_dir(data_root: str, person: str, start_activity: str = "") -> List[Dict]:
    base = Path(data_root) / person
    csvs = sorted(base.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found for person '{person}' under {base}")

    all_windows: List[Dict] = []
    for csv_path in csvs:
        df = load_csv(str(csv_path))
        ws = build_windows(df, subject_id=person, start_activity=start_activity)
        all_windows.extend(ws)
        print(f"[LOAD:{person}] {csv_path.name}: {len(ws)} windows")
    print(f"[LOAD:{person}] Total windows from {base}: {len(all_windows)}")
    return all_windows

# ------------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------------
def robust_mad_clip(arr: np.ndarray, k: float = 6.0) -> np.ndarray:
    if arr.size == 0:
        return arr
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    cutoff = k * (mad if mad > 1e-9 else np.std(arr) + 1e-9)
    lo, hi = med - cutoff, med + cutoff
    return np.clip(arr, lo, hi)

def highpass_butter(x: np.ndarray, fs: float, fc: float = 0.5, order: int = 2) -> np.ndarray:
    s = np.asarray(x, dtype=np.float64)
    nmin = order * 3
    if s.size < max(8, nmin):
        return s - np.mean(s)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    wn = fc / (0.5 * fs)
    wn = float(min(max(wn, 1e-5), 0.999))
    try:
        b, a = butter(order, wn, btype='highpass')
        return filtfilt(b, a, s, method="gust")
    except Exception:
        s0 = s - np.mean(s)
        try:
            d = np.diff(s0, prepend=s0[0])
            return d - np.mean(d)
        except Exception:
            return s0

def preprocess_window(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                      fs: float,
                      denoise: bool = True,
                      gravity_hp_hz: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = robust_mad_clip(np.asarray(x, dtype=np.float64))
    y = robust_mad_clip(np.asarray(y, dtype=np.float64))
    z = robust_mad_clip(np.asarray(z, dtype=np.float64))
    if denoise:
        k = 5 if len(x) >= 5 else 3
        if k % 2 == 0: k += 1
        x = medfilt(x, kernel_size=k)
        y = medfilt(y, kernel_size=k)
        z = medfilt(z, kernel_size=k)
    x = highpass_butter(x, fs, fc=gravity_hp_hz)
    y = highpass_butter(y, fs, fc=gravity_hp_hz)
    z = highpass_butter(z, fs, fc=gravity_hp_hz)
    return x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)

# ------------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------------
def mag_from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.sqrt(x * x + y * y + z * z)

def safe_var(x: np.ndarray) -> float:
    return float(np.var(x)) if x.size else 0.0

def axis_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    aa, bb = a - np.mean(a), b - np.mean(b)
    sa, sb = np.std(aa), np.std(bb)
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return float(np.mean(aa * bb) / (sa * sb))

def bandpower_welch(sig: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    nperseg = int(min(sig.size, 128))
    if nperseg < 16:
        return 0.0
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg)
    m = (f >= f_lo) & (f < f_hi)
    return float(np.trapezoid(Pxx[m], f[m])) if np.any(m) else 0.0

def safe_welch(sig: np.ndarray, fs: float, nperseg: int = 128):
    s = np.asarray(sig, dtype=np.float64)
    if s.size < 8:
        return np.array([0.0]), np.array([0.0])
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(np.isfinite(s)):
        return np.array([0.0]), np.array([0.0])
    if np.allclose(s, s[0], atol=1e-12):
        return np.array([0.0]), np.array([0.0])
    nps = int(max(8, min(nperseg, s.size)))
    try:
        f, Pxx = welch(s, fs=fs, nperseg=nps)
        return f, Pxx
    except Exception:
        return np.array([0.0]), np.array([0.0])

def peak_freq(sig: np.ndarray, fs: float) -> float:
    nperseg = int(min(sig.size, 256))
    if nperseg < 16:
        return 0.0
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg)
    return float(f[int(np.argmax(Pxx))])

def ar_fit_residuals(sig: np.ndarray, p: int) -> Tuple[np.ndarray, float]:
    x = sig.astype(np.float64)
    x = x - x.mean()
    if x.size <= p + 2:
        return np.zeros(p, dtype=np.float64), 0.0
    N = x.size
    Y = x[p:]
    X = np.column_stack([x[p - i - 1: N - i - 1] for i in range(p)])
    a, *_ = lstsq(X, Y, rcond=None)
    resid = Y - X @ a
    return a, float(np.var(resid))

def feature_static_core(x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float) -> Tuple[List[float], List[str]]:
    names, vals = [], []
    mag = mag_from_xyz(x, y, z)
    vals += [float(skew(mag, bias=False)), float(kurtosis(mag, fisher=True, bias=False))]
    names += ["skewness", "kurtosis"]
    a, res_var = ar_fit_residuals(mag, AR_ORDER)
    for i, coeff in enumerate(a, 1):
        vals.append(float(coeff)); names.append(f"ar_a{i}")
    vals.append(res_var); names.append("ar_res_var")
    vals.append(res_var / (safe_var(mag) + 1e-12)); names.append("ar_white_noise_rate")
    for arr, nm in [(x,"x"),(y,"y"),(z,"z")]:
        vals += [float(np.mean(arr)), float(np.std(arr))]
        names += [f"mean_{nm}", f"std_{nm}"]
    vals += [axis_corr(x,y), axis_corr(y,z), axis_corr(z,x)]
    names += ["corr_xy","corr_yz","corr_zx"]
    return vals, names

def feature_dynamic_core(x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float) -> Tuple[List[float], List[str]]:
    names, vals = [], []
    mag = mag_from_xyz(x, y, z)

    if mag.size < 16 or np.allclose(mag, mag[0], atol=1e-12):
        return [0.0]*9, [
            "psd_median","psd_mean","psd_min",
            "fft_mean","fft_var","fft_std",
            "bp_0p5_3","bp_3_10","peak_freq"
        ]

    nperseg = int(min(mag.size, 128))
    f, Pxx = safe_welch(mag, fs=fs, nperseg=nperseg)

    if f.size < 2 or Pxx.size < 2:
        spec = np.abs(np.fft.rfft(mag - mag.mean()))
        return [
            0.0, 0.0, 0.0,
            float(np.mean(spec)), float(np.var(spec)), float(np.std(spec)),
            0.0, 0.0, 0.0
        ], [
            "psd_median","psd_mean","psd_min",
            "fft_mean","fft_var","fft_std",
            "bp_0.5_3","bp_3_10","peak_freq"
        ]

    psd_median = float(np.median(Pxx))
    psd_mean   = float(np.mean(Pxx))
    psd_min    = float(np.min(Pxx))
    vals += [psd_median, psd_mean, psd_min]
    names += ["psd_median","psd_mean","psd_min"]

    spec = np.abs(np.fft.rfft(mag - mag.mean()))
    vals += [float(np.mean(spec)), float(np.var(spec)), float(np.std(spec))]
    names += ["fft_mean","fft_var","fft_std"]

    m1 = (f >= 0.5) & (f < 3.0)
    m2 = (f >= 3.0) & (f < 10.0)
    bp1 = float(np.trapezoid(Pxx[m1], f[m1])) if np.any(m1) else 0.0
    bp2 = float(np.trapezoid(Pxx[m2], f[m2])) if np.any(m2) else 0.0
    vals += [bp1, bp2]
    names += ["bp_0p5_3","bp_3_10"]

    peak_f = float(f[int(np.argmax(Pxx))]) if Pxx.size else 0.0
    vals += [peak_f]
    names += ["peak_freq"]

    return vals, names

def compute_features(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     fs: float,
                     mode: str) -> Tuple[List[float], List[str]]:
    x_p, y_p, z_p = preprocess_window(x, y, z, fs=fs, denoise=True, gravity_hp_hz=0.5)
    if mode == "static":
        return feature_static_core(x_p, y_p, z_p, fs)
    elif mode == "dynamic":
        return feature_dynamic_core(x_p, y_p, z_p, fs)
    else:
        raise ValueError("mode must be 'static' or 'dynamic'")

# ------------------------------------------------------------------------
# In-pipeline feature selection
# ------------------------------------------------------------------------
def corr_filter(X: np.ndarray, names: List[str], thr: float = 0.90) -> Tuple[np.ndarray, List[str], List[int]]:
    n, d = X.shape
    if d <= 1:
        return X, names, list(range(d))

    X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    std = X.std(axis=0)
    eps = 1e-15
    nonconst_mask = std > eps
    keep_initial = np.where(nonconst_mask)[0].tolist()

    if len(keep_initial) <= 1:
        Xr = X[:, keep_initial]
        nr = [names[i] for i in keep_initial]
        return Xr, nr, keep_initial

    Xnc = X[:, keep_initial]
    mu  = Xnc.mean(axis=0)
    std_nc = Xnc.std(axis=0)
    std_nc[std_nc < eps] = 1.0
    Z = (Xnc - mu) / std_nc

    denom = max(n - 1, 1)
    C = (Z.T @ Z) / denom
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 0.0)

    picked_rel = []
    dropped_rel = set()
    for i in range(C.shape[0]):
        if i in dropped_rel:
            continue
        picked_rel.append(i)
        too_corr = np.where(np.abs(C[i, i+1:]) > thr)[0]
        for off in too_corr:
            dropped_rel.add(i + 1 + off)

    kept_global = [keep_initial[i] for i in picked_rel]
    kept_global = sorted(set(kept_global))

    Xr = X[:, kept_global]
    nr = [names[i] for i in kept_global]
    return Xr, nr, kept_global

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k_best=10, sfs_k=3, corr_thr=0.90, C=4.0, gamma='scale', cv_sfs=3, random_state=RNG):
        self.k_best = k_best
        self.sfs_k = sfs_k
        self.corr_thr = corr_thr
        self.C = C
        self.gamma = gamma
        self.cv_sfs = cv_sfs
        self.random_state = random_state

        self.selected_indices_ = None
        self.selected_names_ = None
        self._orig_names = None

        self.mi_scores_ = None
        self.mi_scores_vec_ = None
        self.sfs_selected_names_order_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        supplied = fit_params.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
        # If PCA is enabled upstream, the dimensionality after PCA may differ.
        # Use principal-component names when lengths don't match.
        if len(supplied) != X.shape[1]:
            self._orig_names = [f"pc{i+1}" for i in range(X.shape[1])]
        else:
            self._orig_names = supplied

        # (1) MI filter
        k = min(max(1, self.k_best), X.shape[1])
        skb = SelectKBest(mutual_info_classif, k=k)
        X_mi = skb.fit_transform(X, y)
        mi_idx = skb.get_support(indices=True)
        mi_scores_raw = getattr(skb, "scores_", None)

        self.mi_scores_vec_ = np.full(X.shape[1], np.nan, dtype=float)
        if mi_scores_raw is not None:
            self.mi_scores_vec_[:] = mi_scores_raw
        self.mi_scores_ = { self._orig_names[i]: (None if np.isnan(self.mi_scores_vec_[i]) else float(self.mi_scores_vec_[i])) for i in mi_idx }

        names_mi = [self._orig_names[i] for i in mi_idx]

        # (2) corr filter
        X_unc, names_unc, kept_rel = corr_filter(X_mi, names_mi, thr=self.corr_thr)
        kept_global = [mi_idx[i] for i in kept_rel]

        if X_unc.shape[1] == 0 or self.sfs_k == 0:
            self.selected_names_  = names_unc if self.sfs_k == 0 else []
            self.selected_indices_ = kept_global if self.sfs_k == 0 else []
            self.sfs_selected_names_order_ = self.selected_names_.copy()
            return self

        # (3) greedy SFS (RBF SVC)
        skf = StratifiedKFold(n_splits=max(2, self.cv_sfs), shuffle=True, random_state=self.random_state)
        selected_rel, remaining = [], list(range(X_unc.shape[1]))
        self.sfs_selected_names_order_ = []

        def cv_score(cols_rel: List[int]) -> float:
            Xs = X_unc[:, cols_rel]
            est = SVC(kernel='rbf', C=self.C, gamma=self.gamma, class_weight='balanced')
            scores = cross_val_score(est, Xs, y, cv=skf, n_jobs=-1)
            return float(np.mean(scores))

        while len(selected_rel) < min(self.sfs_k, X_unc.shape[1]) and remaining:
            best_j, best_sc = None, -1.0
            for j in remaining:
                sc = cv_score(selected_rel + [j])
                if sc > best_sc:
                    best_sc, best_j = sc, j
            selected_rel.append(best_j)
            remaining.remove(best_j)
            self.sfs_selected_names_order_.append(names_unc[best_j])

        self.selected_names_  = [names_unc[i] for i in selected_rel]
        self.selected_indices_ = [kept_global[i] for i in selected_rel]
        return self

    def transform(self, X: np.ndarray):
        if not self.selected_indices_:
            return X[:, :0]
        return X[:, self.selected_indices_]

# ------------------------------------------------------------------------
# Training utilities (+ SMA pre-classifier learning)
# ------------------------------------------------------------------------
def compute_sma(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    n = max(1, len(x))
    return float((np.abs(x).sum() + np.abs(y).sum() + np.abs(z).sum()) / n)

def learn_sma_threshold_from_windows(windows: List[Dict], fs: float) -> Dict[str, float]:
    """
    Learn an SMA threshold that best separates STATIC_CLASSES vs DYNAMIC_CLASSES.
    Uses preprocessed signals, then computes per-window SMA.
    Threshold is chosen by maximizing accuracy across midpoints of sorted SMA values.
    """
    vals = []
    labs = []
    for w in windows:
        if w["label"] in STATIC_CLASSES + DYNAMIC_CLASSES:
            x_p, y_p, z_p = preprocess_window(w["x"], w["y"], w["z"], fs=fs, denoise=True, gravity_hp_hz=0.5)
            s = compute_sma(x_p, y_p, z_p)
            vals.append(s)
            labs.append(0 if w["label"] in STATIC_CLASSES else 1)  # 0=static, 1=dynamic

    if not vals:
        return {"sma_threshold": 0.0, "sma_static_mean": 0.0, "sma_dynamic_mean": 0.0, "sma_acc_at_threshold": 0.0}

    vals = np.array(vals, dtype=float)
    labs = np.array(labs, dtype=int)

    order = np.argsort(vals)
    v_sorted = vals[order]
    y_sorted = labs[order]

    # candidate thresholds: midpoints between consecutive distinct SMA values
    uniq = np.unique(v_sorted)
    if uniq.size == 1:
        thr = float(uniq[0])
        acc = float(np.mean(y_sorted == (v_sorted >= thr).astype(int)))
        return {
            "sma_threshold": thr,
            "sma_static_mean": float(np.mean(vals[labs == 0])) if np.any(labs == 0) else 0.0,
            "sma_dynamic_mean": float(np.mean(vals[labs == 1])) if np.any(labs == 1) else 0.0,
            "sma_acc_at_threshold": acc
        }

    mids = (uniq[:-1] + uniq[1:]) * 0.5
    best_thr, best_acc = mids[0], -1.0
    for t in mids:
        pred = (vals >= t).astype(int)   # predict dynamic if >= t
        acc = float(np.mean(pred == labs))
        if acc > best_acc:
            best_acc = acc
            best_thr = float(t)

    return {
        "sma_threshold": float(best_thr),
        "sma_static_mean": float(np.mean(vals[labs == 0])) if np.any(labs == 0) else 0.0,
        "sma_dynamic_mean": float(np.mean(vals[labs == 1])) if np.any(labs == 1) else 0.0,
        "sma_acc_at_threshold": float(best_acc)
    }

def build_feature_table_for_category(
    windows: List[Dict],
    fs: float,
    category: str,
    group_by: str = "subject",   # "subject" | "block" | "none"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    X_list, y_list, g_list = [], [], []
    fnames: Optional[List[str]] = None
    target_classes = STATIC_CLASSES if category == "static" else DYNAMIC_CLASSES

    processed = 0
    skipped = 0
    total_target = sum(1 for w in windows if w["label"] in target_classes)
    print(f"[{category}] expecting ~{total_target} windows")

    for idx, w in enumerate(windows):
        if w["label"] not in target_classes:
            continue
        try:
            vals, names = compute_features(w["x"], w["y"], w["z"], fs=fs, mode=category)
        except Exception as e:
            skipped += 1
            print(f"\n[{category}] SKIP window idx={idx} subj={w.get('subject_id')} blk={w.get('block_id')} err={e}")
            continue

        if fnames is None:
            fnames = names
        X_list.append(vals)
        y_list.append(w["label"])

        if group_by == "subject":
            g_list.append(w["subject_id"])
        elif group_by == "block":
            g_list.append(w.get("block_uid", f"{w['subject_id']}#blk{w['block_id']}"))
        elif group_by == "none":
            g_list.append("__all__")
        else:
            raise ValueError(f"Unknown group_by='{group_by}' (use 'subject' | 'block' | 'none')")

        processed += 1
        if processed % 500 == 0:
            print(f"[{category}] windows processed (features extracted): {processed}")

    if not X_list:
        raise RuntimeError(f"No windows for category '{category}'.")
    if skipped:
        print(f"[{category}] skipped {skipped} windows due to errors")

    if group_by == "none":
        groups = np.zeros(len(g_list), dtype=np.int32)
    else:
        groups, _ = pd.factorize(pd.Series(g_list), sort=True)
        groups = groups.astype(np.int32)

    return (np.asarray(X_list, dtype=np.float32),
            np.asarray(y_list, dtype=object),
            groups,
            fnames)

def make_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", 'passthrough'),  # toggled by grid
        ("selector", FeatureSelector()),
        ("svm", SVC(kernel="rbf", probability=False, class_weight="balanced"))
    ])

def nested_cv_train_category(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    default_kbest: int,
    default_sfsk: int,
    label: str,
    fast: bool = True,
) -> Tuple[Pipeline, float, dict, List[str]]:
    def class_counts(arr):
        return pd.Series(arr).value_counts()

    def can_use_sgkf(y_labels, grp):
        return (np.unique(grp).size >= 2)

    def max_splits_for_sgkf(y_labels, grp):
        df = pd.DataFrame({"y": y_labels, "g": grp})
        per_class_group_counts = df.groupby("y")["g"].nunique()
        return int(per_class_group_counts.min())

    def max_splits_for_skf(y_labels):
        return int(class_counts(y_labels).min())

    use_groups_outer = can_use_sgkf(y, groups)
    if use_groups_outer:
        outer_k_cap = max_splits_for_sgkf(y, groups)
        if outer_k_cap >= 2:
            outer_k = max(2, min(5, outer_k_cap))
            outer = StratifiedGroupKFold(n_splits=outer_k, shuffle=True, random_state=RNG)
            outer_iter = outer.split(X, y, groups)
            outer_desc = f"StratifiedGroupKFold (n_splits={outer_k})"
        else:
            use_groups_outer = False

    if not use_groups_outer:
        outer_k_cap = max_splits_for_skf(y)
        if outer_k_cap < 2:
            raise ValueError(f"[{label}] Not enough samples per class to perform 2-fold CV (min class count={outer_k_cap}).")
        outer_k = max(2, min(5, outer_k_cap))
        outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=RNG)
        outer_iter = outer.split(X, y)
        outer_desc = f"StratifiedKFold (within-windows, n_splits={outer_k})"

    print(f"[{label}] Starting nested CV: {outer_desc}, fast={fast}")

    # HYPERPARAMETER TUNING !
    if fast:
        param_grid = {
            # fast pipeline never uses pca
            #"pca": ['passthrough'],
            "pca": ['passthrough', PCA(n_components=0.95, svd_solver='full', random_state=RNG)],
            "selector__k_best":   [max(2, default_kbest)],
            "selector__sfs_k":    [0, min(2, max(0, default_sfsk))],
            "selector__corr_thr": [0.90],
            "selector__C":        [1.0, 4.0],
            "selector__gamma":    ["scale", 0.05],
            "svm__C":             [1.0, 2.0, 4.0],
            "svm__gamma":         ["scale", 0.02, 0.05],
        }
    else:
        param_grid = {
            "pca": ['passthrough', PCA(n_components=0.95, svd_solver='full', random_state=RNG)],
            "selector__k_best":   [max(2, default_kbest - 4), default_kbest, default_kbest + 4],
            "selector__sfs_k":    [0, max(1, default_sfsk - 1), default_sfsk, default_sfsk + 1],
            "selector__corr_thr": [0.85, 0.90, 0.95],
            "selector__C":        [0.5, 1.0, 2.0, 4.0],
            "selector__gamma":    ["scale", 0.01, 0.05, 0.1],
            "svm__C":             [0.5, 1.0, 2.0, 4.0, 8.0],
            "svm__gamma":         ["scale", 0.01, 0.05, 0.1],
        }

    pipe = make_pipeline()

    outer_scores: List[float] = []
    outer_train_scores: List[float] = []
    for fold_id, split in enumerate(outer_iter, start=1):
        tr_idx, te_idx = split
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        g_tr = groups[tr_idx] if use_groups_outer else None

        use_groups_inner = (g_tr is not None) and can_use_sgkf(y_tr, g_tr)
        if use_groups_inner:
            inner_cap = max_splits_for_sgkf(y_tr, g_tr)
            if inner_cap >= 2:
                inner_k = max(2, min(3, inner_cap))
                inner_cv = StratifiedGroupKFold(n_splits=inner_k, shuffle=True, random_state=RNG)
                fit_kwargs = {"groups": g_tr, "selector__feature_names": feature_names}
                inner_desc = f"StratifiedGroupKFold(n_splits={inner_k})"
            else:
                use_groups_inner = False

        if not use_groups_inner:
            inner_cap = max_splits_for_skf(y_tr)
            if inner_cap < 2:
                raise ValueError(f"[{label}] Inner CV cannot stratify: min class count in train fold = {inner_cap}.")
            inner_k = max(2, min(3, inner_cap))
            inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=RNG)
            fit_kwargs = {"selector__feature_names": feature_names}
            inner_desc = f"StratifiedKFold(n_splits={inner_k})"

        gs = GridSearchCV(
            estimator=clone(pipe),
            param_grid=param_grid,
            cv=inner_cv,
            n_jobs=-1,
            scoring="accuracy",
            refit=True,
        )
        gs.fit(X_tr, y_tr, **fit_kwargs)

        best = gs.best_estimator_
        fold_acc = best.score(X_te, y_te)
        fold_train_acc = best.score(X_tr, y_tr)
        outer_scores.append(fold_acc)
        outer_train_scores.append(fold_train_acc)
        print(f"[{label}] Outer fold {fold_id}/{outer_k}: train={len(tr_idx)} test={len(te_idx)} | inner={inner_desc}")
        print(f"[{label}]   fold_acc={fold_acc:.3f} | best={gs.best_params_}")

    mean_outer = float(np.mean(outer_scores)) if outer_scores else 0.0
    std_outer  = float(np.std(outer_scores)) if outer_scores else 0.0
    mean_train = np.mean(outer_train_scores)
    print(f"[{label}] Nested CV done: {mean_outer:.3f} ± {std_outer:.3f}")

    # final refit with larger inner CV
    def can_use_groups_all(y_all, g_all):
        return np.unique(g_all).size >= 2 and max_splits_for_sgkf(y_all, g_all) >= 2

    if can_use_groups_all(y, groups):
        final_k = max(2, min(5, max_splits_for_sgkf(y, groups)))
        final_inner = StratifiedGroupKFold(n_splits=final_k, shuffle=True, random_state=RNG)
        final_fit_kwargs = {"groups": groups, "selector__feature_names": feature_names}
    else:
        final_k = max(2, min(5, max_splits_for_skf(y)))
        final_inner = StratifiedKFold(n_splits=final_k, shuffle=True, random_state=RNG)
        final_fit_kwargs = {"selector__feature_names": feature_names}

    final_gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=final_inner,
        n_jobs=-1,
        scoring="accuracy",
        refit=True,
    )
    final_gs.fit(X, y, **final_fit_kwargs)

    final_best: Pipeline = final_gs.best_estimator_
    best_params = final_gs.best_params_

    try:
        selector: FeatureSelector = final_best.named_steps["selector"]  # type: ignore
        sel_names = selector.selected_names_ if selector.selected_names_ else []
    except Exception:
        sel_names = []

    print(f"[{label}] Final best params: {best_params}")
    print(f"[{label}] Selected features: {sel_names}")
    return final_best, mean_outer, best_params, sel_names

def sanity_print_block_summary(windows: List[Dict], title: str) -> None:
    print(f"[sanity] {title}")
    if not windows:
        print("[sanity] no windows")
        return

    dfw = pd.DataFrame([
        {"subject": w["subject_id"], "label": w["label"], "block": w["block_id"]}
        for w in windows
    ])
    if dfw.empty:
        print("[sanity] empty dataframe"); 
        return

    blk = (
        dfw.drop_duplicates(subset=["subject", "label", "block"])
           .groupby(["subject", "label"]).size()
           .unstack(fill_value=0)
           .sort_index()
    )
    print("[sanity] unique blocks per (subject,label):")
    print(blk)

    cnt = (
        dfw.groupby(["subject", "label"]).size()
           .unstack(fill_value=0)
           .sort_index()
    )
    print("[sanity] window counts per (subject,label):")
    print(cnt)

# ------------------------------------------------------------------------
# ONNX helpers
# ------------------------------------------------------------------------
from sklearn.pipeline import Pipeline as _SkPipeline

def strip_passthrough_steps(pipe: Pipeline) -> Pipeline:
    new_steps = []
    for name, step in pipe.steps:
        if step == 'passthrough':
            continue
        new_steps.append((name, step))
    return _SkPipeline(new_steps)

def export_onnx(pipe: Pipeline, n_features: int, out_path: str):
    if convert_sklearn is None:
        print(f"[WARN] skl2onnx not available; skipping ONNX export for {out_path}")
        return
    try:
        pipe2 = strip_passthrough_steps(pipe)
        initial_types = [("input", FloatTensorType([None, n_features]))]
        onx = convert_sklearn(pipe2, initial_types=initial_types, options={"zipmap": False})
        with open(out_path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"[OK] Wrote {out_path}")
    except Exception as e:
        print(f"[WARN] ONNX export failed for {out_path}: {e}")

def extract_selected_indices(fitted_best: Pipeline) -> List[int]:
    try:
        sel = fitted_best.named_steps.get("selector", None)
        if sel is None:
            return list(range(fitted_best.named_steps["scaler"].n_features_in_))
        if getattr(sel, "selected_indices_", None):
            return sel.selected_indices_
        support_len = fitted_best.named_steps["scaler"].n_features_in_
        return list(range(support_len))
    except Exception:
        return list(range(fitted_best.named_steps["scaler"].n_features_in_))

def make_deployable_pipeline(
    fitted_best: Pipeline,
    X_all: np.ndarray,
    y_all: np.ndarray,
    selected_idx: List[int],
) -> Pipeline:
    pca_step = fitted_best.named_steps.get("pca", "passthrough")
    svm_step = fitted_best.named_steps["svm"]

    steps = [("scaler", StandardScaler())]
    if pca_step != "passthrough":
        steps.append(("pca", clone(pca_step)))
    steps.append(("svm", SVC(
        kernel=svm_step.kernel,
        C=svm_step.C,
        gamma=svm_step.gamma,
        class_weight=svm_step.class_weight,
        probability=False,
        random_state=getattr(svm_step, "random_state", None),
    )))

    deploy = _SkPipeline(steps)
    deploy.fit(X_all[:, selected_idx], y_all)
    return deploy

def export_onnx_deploy(pipe: Pipeline, n_features: int, out_path: str):
    if convert_sklearn is None:
        print(f"[WARN] skl2onnx not available; skipping ONNX export for {out_path}")
        return
    try:
        initial_types = [("input", FloatTensorType([None, n_features]))]
        onx = convert_sklearn(pipe, initial_types=initial_types, options={"zipmap": False})
        with open(out_path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"[OK] Wrote {out_path}")
    except Exception as e:
        print(f"[WARN] ONNX export failed for {out_path}: {e}")

# ------------------------------------------------------------------------
# CLI / main
# ------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train static/dynamic SVMs with nested CV and subject- or block-aware splits.")
    # Per-person mode
    p.add_argument("--person", type=str, default="", help="Enable per-person calibration. Example: --person paul")
    p.add_argument("--data_root", type=str, default="data", help="Root folder containing per-person subfolders (e.g., data/paul/*.csv)")
    # Legacy multi-subject mode
    p.add_argument("--data_dir", type=str, default="data", help="(Generalization mode) Directory of CSVs (auto-load all). Ignored if --person is set.")
    p.add_argument("--csv", type=str, default="build/accel_calib_data.csv", help="Fallback single CSV if data_dir empty (generalization mode).")

    # Windowing / sampling
    p.add_argument("--win", type=int, default=WIN)
    p.add_argument("--hop", type=int, default=HOP)
    p.add_argument("--guard", type=int, default=GUARD)
    p.add_argument("--fs", type=float, default=FS)

    # Output
    p.add_argument("--outdir", type=str, default="", help="Output directory. If --person is set and this is empty, defaults to models/<person>.")
    p.add_argument("--clean_outdir", action="store_true", help="Delete existing files in outdir before writing (safe overwrite).")

    # Training knobs
    p.add_argument("--fast", action="store_true", help="Smaller grids/folds for speed.")
    p.add_argument("--start_activity", type=str, default="", choices=["", "sit", "stand", "walk", "turn"],
                   help="Rotate label order so the FIRST active block is this activity.")
    p.add_argument("--subject_filter", type=str, default="",
                   help="(Generalization mode) Comma-separated subject_id stems to include.")

    # Grouping policy (auto if empty)
    p.add_argument("--group_by", type=str, default="", choices=["", "subject", "block", "none"],
                   help="CV grouping key. Empty = auto: 'block' if --person else 'subject'.")

    return p.parse_args()

@dataclass
class TrainResult:
    static_pipe: Pipeline
    dynamic_pipe: Pipeline
    static_feats: List[str]
    dynamic_feats: List[str]
    static_cv: float
    dynamic_cv: float

if __name__ == "__main__":
    args = parse_args()
    FAST = bool(args.fast)
    FS, WIN, HOP, GUARD = float(args.fs), int(args.win), int(args.hop), int(args.guard)

    # Resolve grouping policy
    if args.group_by:
        group_by = args.group_by
    else:
        group_by = "block" if args.person else "subject"  # AUTO

    # Resolve outdir (stable per-person location if --person)
    outdir = Path(args.outdir) if args.outdir else (Path("models") / args.person if args.person else Path("models"))
    outdir.mkdir(parents=True, exist_ok=True)

    # Optional hard overwrite
    if args.clean_outdir:
        for pth in outdir.glob("*"):
            try:
                if pth.is_file():
                    pth.unlink()
                elif pth.is_dir():
                    import shutil
                    shutil.rmtree(pth)
            except Exception:
                pass

    # Load data
    if args.person:
        person = args.person.strip()
        print(f"[MODE] Per-person calibration for '{person}' (group_by={group_by})")
        windows = collect_windows_from_person_dir(args.data_root, person, start_activity=args.start_activity)
    else:
        data_dir = Path(args.data_dir)
        if data_dir.exists() and any(data_dir.glob("*.csv")):
            print(f"[MODE] Generalization (multi-subject). Loading all CSVs from: {data_dir}")
            windows = collect_windows_from_dir(str(data_dir), start_activity=args.start_activity)
            if args.subject_filter:
                keep = {s.strip() for s in args.subject_filter.split(",") if s.strip()}
                windows = [w for w in windows if w["subject_id"] in keep]
                print(f"[INFO] Subject filter active -> keeping: {sorted(keep)}; windows now: {len(windows)}")
        else:
            csv_path = args.csv if os.path.exists(args.csv) else "accel_calib_data.csv"
            print(f"[MODE] Generalization fallback; using single CSV: {csv_path}")
            windows = build_windows(load_csv(csv_path), subject_id=Path(csv_path).stem, start_activity=args.start_activity)

    sanity_print_block_summary(windows, "blocks & windows per subject/label")
    if not windows:
        raise RuntimeError("No windows constructed; check labels and window params.")

    # Learn SMA pre-classifier threshold from all windows (matches header comment)
    sma_stats = learn_sma_threshold_from_windows(windows, FS)
    print(f"[SMA] threshold={sma_stats['sma_threshold']:.4f} | acc@thr={sma_stats['sma_acc_at_threshold']:.3f} "
          f"| mean(static)={sma_stats['sma_static_mean']:.4f} mean(dynamic)={sma_stats['sma_dynamic_mean']:.4f}")

    # Build per-category feature tables with chosen grouping
    Xs, ys, gs, fn_s = build_feature_table_for_category(windows, FS, category="static",  group_by=group_by)
    Xd, yd, gd, fn_d = build_feature_table_for_category(windows, FS, category="dynamic", group_by=group_by)

    # Train with nested CV
    static_pipe, static_cv, static_params, static_sel = nested_cv_train_category(
        Xs, ys, gs, fn_s, default_kbest=K_BEST_STATIC, default_sfsk=SFS_K_STATIC, label="Static", fast=FAST
    )
    dynamic_pipe, dynamic_cv, dynamic_params, dynamic_sel = nested_cv_train_category(
        Xd, yd, gd, fn_d, default_kbest=K_BEST_DYNAMIC, default_sfsk=SFS_K_DYNAMIC, label="Dynamic", fast=FAST
    )

    print(f"[summary] static windows={len(ys)}, features_per_window={len(fn_s)}")
    print(f"[summary] dynamic windows={len(yd)}, features_per_window={len(fn_d)}")
    print(f"[summary] subjects static={np.unique(gs).size}, dynamic={np.unique(gd).size}")
    print(f"[summary] static y counts: {pd.Series(ys).value_counts().to_dict()}")
    print(f"[summary] dynamic y counts: {pd.Series(yd).value_counts().to_dict()}")

    # ONNX (best effort) — build deployable (no custom selector) and export
    static_sel_idx = extract_selected_indices(static_pipe)
    dynamic_sel_idx = extract_selected_indices(dynamic_pipe)
    static_sel_idx_py  = [int(i) for i in static_sel_idx]
    dynamic_sel_idx_py = [int(i) for i in dynamic_sel_idx]

    static_deploy  = make_deployable_pipeline(static_pipe,  Xs, ys, static_sel_idx)
    dynamic_deploy = make_deployable_pipeline(dynamic_pipe, Xd, yd, dynamic_sel_idx)

    export_onnx_deploy(static_deploy,  n_features=len(static_sel_idx),  out_path=str(outdir / "static_svm.onnx"))
    export_onnx_deploy(dynamic_deploy, n_features=len(dynamic_sel_idx), out_path=str(outdir / "dynamic_svm.onnx"))

    # Reports
    generate_report(
    Xs, ys, gs, fn_s, static_pipe, label="Static", outdir=str(outdir),
    outer_n_splits=5, best_params=static_params
    )
    generate_report(
        Xd, yd, gd, fn_d, dynamic_pipe, label="Dynamic", outdir=str(outdir),
        outer_n_splits=5, best_params=dynamic_params
    )

    # Meta (json for c++)

    # Did each fitted pipeline actually use PCA? (passthrough = no pca)
    static_pca_step  = static_pipe.named_steps.get("pca", "passthrough")
    dynamic_pca_step = dynamic_pipe.named_steps.get("pca", "passthrough")
    static_pca_used  = (static_pca_step != "passthrough")  and hasattr(static_pca_step,  "components_")
    dynamic_pca_used = (dynamic_pca_step != "passthrough") and hasattr(dynamic_pca_step, "components_")

    meta = {
        # run context
        "person": args.person if args.person else None,
        "mode": "per_person" if args.person else "generalization",
        "group_by": group_by,

        # windowing (C++ needs this to build windows the same way)
        "windowing": {
            "fs_hz": float(FS),
            "win": int(WIN),
            "hop": int(HOP),
            "guard": int(GUARD)
        },

        # router (SMA pre-classifier) — whatever learner returned
        "router": {
            "type": "SMA",
            "stats": sma_stats  # contains threshold + any diagnostics computed
        },

        # feature registry: exact order Python generates (C++ should compute in this order, then slice)
        "feature_order": {
            "static": fn_s if fn_s else [],
            "dynamic": fn_d if fn_d else []
        },

        # models: info C++ needs to slice features and map outputs
        "static": {
            "classes": STATIC_CLASSES,
            "cv_acc_outer_mean": float(static_cv),
            "pca": {"used": bool(static_pca_used)},
            "selected_indices": [int(i) for i in static_sel_idx_py], # from c++ usable onxx model table
            "selected_features": ([fn_s[i] for i in static_sel_idx_py] if fn_s else [])
        },
        "dynamic": {
            "classes": DYNAMIC_CLASSES,
            "cv_acc_outer_mean": float(dynamic_cv),
            "pca": {"used": bool(dynamic_pca_used)},
            "selected_indices": [int(i) for i in dynamic_sel_idx_py],
            "selected_features": ([fn_d[i] for i in dynamic_sel_idx_py] if fn_d else [])
        },

        # integration note
        "note": (
            "C++: compute features in 'feature_order' order, slice by 'selected_indices' "
            "for each branch, then feed to the ONNX model. Scaler/PCA/SVM are embedded in ONNX."
        )
    }

    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved {outdir/'meta.json'}")
    print("[DONE] Static outer acc (mean): {:.3f}".format(static_cv))
    print("[DONE] Dynamic outer acc (mean): {:.3f}".format(dynamic_cv))

