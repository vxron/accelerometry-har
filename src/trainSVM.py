#!/usr/bin/env python3
"""
Train Activity SVMs (static vs dynamic) with feature selection and ONNX export.

This script:
(1) Loads a CSV (columns: tick,x,y,z,active)
(2) Detects contiguous 'active == 1' blocks and assigns labels in fixed order:
    sit -> stand -> walk -> turn_cw -> (repeat)
(3) Builds sliding windows (win_len, hop) ONLY if fully inside an activity block,
    with guard margins trimmed from both block edges to avoid toggle jitter
(4) Extracts features (time + frequency); standardizes inside the model pipeline
(5) Learns an intensity-based pre-classifier (SMA) to route:
        static {sit, stand} vs dynamic {walk, turn_cw}
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

#***** imports *****
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
from sklearn.model_selection import GroupKFold, StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# ONNX export (optional)
try:
    import onnx  # noqa: F401
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except Exception:
    convert_sklearn = None

#***** config (defaults) *****
STATIC_CLASSES = ["sit", "stand"]
DYNAMIC_CLASSES = ["walk", "turn_cw"]
ACTIVITY_ORDER  = ["sit", "stand", "walk", "turn_cw"]

FS    = 119.0   # Hz (CLI overridable)
WIN   = 200     # samples (CLI)
HOP   = 100     # samples (CLI)
GUARD = 50      # samples (CLI)

AR_ORDER = 4
RNG      = 13

# selector default knobs (tuned by inner CV too)
K_BEST_STATIC  = 12
K_BEST_DYNAMIC = 18
SFS_K_STATIC   = 2
SFS_K_DYNAMIC  = 3

# PCA grid values used in inner CV
PCA_GRID = ['passthrough', PCA(n_components=0.95, svd_solver='full', random_state=RNG)]

#***** reproducibility *****
np.random.seed(RNG)

#===============================================================================
#                               DATA & LABELING
#===============================================================================

#***** load_csv *****
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["x", "y", "z", "active"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {path}")
    return df

#***** find_active_blocks *****
def find_active_blocks(active: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return half-open runs [start, end) where active == 1.
    Example: [0,1,1,0,1] -> [(1,3), (4,5)]
    """
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

#***** windows_from_block *****
def windows_from_block(start: int, end: int, win: int, hop: int, guard: int) -> List[Tuple[int, int]]:
    s, e = start + guard, end - guard
    spans, i = [], s
    while i + win <= e:
        spans.append((i, i + win))
        i += hop
    return spans

#***** assign_labels_to_blocks *****
def assign_labels_to_blocks(n: int, one_blocks: List[Tuple[int, int]]) -> np.ndarray:
    labels = np.array([""] * n, dtype=object)
    for k, (a, b) in enumerate(one_blocks):
        cls = ACTIVITY_ORDER[k % len(ACTIVITY_ORDER)]
        labels[a:b] = cls
    return labels

#***** build_windows (adds block_id for GroupKFold) *****
def build_windows(df: pd.DataFrame) -> List[Dict]:
    """
    Returns windows with: s,e,x,y,z,label,block_id
    """
    x = df["x"].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    z = df["z"].to_numpy(dtype=np.float32)
    active = df["active"].astype(int).to_numpy()

    blocks = find_active_blocks(active)
    labels_series = assign_labels_to_blocks(len(active), blocks)

    windows = []
    for block_id, (a, b) in enumerate(blocks):
        lab = labels_series[a] if a < len(labels_series) else ""
        if lab == "":
            continue
        for s, e in windows_from_block(a, b, WIN, HOP, GUARD):
            if (labels_series[s:e] == lab).all():
                windows.append({
                    "s": s, "e": e, "x": x[s:e], "y": y[s:e], "z": z[s:e],
                    "label": lab, "block_id": block_id
                })
    return windows

#===============================================================================
#                         PREPROCESSING (REUSABLE)
#===============================================================================

#***** robust_mad_clip *****
def robust_mad_clip(arr: np.ndarray, k: float = 6.0) -> np.ndarray:
    """
    Robust outlier guard using Median Absolute Deviation (MAD).
    Clips arr to [median - k*MAD, median + k*MAD].
    """
    if arr.size == 0:
        return arr
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    # avoid zero MAD collapse
    cutoff = k * (mad if mad > 1e-9 else np.std(arr) + 1e-9)
    lo, hi = med - cutoff, med + cutoff
    return np.clip(arr, lo, hi)

#***** highpass_butter *****
def highpass_butter(x: np.ndarray, fs: float, fc: float = 0.5, order: int = 2) -> np.ndarray:
    """
    Zero-phase high-pass Butterworth (filtfilt) to remove gravity (~DC/very-low freq).
    """
    if x.size < order * 3:
        return x - np.mean(x)  # fallback to mean-removal if too short
    wn = fc / (0.5 * fs)
    wn = min(max(wn, 1e-5), 0.999)  # keep in (0,1)
    b, a = butter(order, wn, btype='highpass')
    return filtfilt(b, a, x, method="gust")

#***** preprocess_window (REUSABLE) *****
def preprocess_window(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                      fs: float,
                      denoise: bool = True,
                      gravity_hp_hz: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Steps (per axis):
      1) robust MAD clip (outlier guard)
      2) optional median denoise (kernel 5)
      3) high-pass to remove gravity (default 0.5 Hz)
    Returns cleaned (x,y,z) arrays, same length as input.
    """
    # 1) outlier guard
    x = robust_mad_clip(np.asarray(x, dtype=np.float64))
    y = robust_mad_clip(np.asarray(y, dtype=np.float64))
    z = robust_mad_clip(np.asarray(z, dtype=np.float64))

    # 2) median filter (light denoise)
    if denoise:
        k = 5 if len(x) >= 5 else 3
        if k % 2 == 0: k += 1
        x = medfilt(x, kernel_size=k)
        y = medfilt(y, kernel_size=k)
        z = medfilt(z, kernel_size=k)

    # 3) gravity removal (HPF)
    x = highpass_butter(x, fs, fc=gravity_hp_hz)
    y = highpass_butter(y, fs, fc=gravity_hp_hz)
    z = highpass_butter(z, fs, fc=gravity_hp_hz)

    return x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)

#===============================================================================
#                            FEATURE EXTRACTION (REUSABLE)
#===============================================================================

#***** small helpers *****
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
    nperseg = int(min(sig.size, 256))
    if nperseg < 16:
        return 0.0
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg)
    m = (f >= f_lo) & (f < f_hi)
    return float(np.trapz(Pxx[m], f[m])) if np.any(m) else 0.0

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

#***** feature_static_core *****
def feature_static_core(x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float) -> Tuple[List[float], List[str]]:
    names, vals = [], []
    mag = mag_from_xyz(x, y, z)

    # moments on |a|
    vals += [float(skew(mag, bias=False)), float(kurtosis(mag, fisher=True, bias=False))]
    names += ["skewness", "kurtosis"]

    # AR on |a|
    a, res_var = ar_fit_residuals(mag, AR_ORDER)
    for i, coeff in enumerate(a, 1):
        vals.append(float(coeff)); names.append(f"ar_a{i}")
    vals.append(res_var); names.append("ar_res_var")
    vals.append(res_var / (safe_var(mag) + 1e-12)); names.append("ar_white_noise_rate")

    # per-axis stats
    for arr, nm in [(x,"x"),(y,"y"),(z,"z")]:
        vals += [float(np.mean(arr)), float(np.std(arr))]
        names += [f"mean_{nm}", f"std_{nm}"]

    # correlations
    vals += [axis_corr(x,y), axis_corr(y,z), axis_corr(z,x)]
    names += ["corr_xy","corr_yz","corr_zx"]
    return vals, names

#***** feature_dynamic_core *****
def feature_dynamic_core(x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float) -> Tuple[List[float], List[str]]:
    names, vals = [], []
    mag = mag_from_xyz(x, y, z)

    nperseg = int(min(mag.size, 256))
    if nperseg < 16:
        return [0.0]*9, ["psd_median","psd_mean","psd_min","fft_mean","fft_var","fft_std",
                         "bp_0p5_3","bp_3_10","peak_freq"]

    f, Pxx = welch(mag, fs=fs, nperseg=nperseg)
    vals += [float(np.median(Pxx)), float(np.mean(Pxx)), float(np.min(Pxx))]
    names += ["psd_median","psd_mean","psd_min"]

    spec = np.abs(np.fft.rfft(mag - mag.mean()))
    vals += [float(np.mean(spec)), float(np.var(spec)), float(np.std(spec))]
    names += ["fft_mean","fft_var","fft_std"]

    vals += [bandpower_welch(mag, fs, 0.5, 3.0), bandpower_welch(mag, fs, 3.0, 10.0)]
    names += ["bp_0p5_3","bp_3_10"]

    vals += [peak_freq(mag, fs)]
    names += ["peak_freq"]
    return vals, names

#***** compute_features (REUSABLE entrypoint) *****
def compute_features(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     fs: float,
                     mode: str) -> Tuple[List[float], List[str]]:
    """
    Normal operation entrypoint:
      raw window (x,y,z) -> preprocess_window -> features for given mode.
      mode ∈ {"static","dynamic"}
    """
    x_p, y_p, z_p = preprocess_window(x, y, z, fs=fs, denoise=True, gravity_hp_hz=0.5)
    if mode == "static":
        return feature_static_core(x_p, y_p, z_p, fs)
    elif mode == "dynamic":
        return feature_dynamic_core(x_p, y_p, z_p, fs)
    else:
        raise ValueError("mode must be 'static' or 'dynamic'")

#===============================================================================
#                    IN-PIPELINE FEATURE SELECTION (Transformer)
#===============================================================================

#***** corr_filter *****
def corr_filter(X: np.ndarray, names: List[str], thr: float = 0.90) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Greedy correlation filter on columns: keep first, drop later with |r|>thr.
    Returns (X_reduced, names_reduced, kept_col_indices)
    """
    if X.shape[1] <= 1:
        return X, names, list(range(X.shape[1]))
    C = np.corrcoef(X, rowvar=False)
    keep, dropped = [], set()
    for i in range(C.shape[0]):
        if i in dropped: continue
        keep.append(i)
        for j in range(i+1, C.shape[0]):
            if abs(C[i, j]) > thr:
                dropped.add(j)
    return X[:, keep], [names[i] for i in keep], keep

#***** FeatureSelector (MI -> Corr -> SFS) *****
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    In-fold feature selection:
      1) SelectKBest(mutual_info_classif, k=k_best)
      2) Correlation filter (|r|>corr_thr)
      3) Sequential Forward Selection (SFS) using an RBF SVC with provided C,gamma
         evaluated by StratifiedKFold(cv_sfs)

    Parameters (grid-tunable):
      k_best, sfs_k, corr_thr, C, gamma

    Attributes after fit:
      selected_indices_ (relative to input X columns)
      selected_names_
    """
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

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        # original feature names (if provided via fit_params)
        self._orig_names = fit_params.get("feature_names", [f"f{i}" for i in range(X.shape[1])])

        # 1) MI filter
        k = min(max(1, self.k_best), X.shape[1])
        skb = SelectKBest(mutual_info_classif, k=k)
        X_mi = skb.fit_transform(X, y)
        mi_idx = skb.get_support(indices=True)
        names_mi = [self._orig_names[i] for i in mi_idx]

        # 2) correlation filter
        X_unc, names_unc, kept_rel = corr_filter(X_mi, names_mi, thr=self.corr_thr)
        kept_global = [mi_idx[i] for i in kept_rel]  # to original columns

        # if not enough columns survive, keep what we have
        if X_unc.shape[1] == 0:
            self.selected_indices_ = []
            self.selected_names_ = []
            return self

        # 3) SFS (small CV inside)
        skf = StratifiedKFold(n_splits=self.cv_sfs, shuffle=True, random_state=self.random_state)
        selected_rel: List[int] = []
        remaining = list(range(X_unc.shape[1]))
        best_cols_global: List[int] = []

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

        self.selected_names_  = [names_unc[i] for i in selected_rel]
        best_cols_global      = [kept_global[i] for i in selected_rel]
        self.selected_indices_ = best_cols_global
        return self

    def transform(self, X: np.ndarray):
        if not self.selected_indices_:
            return X[:, :0]  # empty selection (rare)
        return X[:, self.selected_indices_]

#===============================================================================
#                            TRAINING UTILITIES
#===============================================================================

#***** compute_sma (kept for possible routing later) *****
def compute_sma(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    n = max(1, len(x))
    return float((np.abs(x).sum() + np.abs(y).sum() + np.abs(z).sum()) / n)

#***** build_feature_table_for_category *****
def build_feature_table_for_category(windows: List[Dict], fs: float, category: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    category: "static" or "dynamic"
    Returns: X, y, groups(block_id), feature_names
    """
    X_list, y_list, g_list = [], [], []
    fnames: Optional[List[str]] = None

    target_classes = STATIC_CLASSES if category == "static" else DYNAMIC_CLASSES

    for w in windows:
        if w["label"] not in target_classes:
            continue
        vals, names = compute_features(w["x"], w["y"], w["z"], fs=fs, mode=category)
        if fnames is None:
            fnames = names
        X_list.append(vals)
        y_list.append(w["label"])
        g_list.append(w["block_id"])

    if not X_list:
        raise RuntimeError(f"No windows for category '{category}'.")
    return (np.asarray(X_list, dtype=np.float32),
            np.asarray(y_list, dtype=object),
            np.asarray(g_list, dtype=np.int32),
            fnames)

#***** make_pipeline *****
def make_pipeline() -> Pipeline:
    """
    StandardScaler -> optional PCA (toggled by grid) -> FeatureSelector -> SVC
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", 'passthrough'),  # toggled by grid
        ("selector", FeatureSelector()),
        ("svm", SVC(kernel="rbf", probability=False, class_weight="balanced"))
    ])
    return pipe

#***** nested_cv_train_category *****
def nested_cv_train_category(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                             feature_names: List[str],
                             default_kbest: int, default_sfsk: int,
                             label: str) -> Tuple[Pipeline, float, dict, List[str]]:
    """
    Performs nested CV:
      outer = GroupKFold; inner = GridSearchCV
    Returns fitted best_estimator (retrained on all data via inner CV),
            mean outer score, best_params (from full-data inner CV),
            selected feature names (from final fit).
    """
    outer = GroupKFold(n_splits=5)

    # inner grid (kept modest)
    param_grid = {
        "pca": PCA_GRID,
        "selector__k_best":   [max(2, default_kbest - 4), default_kbest, default_kbest + 4],
        "selector__sfs_k":    [max(1, default_sfsk - 1), default_sfsk, default_sfsk + 1],
        "selector__corr_thr": [0.85, 0.90, 0.95],
        "selector__C":        [0.5, 1.0, 2.0, 4.0],
        "selector__gamma":    ["scale", 0.01, 0.05, 0.1],
        "svm__C":             [0.5, 1.0, 2.0, 4.0, 8.0],
        "svm__gamma":         ["scale", 0.01, 0.05, 0.1],
    }

    pipe = make_pipeline()

    # outer loop score collection
    outer_scores = []
    for train_idx, test_idx in outer.split(X, y, groups):
        X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]
        X_te, y_te       = X[test_idx], y[test_idx]

        inner = GroupKFold(n_splits=3)
        gs = GridSearchCV(
            estimator=clone(pipe),
            param_grid=param_grid,
            cv=inner,
            n_jobs=-1,
            scoring="accuracy",
            refit=True  # keep best on the inner train fold
        )
        # pass feature_names to selector.fit via fit_params
        gs.fit(X_tr, y_tr, groups=g_tr, selector__feature_names=feature_names)
        best = gs.best_estimator_
        fold_score = best.score(X_te, y_te)
        outer_scores.append(fold_score)
        print(f"[{label}] Outer fold acc: {fold_score:.3f} | best params: {gs.best_params_}")

    mean_outer = float(np.mean(outer_scores))
    std_outer  = float(np.std(outer_scores))
    print(f"[{label}] Nested CV: {mean_outer:.3f} ± {std_outer:.3f}")

    # final model: run inner CV on full data to select params, then refit on all data
    final_inner = GroupKFold(n_splits=5)
    final_gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=final_inner,
        n_jobs=-1,
        scoring="accuracy",
        refit=True
    )
    final_gs.fit(X, y, groups=groups, selector__feature_names=feature_names)
    final_best = final_gs.best_estimator_
    best_params = final_gs.best_params_

    # pull selected feature names from selector (after refit)
    selector: FeatureSelector = final_best.named_steps["selector"]
    sel_names = selector.selected_names_ if selector.selected_names_ is not None else []

    print(f"[{label}] Final best params: {best_params}")
    print(f"[{label}] Selected features: {sel_names}")
    return final_best, mean_outer, best_params, sel_names

#===============================================================================
#                                   ONNX
#===============================================================================

def export_onnx(pipe: Pipeline, n_features: int, out_path: str):
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

#===============================================================================
#                                    MAIN
#===============================================================================

#***** CLI *****
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train static/dynamic SVMs with nested CV and block-aware splits.")
    p.add_argument("--csv", type=str, default="build/accel_calib_data.csv")
    p.add_argument("--win", type=int, default=WIN)
    p.add_argument("--hop", type=int, default=HOP)
    p.add_argument("--guard", type=int, default=GUARD)
    p.add_argument("--fs", type=float, default=FS)
    p.add_argument("--outdir", type=str, default="models")
    return p.parse_args()

#***** dataclass *****
@dataclass
class TrainResult:
    static_pipe: Pipeline
    dynamic_pipe: Pipeline
    static_feats: List[str]
    dynamic_feats: List[str]
    static_cv: float
    dynamic_cv: float

#***** main *****
if __name__ == "__main__":
    args = parse_args()
    FS, WIN, HOP, GUARD = float(args.fs), int(args.win), int(args.hop), int(args.guard)

    csv_path = args.csv if os.path.exists(args.csv) else "accel_calib_data.csv"
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] FS={FS}, WIN={WIN}, HOP={HOP}, GUARD={GUARD}")

    df = load_csv(csv_path)
    windows = build_windows(df)
    if not windows:
        raise RuntimeError("No windows constructed; check labels and window params.")

    # Build category tables (features computed via reusable preprocessing+features)
    Xs, ys, gs, fn_s = build_feature_table_for_category(windows, FS, category="static")
    Xd, yd, gd, fn_d = build_feature_table_for_category(windows, FS, category="dynamic")

    # Train with nested CV (block-aware)
    static_pipe, static_cv, static_params, static_sel = nested_cv_train_category(
        Xs, ys, gs, fn_s, default_kbest=K_BEST_STATIC, default_sfsk=SFS_K_STATIC, label="Static"
    )
    dynamic_pipe, dynamic_cv, dynamic_params, dynamic_sel = nested_cv_train_category(
        Xd, yd, gd, fn_d, default_kbest=K_BEST_DYNAMIC, default_sfsk=SFS_K_DYNAMIC, label="Dynamic"
    )

    # Save artifacts
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ONNX (best effort)
    export_onnx(static_pipe,  n_features=Xs.shape[1], out_path=str(outdir / "static_svm.onnx"))
    export_onnx(dynamic_pipe, n_features=Xd.shape[1], out_path=str(outdir / "dynamic_svm.onnx"))

    meta = {
        "fs_hz": FS, "win": WIN, "hop": HOP, "guard": GUARD, "ar_order": AR_ORDER,
        "static": {
            "cv_acc_outer_mean": static_cv,
            "best_params": static_params,
            "selected_features": static_sel,
            "classes": STATIC_CLASSES
        },
        "dynamic": {
            "cv_acc_outer_mean": dynamic_cv,
            "best_params": dynamic_params,
            "selected_features": dynamic_sel,
            "classes": DYNAMIC_CLASSES
        },
        "feature_contract_note": "FeatureSelector runs in-pipeline; selected_features list from final refit."
    }
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved {outdir/'meta.json'}")

    print("[DONE] Static outer acc (mean): {:.3f}".format(static_cv))
    print("[DONE] Dynamic outer acc (mean): {:.3f}".format(dynamic_cv))
