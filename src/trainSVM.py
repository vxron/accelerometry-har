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
STATIC_CLASSES = ["stand","sit"]
DYNAMIC_CLASSES = ["walk", "turn"]
ACTIVITY_ORDER  = ["stand", "walk", "sit", "turn"]

FS    = 119.0   # Hz (CLI overridable)
WIN   = 200     # samples (CLI)
HOP   = 100     # samples (CLI)
GUARD = 50      # samples (CLI)

AR_ORDER = 4
RNG      = 13

# feature-selection defaults (inner CV can move these)
K_BEST_STATIC  = 12
K_BEST_DYNAMIC = 18
SFS_K_STATIC   = 2
SFS_K_DYNAMIC  = 3

PCA_GRID = ['passthrough', PCA(n_components=0.95, svd_solver='full', random_state=RNG)]

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
def assign_labels_to_blocks(n: int, one_blocks: List[Tuple[int, int]], order: List[str]) -> np.ndarray:
    labels = np.array([""] * n, dtype=object)
    for k, (a, b) in enumerate(one_blocks):
        cls = order[k % len(order)]
        labels[a:b] = cls
    return labels

#***** rotate_activity_order *****
def rotate_activity_order(start_activity: str, base_order: List[str]) -> List[str]:
    """Rotate ACTIVITY_ORDER so it begins with start_activity (if provided/valid)."""
    if not start_activity or start_activity not in base_order:
        return base_order
    i = base_order.index(start_activity)
    return base_order[i:] + base_order[:i]

#***** build_windows (now carries subject_id and block_uid) *****
def build_windows(df: pd.DataFrame, subject_id: str, start_activity: str = "") -> List[Dict]:
    """
    Returns windows with: s,e,x,y,z,label,subject_id,block_id,block_uid
    block_uid is unique per (subject_id, block_id) -> safe for grouping if needed.
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
                    # stable unique key string; factorize later if you want int ids
                    "block_uid": f"{subject_id}#blk{block_id}"
                })
    return windows

#***** collect_windows_from_dir (AUTO-LOAD ALL CSVs) *****
#***** collect_windows_from_dir (accepts start_activity) *****
def collect_windows_from_dir(data_dir: str, start_activity: str = "") -> List[Dict]:
    """
    Walks data_dir, loads every *.csv, builds windows, tags subject_id from filename.
    start_activity: optional first label to rotate order to (e.g., 'walk').
    """
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


#===============================================================================
#                         PREPROCESSING (REUSABLE)
#===============================================================================

#***** robust_mad_clip *****
def robust_mad_clip(arr: np.ndarray, k: float = 6.0) -> np.ndarray:
    if arr.size == 0:
        return arr
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    cutoff = k * (mad if mad > 1e-9 else np.std(arr) + 1e-9)
    lo, hi = med - cutoff, med + cutoff
    return np.clip(arr, lo, hi)

#***** highpass_butter (safe) *****
def highpass_butter(x: np.ndarray, fs: float, fc: float = 0.5, order: int = 2) -> np.ndarray:
    """Zero-phase HPF with robust fallbacks if filtfilt/butter complain."""
    s = np.asarray(x, dtype=np.float64)
    nmin = order * 3
    if s.size < max(8, nmin):
        return s - np.mean(s)  # fallback if too short
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    wn = fc / (0.5 * fs)
    wn = float(min(max(wn, 1e-5), 0.999))
    try:
        b, a = butter(order, wn, btype='highpass')
        return filtfilt(b, a, s, method="gust")
    except Exception:
        # fallback #1: mean removal
        s0 = s - np.mean(s)
        # fallback #2: very light difference filter
        try:
            d = np.diff(s0, prepend=s0[0])
            return d - np.mean(d)
        except Exception:
            return s0

#***** preprocess_window (REUSABLE) *****
def preprocess_window(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                      fs: float,
                      denoise: bool = True,
                      gravity_hp_hz: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 1) outlier guard
    x = robust_mad_clip(np.asarray(x, dtype=np.float64))
    y = robust_mad_clip(np.asarray(y, dtype=np.float64))
    z = robust_mad_clip(np.asarray(z, dtype=np.float64))
    # 2) median denoise
    if denoise:
        k = 5 if len(x) >= 5 else 3
        if k % 2 == 0: k += 1
        x = medfilt(x, kernel_size=k)
        y = medfilt(y, kernel_size=k)
        z = medfilt(z, kernel_size=k)
    # 3) gravity removal
    x = highpass_butter(x, fs, fc=gravity_hp_hz)
    y = highpass_butter(y, fs, fc=gravity_hp_hz)
    z = highpass_butter(z, fs, fc=gravity_hp_hz)
    return x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)

#===============================================================================
#                            FEATURE EXTRACTION (REUSABLE)
#===============================================================================

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

#***** safe_welch (robust wrapper) *****
def safe_welch(sig: np.ndarray, fs: float, nperseg: int = 128):
    """Welch PSD that never explodes: sanitizes input, bounds nperseg, and falls back gracefully."""
    s = np.asarray(sig, dtype=np.float64)
    if s.size < 8:
        return np.array([0.0]), np.array([0.0])
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(np.isfinite(s)):
        return np.array([0.0]), np.array([0.0])
    if np.allclose(s, s[0], atol=1e-12):  # essentially constant
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

#***** feature cores *****
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
    """
    Dynamic features computed with a SINGLE Welch call reused for:
      - PSD median/mean/min
      - Bandpower in [0.5,3) and [3,10)
      - Peak frequency
      - Plus simple FFT magnitude stats
    This avoids repeated Welch calls that can hang on pathological windows.
    """
    names, vals = [], []
    mag = mag_from_xyz(x, y, z)

    # guard small/flat windows quickly
    if mag.size < 16 or np.allclose(mag, mag[0], atol=1e-12):
        return [0.0]*9, [
            "psd_median","psd_mean","psd_min",
            "fft_mean","fft_var","fft_std",
            "bp_0p5_3","bp_3_10","peak_freq"
        ]

    # one safe Welch only
    nperseg = int(min(mag.size, 128))
    f, Pxx = safe_welch(mag, fs=fs, nperseg=nperseg)

    # if safe_welch had to fall back
    if f.size < 2 or Pxx.size < 2:
        spec = np.abs(np.fft.rfft(mag - mag.mean()))
        return [
            0.0, 0.0, 0.0,                          # psd*
            float(np.mean(spec)), float(np.var(spec)), float(np.std(spec)),  # fft*
            0.0, 0.0, 0.0                            # bp*, peak_freq
        ], [
            "psd_median","psd_mean","psd_min",
            "fft_mean","fft_var","fft_std",
            "bp_0p5_3","bp_3_10","peak_freq"
        ]

    # PSD stats
    psd_median = float(np.median(Pxx))
    psd_mean   = float(np.mean(Pxx))
    psd_min    = float(np.min(Pxx))
    vals += [psd_median, psd_mean, psd_min]
    names += ["psd_median","psd_mean","psd_min"]

    # FFT (cheap) stats
    spec = np.abs(np.fft.rfft(mag - mag.mean()))
    vals += [float(np.mean(spec)), float(np.var(spec)), float(np.std(spec))]
    names += ["fft_mean","fft_var","fft_std"]

    # Bandpowers reusing the SAME f,Pxx
    m1 = (f >= 0.5) & (f < 3.0)
    m2 = (f >= 3.0) & (f < 10.0)
    bp1 = float(np.trapezoid(Pxx[m1], f[m1])) if np.any(m1) else 0.0
    bp2 = float(np.trapezoid(Pxx[m2], f[m2])) if np.any(m2) else 0.0
    vals += [bp1, bp2]
    names += ["bp_0p5_3","bp_3_10"]

    # Peak frequency from same Welch
    peak_f = float(f[int(np.argmax(Pxx))]) if Pxx.size else 0.0
    vals += [peak_f]
    names += ["peak_freq"]

    return vals, names


#***** compute_features (REUSABLE entrypoint) *****
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

#===============================================================================
#                    IN-PIPELINE FEATURE SELECTION (Transformer)
#===============================================================================

#***** corr_filter *****
def corr_filter(X: np.ndarray, names: List[str], thr: float = 0.90) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Remove highly correlated features (|r| > thr) with zero-variance columns
    dropped up-front. Computes correlation without numpy.corrcoef to avoid
    RuntimeWarning: invalid value encountered in divide.
    Returns:
        X_reduced, reduced_names, kept_indices_from_input
    """
    n, d = X.shape
    if d <= 1:
        return X, names, list(range(d))

    # sanitize
    X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    # drop zero-variance columns
    std = X.std(axis=0)
    eps = 1e-15
    nonconst_mask = std > eps
    keep_initial = np.where(nonconst_mask)[0].tolist()  # candidates for corr filtering

    if len(keep_initial) <= 1:
        # either all const or only one non-const -> nothing to correlate
        Xr = X[:, keep_initial]
        nr = [names[i] for i in keep_initial]
        return Xr, nr, keep_initial

    # standardize non-constant block (safe)
    Xnc = X[:, keep_initial]
    mu  = Xnc.mean(axis=0)
    std_nc = Xnc.std(axis=0)
    std_nc[std_nc < eps] = 1.0  # just in case
    Z = (Xnc - mu) / std_nc

    # correlation matrix via normalized dot-products
    # r_ij = (z_i^T z_j) / (n-1) with z standardized
    denom = max(n - 1, 1)
    C = (Z.T @ Z) / denom
    # numerical clip
    C = np.clip(C, -1.0, 1.0)
    # ignore self-correlation
    np.fill_diagonal(C, 0.0)

    # greedy keep/drop on non-constant set
    picked_rel = []
    dropped_rel = set()
    for i in range(C.shape[0]):
        if i in dropped_rel:
            continue
        picked_rel.append(i)
        # drop later features highly correlated with the picked one
        too_corr = np.where(np.abs(C[i, i+1:]) > thr)[0]
        for off in too_corr:
            dropped_rel.add(i + 1 + off)

    # map relative picks back to global indices
    kept_global = [keep_initial[i] for i in picked_rel]
    kept_global = sorted(set(kept_global))

    Xr = X[:, kept_global]
    nr = [names[i] for i in kept_global]
    return Xr, nr, kept_global


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    1) SelectKBest(mutual_info) -> top-k
    2) Correlation filter (drop > corr_thr with earlier keeps)
    3) Optional SFS wrapper (sfs_k > 0)
    Guarantees: returns at least 1 feature (fallback to best MI if needed).
    """
    def __init__(self, k_best=10, sfs_k=3, corr_thr=0.90, C=4.0, gamma='scale', cv_sfs=3, random_state=RNG):
        self.k_best = int(k_best)
        self.sfs_k = int(sfs_k)
        self.corr_thr = float(corr_thr)
        self.C = C
        self.gamma = gamma
        self.cv_sfs = int(cv_sfs)
        self.random_state = random_state
        self.selected_indices_ = None
        self.selected_names_ = None
        self._orig_names = None
        self._fallback_used_ = False

    def _ensure_nonempty(self, X: np.ndarray, mi_idx: np.ndarray, names_all: List[str]):
        if mi_idx.size == 0:
            idx = [0]
            names = [names_all[0] if names_all else "f0"]
        else:
            idx = [int(mi_idx[0])]
            names = [names_all[idx[0]] if names_all else f"f{idx[0]}"]
        self._fallback_used_ = True
        return idx, names

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        n_samples, n_features = X.shape
        self._orig_names = fit_params.get("feature_names", [f"f{i}" for i in range(n_features)])
        self._fallback_used_ = False

        # --- 1) MI (k >= 1)
        k = max(1, min(self.k_best, n_features))
        skb = SelectKBest(mutual_info_classif, k=k)
        try:
            X_mi = skb.fit_transform(X, y)
            mi_idx = skb.get_support(indices=True)
            scores = np.asarray(skb.scores_) if getattr(skb, "scores_", None) is not None else np.zeros(n_features)
            order = np.argsort(scores[mi_idx])[::-1] if scores.size == n_features else np.arange(mi_idx.size)
            mi_idx = mi_idx[order]
            X_mi = X[:, mi_idx]
        except Exception:
            vari = np.var(X, axis=0)
            mi_idx = np.argsort(vari)[::-1][:k]
            X_mi = X[:, mi_idx]

        names_mi = [self._orig_names[i] for i in mi_idx]

        # --- 2) corr filter
        if X_mi.shape[1] <= 1:
            kept_global = list(mi_idx)
            names_unc = names_mi
            X_unc = X_mi
        else:
            C = np.corrcoef(X_mi, rowvar=False)
            keep, dropped = [], set()
            # Handle NaNs in C safely
            C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
            for i in range(C.shape[0]):
                if i in dropped: continue
                keep.append(i)
                for j in range(i+1, C.shape[0]):
                    if abs(C[i, j]) > self.corr_thr:
                        dropped.add(j)
            kept_global = [int(mi_idx[i]) for i in keep]
            names_unc = [names_mi[i] for i in keep]
            X_unc = X[:, kept_global] if kept_global else X[:, :0]

        if X_unc.shape[1] == 0:
            self.selected_indices_, self.selected_names_ = self._ensure_nonempty(X, mi_idx, self._orig_names)
            return self

        # --- 3) SFS (optional)
        if self.sfs_k <= 0 or X_unc.shape[1] == 1:
            self.selected_indices_ = kept_global
            self.selected_names_ = names_unc
            return self

        skf = StratifiedKFold(n_splits=max(2, self.cv_sfs), shuffle=True, random_state=self.random_state)
        selected_rel, remaining = [], list(range(X_unc.shape[1]))

        def cv_score(cols_rel):
            Xs = X_unc[:, cols_rel]
            est = SVC(kernel='rbf', C=self.C, gamma=self.gamma, class_weight='balanced')
            scores = cross_val_score(est, Xs, y, cv=skf, n_jobs=1)
            return float(np.mean(scores))

        target = min(self.sfs_k, X_unc.shape[1])
        while len(selected_rel) < target and remaining:
            best_j, best_sc = None, -1.0
            for j in remaining:
                sc = cv_score(selected_rel + [j])
                if sc > best_sc:
                    best_sc, best_j = sc, j
            selected_rel.append(best_j)
            remaining.remove(best_j)

        if len(selected_rel) == 0:
            self.selected_indices_, self.selected_names_ = self._ensure_nonempty(X, mi_idx, self._orig_names)
        else:
            self.selected_names_ = [names_unc[i] for i in selected_rel]
            self.selected_indices_ = [kept_global[i] for i in selected_rel]
        return self

    def transform(self, X: np.ndarray):
        if not self.selected_indices_:
            return X[:, [0]]  # ultimate guard: never 0 columns
        return X[:, self.selected_indices_]


#===============================================================================
#                            TRAINING UTILITIES
#===============================================================================

def compute_sma(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    n = max(1, len(x))
    return float((np.abs(x).sum() + np.abs(y).sum() + np.abs(z).sum()) / n)

def build_feature_table_for_category(windows: List[Dict], fs: float, category: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    category: "static" or "dynamic"
    Returns: X, y, groups, feature_names
    groups = subject_id (subject-aware GroupKFold)
    """
    X_list, y_list, g_list = [], [], []
    fnames: Optional[List[str]] = None
    target_classes = STATIC_CLASSES if category == "static" else DYNAMIC_CLASSES

    processed = 0
    skipped = 0

    # pre-count how many windows we expect for progress clarity
    total_target = sum(1 for w in windows if w["label"] in target_classes)
    print(f"[{category}] expecting ~{total_target} windows")

    for idx, w in enumerate(windows):
        if w["label"] not in target_classes:
            continue

        # extra debug for late windows
        if processed >= 2950:
            print(f"[{category}] idx={idx} subj={w.get('subject_id')} blk={w.get('block_id')} s={w.get('s')} e={w.get('e')} ...", end="", flush=True)

        try:
            vals, names = compute_features(w["x"], w["y"], w["z"], fs=fs, mode=category)
        except Exception as e:
            skipped += 1
            print(f"\n[{category}] SKIP window idx={idx} subj={w.get('subject_id')} blk={w.get('block_id')} err={e}")
            continue

        if processed >= 2950:
            print(" ok")

        if fnames is None:
            fnames = names
        X_list.append(vals)
        y_list.append(w["label"])
        g_list.append(w["subject_id"])
        processed += 1

        if processed % 500 == 0:
            print(f"[{category}] windows processed (features extracted): {processed}")
        if processed >= 2900 and processed % 25 == 0:
            print(f"[{category}] windows processed (features extracted): {processed}")

    if not X_list:
        raise RuntimeError(f"No windows for category '{category}'.")
    if skipped:
        print(f"[{category}] skipped {skipped} windows due to errors")

    groups, _ = pd.factorize(pd.Series(g_list), sort=True)
    return (np.asarray(X_list, dtype=np.float32),
            np.asarray(y_list, dtype=object),
            np.asarray(groups, dtype=np.int32),
            fnames)

#***** make_pipeline *****
def make_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", 'passthrough'),  # toggled by grid
        ("selector", FeatureSelector()),
        ("svm", SVC(kernel="rbf", probability=False, class_weight="balanced"))
    ])

#***** nested_cv_train_category (dynamic GroupKFold, inner fallback) *****
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
    """
    Train a pipeline with nested CV.
      - Outer split: GroupKFold if >=2 groups, else StratifiedKFold (within-subject).
      - Inner split: GroupKFold if >=2 train groups, else StratifiedKFold.
      - 'fast' shrinks the grid for speed.
    Returns: (best_pipeline_refit_on_all_data, outer_mean_acc, best_params, selected_feature_names)
    """
    # ----- choose outer CV -----
    uniq_groups = np.unique(groups).size
    if uniq_groups >= 2:
        outer_k = min(5, uniq_groups)
        outer = GroupKFold(n_splits=outer_k)
        outer_iter = outer.split(X, y, groups)
        outer_desc = f"GroupKFold (n_splits={outer_k})"
    else:
        # one subject -> within-subject stratified CV
        # ensure at least 2 folds and at least 2 samples per class
        outer_k = min(5, max(2, np.min(np.unique(np.bincount(pd.factorize(y)[0])))))
        outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=RNG)
        outer_iter = outer.split(X, y)
        outer_desc = f"StratifiedKFold (within-subject, n_splits={outer_k})"

    print(f"[{label}] Starting nested CV: {outer_desc}, fast={fast}")

    # ----- parameter grid -----
    if fast:
        param_grid = {
            "pca": ['passthrough'],  # keep simple for speed
            "selector__k_best":   [max(2, default_kbest // 2), default_kbest],
            "selector__sfs_k":    [0, min(2, max(0, default_sfsk))],  # allow 0 to skip SFS if MI already good
            "selector__corr_thr": [0.90],
            "selector__C":        [1.0, 4.0],
            "selector__gamma":    ["scale", 0.05],
            "svm__C":             [1.0, 4.0],
            "svm__gamma":         ["scale", 0.05],
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

    # ----- outer loop -----
    outer_scores: List[float] = []
    fold_id = 0
    for split in outer_iter:
        fold_id += 1
        if uniq_groups >= 2:
            tr_idx, te_idx = split
        else:
            # StratifiedKFold yields (train, test) directly
            tr_idx, te_idx = split

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        g_tr = groups[tr_idx] if uniq_groups >= 2 else None

        # choose inner CV (grouped if >=2 train groups)
        if g_tr is not None and np.unique(g_tr).size >= 2:
            inner_k = min(3, np.unique(g_tr).size)
            inner_cv = GroupKFold(n_splits=inner_k)
            fit_kwargs = {"groups": g_tr, "selector__feature_names": feature_names}
            inner_desc = f"GroupKFold(n_splits={inner_k})"
        else:
            inner_k = 3
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

        # fit inner
        gs.fit(X_tr, y_tr, **fit_kwargs)

        best = gs.best_estimator_
        fold_acc = best.score(X_te, y_te)
        outer_scores.append(fold_acc)
        try:
            print(f"[{label}] Outer fold {fold_id}/{outer_k}: "
                  f"train={len(tr_idx)} test={len(te_idx)} | inner={inner_desc}")
        except Exception:
            print(f"[{label}] Outer fold {fold_id}/{outer_k}: inner={inner_desc}")
        print(f"[{label}]   fold_acc={fold_acc:.3f} | best={gs.best_params_}")

    mean_outer = float(np.mean(outer_scores)) if outer_scores else 0.0
    std_outer  = float(np.std(outer_scores)) if outer_scores else 0.0
    print(f"[{label}] Nested CV done: {mean_outer:.3f} ± {std_outer:.3f}")

    # ----- final refit on all data with a larger inner CV -----
    if uniq_groups >= 2:
        final_groups = groups
        final_ng = np.unique(final_groups).size
        if final_ng >= 2:
            final_inner = GroupKFold(n_splits=min(5, final_ng))
            final_fit_kwargs = {"groups": final_groups, "selector__feature_names": feature_names}
        else:
            final_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
            final_fit_kwargs = {"selector__feature_names": feature_names}
    else:
        final_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
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

    # expose selected feature names
    try:
        selector: FeatureSelector = final_best.named_steps["selector"]  # type: ignore
        sel_names = selector.selected_names_ if selector.selected_names_ else []
    except Exception:
        sel_names = []

    print(f"[{label}] Final best params: {best_params}")
    print(f"[{label}] Selected features: {sel_names}")

    return final_best, mean_outer, best_params, sel_names


#***** sanity: per-subject block/label summary *****
def sanity_print_block_summary(windows: List[Dict], title: str) -> None:
    """
    Prints (a) how many unique blocks per (subject,label) and
           (b) how many windows per (subject,label).
    """
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

    # unique blocks per (subject,label)
    blk = (
        dfw.drop_duplicates(subset=["subject", "label", "block"])
           .groupby(["subject", "label"]).size()
           .unstack(fill_value=0)
           .sort_index()
    )
    print("[sanity] unique blocks per (subject,label):")
    print(blk)

    # window counts per (subject,label)
    cnt = (
        dfw.groupby(["subject", "label"]).size()
           .unstack(fill_value=0)
           .sort_index()
    )
    print("[sanity] window counts per (subject,label):")
    print(cnt)



#===============================================================================
#                                   ONNX
#===============================================================================

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

from sklearn.pipeline import Pipeline as _SkPipeline

def extract_selected_indices(fitted_best: Pipeline) -> List[int]:
    try:
        sel = fitted_best.named_steps.get("selector", None)
        if sel is None:
            return list(range(fitted_best.named_steps["scaler"].n_features_in_))
        if getattr(sel, "selected_indices_", None):
            return sel.selected_indices_
        # fallback: if SFS was disabled and MI+corr kept k columns
        support = getattr(sel, "_orig_names", None)
        return list(range(fitted_best.named_steps["scaler"].n_features_in_)) if support is None else list(range(len(support)))
    except Exception:
        # last resort: keep everything
        return list(range(fitted_best.named_steps["scaler"].n_features_in_))

def make_deployable_pipeline(
    fitted_best: Pipeline,
    X_all: np.ndarray,
    y_all: np.ndarray,
    selected_idx: List[int],
) -> Pipeline:
    """
    Build a new pipeline WITHOUT the custom selector:
        [StandardScaler] -> [optional PCA] -> [SVC]
    and fit it on X_all[:, selected_idx].
    """
    # pull the already-chosen PCA config (or passthrough)
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



#===============================================================================
#                                    MAIN
#===============================================================================

#***** CLI *****
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train static/dynamic SVMs with nested CV and subject-aware splits.")
    p.add_argument("--data_dir", type=str, default="data", help="Directory of CSVs (auto-load all).")
    p.add_argument("--csv", type=str, default="build/accel_calib_data.csv", help="Fallback single CSV if data_dir empty.")
    p.add_argument("--win", type=int, default=WIN)
    p.add_argument("--hop", type=int, default=HOP)
    p.add_argument("--guard", type=int, default=GUARD)
    p.add_argument("--fs", type=float, default=FS)
    p.add_argument("--outdir", type=str, default="models")
    p.add_argument("--fast", action="store_true", help="Smaller grids/folds and single-threaded CV for speed.")
    p.add_argument("--start_activity", type=str, default="", choices=["", "sit", "stand", "walk", "turn_cw"],
               help="If set, rotate label order so the FIRST active block is this activity.")
    p.add_argument(
    "--subject_filter",
    type=str,
    default="",
    help="Comma-separated subject_id stems to include (e.g., 'accel_calib_data_paul')."
    )

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
    FAST = bool(args.fast)
    FS, WIN, HOP, GUARD = float(args.fs), int(args.win), int(args.hop), int(args.guard)

    # Prefer multi-subject directory; fall back to single csv
    data_dir = Path(args.data_dir)
    if data_dir.exists() and any(data_dir.glob("*.csv")):
        print(f"[INFO] Loading all CSVs from: {data_dir}")
        windows = collect_windows_from_dir(str(data_dir), start_activity=args.start_activity)
    else:
        csv_path = args.csv if os.path.exists(args.csv) else "accel_calib_data.csv"
        print(f"[INFO] data_dir empty; using single CSV: {csv_path}")
        windows = build_windows(load_csv(csv_path), subject_id=Path(csv_path).stem, start_activity=args.start_activity)

    #***** optional subject filter (Paul-only, etc.) *****
    if args.subject_filter:
        keep = {s.strip() for s in args.subject_filter.split(",") if s.strip()}
        windows = [w for w in windows if w["subject_id"] in keep]
        print(f"[INFO] Subject filter active -> keeping: {sorted(keep)}; windows now: {len(windows)}")

    #***** sanity report right after window construction *****
    sanity_print_block_summary(windows, "blocks & windows per subject/label")


    if not windows:
        raise RuntimeError("No windows constructed; check labels and window params.")

    # Build category tables (features computed via reusable preprocessing+features)
    Xs, ys, gs, fn_s = build_feature_table_for_category(windows, FS, category="static")
    Xd, yd, gd, fn_d = build_feature_table_for_category(windows, FS, category="dynamic")

    # Train with nested CV (subject-aware grouping)
    static_pipe, static_cv, static_params, static_sel = nested_cv_train_category(
        Xs, ys, gs, fn_s, default_kbest=K_BEST_STATIC, default_sfsk=SFS_K_STATIC, label="Static", fast=FAST
    )
    dynamic_pipe, dynamic_cv, dynamic_params, dynamic_sel = nested_cv_train_category(
        Xd, yd, gd, fn_d, default_kbest=K_BEST_DYNAMIC, default_sfsk=SFS_K_DYNAMIC, label="Dynamic", fast=FAST
    )

    #***** sanity: class balance after feature-table build *****
    print(f"[summary] static windows={len(ys)}, features_per_window={len(fn_s)}")
    print(f"[summary] dynamic windows={len(yd)}, features_per_window={len(fn_d)}")
    print(f"[summary] subjects static={np.unique(gs).size}, dynamic={np.unique(gd).size}")
    print(f"[summary] static y counts: {pd.Series(ys).value_counts().to_dict()}")
    print(f"[summary] dynamic y counts: {pd.Series(yd).value_counts().to_dict()}")


    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ONNX (best effort)
    # ----- build deployable (no custom selector) and export -----
    static_sel_idx = extract_selected_indices(static_pipe)
    dynamic_sel_idx = extract_selected_indices(dynamic_pipe)

    static_deploy  = make_deployable_pipeline(static_pipe,  Xs, ys, static_sel_idx)
    dynamic_deploy = make_deployable_pipeline(dynamic_pipe, Xd, yd, dynamic_sel_idx)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    export_onnx_deploy(static_deploy,  n_features=len(static_sel_idx),  out_path=str(outdir / "static_svm.onnx"))
    export_onnx_deploy(dynamic_deploy, n_features=len(dynamic_sel_idx), out_path=str(outdir / "dynamic_svm.onnx"))


    meta = {
        "fs_hz": FS, "win": WIN, "hop": HOP, "guard": GUARD, "ar_order": AR_ORDER,
        "static": {
            "cv_acc_outer_mean": static_cv,
            "selected_features": [fn_s[i] for i in static_sel_idx],
            "selected_indices": static_sel_idx,
            "classes": STATIC_CLASSES
        },
        "dynamic": {
            "cv_acc_outer_mean": dynamic_cv,
            "selected_features": [fn_d[i] for i in dynamic_sel_idx],
            "selected_indices": dynamic_sel_idx,
            "classes": DYNAMIC_CLASSES
        },
        "note": "ONNX exports a deployable pipeline without the custom FeatureSelector. Feed only selected_indices in the same order."
    }

    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved {outdir/'meta.json'}")
    print("[DONE] Static outer acc (mean): {:.3f}".format(static_cv))
    print("[DONE] Dynamic outer acc (mean): {:.3f}".format(dynamic_cv))
