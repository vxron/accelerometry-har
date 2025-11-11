#!/usr/bin/env python3
"""
Plot raw accelerometry with activity labels + a few feature timelines.

Outputs (per CSV) into: plots/<person>/
  - <stem>_raw.png           : x, y, z, |a| with activity-colored bands
  - <stem>_sma.png           : SMA(t) with bands
  - <stem>_peakfreq.png      : Peak frequency(t) with bands
  - <stem>_sma_hist.png      : SMA histograms by activity

Assumptions:
- CSV columns: tick, x, y, z, active   (active ∈ {0,1})
- Labels assigned by cycling active blocks through:
    stand → walk → sit → turn → repeat
- Plots each CSV in data/<person>/*.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# ---------- Defaults (match your training) ----------
FS_DEFAULT   = 119.0
WIN_DEFAULT  = 200
HOP_DEFAULT  = 100
GUARD_DEFAULT= 50
ACTIVITY_ORDER = ["stand", "walk", "sit", "turn"]

# Colors (nice, distinct)
ACTIVITY_COLORS = {
    "stand": "#4C78A8",
    "walk":  "#54A24B",
    "sit":   "#F58518",
    "turn":  "#E45756",
}
AXIS_COLORS = {
    "x": "#3E5C76",
    "y": "#7A5195",
    "z": "#BC5090",
    "mag": "#2F4B7C",
}

# ---------- Labeling helpers (same logic as training) ----------
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

def assign_labels_to_blocks(n: int, one_blocks: List[Tuple[int, int]], order: List[str]) -> np.ndarray:
    labels = np.array([""] * n, dtype=object)
    for k, (a, b) in enumerate(one_blocks):
        cls = order[k % len(order)]
        labels[a:b] = cls
    return labels

def windows_from_block(start: int, end: int, win: int, hop: int, guard: int) -> List[Tuple[int, int]]:
    s, e = start + guard, end - guard
    spans, i = [], s
    while i + win <= e:
        spans.append((i, i + win))
        i += hop
    return spans

# ---------- Feature helpers (consistent with training) ----------
def preprocess_hp(x: np.ndarray, fs: float, fc: float = 0.5, order: int = 2) -> np.ndarray:
    """Butter HP (filtfilt). Fallback to demean if too short or numerical issue."""
    from scipy.signal import butter, filtfilt
    s = np.asarray(x, dtype=np.float64)
    if s.size < max(8, order * 3):
        return s - np.mean(s)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    wn = max(min(fc / (0.5 * fs), 0.999), 1e-5)
    try:
        b, a = butter(order, wn, btype='highpass')
        return filtfilt(b, a, s, method="gust")
    except Exception:
        s0 = s - np.mean(s)
        d = np.diff(s0, prepend=s0[0])
        return d - np.mean(d)

def mag_from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.sqrt(x*x + y*y + z*z)

def compute_sma(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    n = max(1, len(x))
    return float((np.abs(x).sum() + np.abs(y).sum() + np.abs(z).sum()) / n)

def peak_freq(sig: np.ndarray, fs: float) -> float:
    """Welch peak frequency (Hz) with safe fallbacks."""
    from scipy.signal import welch
    s = np.asarray(sig, dtype=np.float64)
    if s.size < 16:
        return 0.0
    nperseg = int(min(s.size, 256))
    try:
        f, Pxx = welch(s - s.mean(), fs=fs, nperseg=nperseg)
        if f.size == 0 or Pxx.size == 0:
            return 0.0
        return float(f[int(np.argmax(Pxx))])
    except Exception:
        return 0.0

# ---------- Plotting ----------
def _shade_blocks(ax, t: np.ndarray, blocks: List[Tuple[int,int]], labels: np.ndarray):
    for (a, b) in blocks:
        lab = labels[a] if a < labels.size else ""
        if not lab:
            continue
        c = ACTIVITY_COLORS.get(lab, "#BBBBBB")
        ax.axvspan(t[a], t[b-1] if b-1 < t.size else t[-1], color=c, alpha=0.12, lw=0)

def _legend_patch_handles():
    import matplotlib.patches as mpatches
    return [mpatches.Patch(color=ACTIVITY_COLORS[k], label=k) for k in ACTIVITY_COLORS]

def plot_raw_with_labels(outpng: Path, df: pd.DataFrame, fs: float, labels: np.ndarray, blocks: List[Tuple[int,int]]):
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    mag = mag_from_xyz(x,y,z)

    if "tick" in df.columns:
        t = df["tick"].to_numpy(dtype=float) / fs
    else:
        t = np.arange(len(x), dtype=float) / fs

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=140)
    ax.plot(t, x, lw=0.8, label="x", color=AXIS_COLORS["x"])
    ax.plot(t, y, lw=0.8, label="y", color=AXIS_COLORS["y"])
    ax.plot(t, z, lw=0.8, label="z", color=AXIS_COLORS["z"])
    ax.plot(t, mag, lw=1.1, label="|a|", color=AXIS_COLORS["mag"])

    _shade_blocks(ax, t, blocks, labels)
    for h in _legend_patch_handles():
        ax.add_artist(plt.Line2D([], [], color=h.get_facecolor(), lw=4, alpha=0.6))
    ax.legend(loc="upper right", frameon=True)
    ax.set_title("Raw accelerometry with activity bands")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (units)")

    fig.tight_layout()
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)

def plot_sma_timeline(outpng: Path, df: pd.DataFrame, fs: float, labels: np.ndarray,
                      blocks: List[Tuple[int,int]], win: int, hop: int, guard: int):
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    active = df["active"].astype(int).to_numpy()

    # Preprocess like training (HP 0.5 Hz) before SMA
    xhp = preprocess_hp(x, fs, fc=0.5)
    yhp = preprocess_hp(y, fs, fc=0.5)
    zhp = preprocess_hp(z, fs, fc=0.5)

    # Build windows inside each active block with guards
    spans = []
    for (a, b) in blocks:
        spans.extend(windows_from_block(a, b, win, hop, guard))

    t_mid = []
    sma_vals = []
    for s, e in spans:
        sma_vals.append(compute_sma(xhp[s:e], yhp[s:e], zhp[s:e]))
        t_mid.append((s + e) * 0.5 / fs)

    if not t_mid:
        return

    t = np.array(t_mid)
    v = np.array(sma_vals)

    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=140)
    ax.plot(t, v, lw=1.2)
    # For shading, build a continuous time vector to reuse bands
    full_t = np.arange(len(x), dtype=float) / fs
    _shade_blocks(ax, full_t, blocks, labels)

    ax.set_title("SMA over time (preprocessed, HP 0.5 Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SMA")
    fig.tight_layout()
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)

def plot_peakfreq_timeline(outpng: Path, df: pd.DataFrame, fs: float, labels: np.ndarray,
                           blocks: List[Tuple[int,int]], win: int, hop: int, guard: int):
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)

    # Preprocess like training prior to spectral features
    xhp = preprocess_hp(x, fs, fc=0.5)
    yhp = preprocess_hp(y, fs, fc=0.5)
    zhp = preprocess_hp(z, fs, fc=0.5)
    mag = mag_from_xyz(xhp, yhp, zhp)

    spans = []
    for (a, b) in blocks:
        spans.extend(windows_from_block(a, b, win, hop, guard))

    t_mid = []
    fpk = []
    for s, e in spans:
        t_mid.append((s + e) * 0.5 / fs)
        fpk.append(peak_freq(mag[s:e], fs))

    if not t_mid:
        return

    t = np.array(t_mid)
    v = np.array(fpk)

    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=140)
    ax.plot(t, v, lw=1.2)
    full_t = np.arange(len(x), dtype=float) / fs
    _shade_blocks(ax, full_t, blocks, labels)

    ax.set_title("Peak frequency over time (Welch, HP 0.5 Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("f_peak (Hz)")
    fig.tight_layout()
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)

def plot_sma_histograms(outpng: Path, df: pd.DataFrame, fs: float, labels: np.ndarray,
                        blocks: List[Tuple[int,int]], win: int, hop: int, guard: int):
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)

    xhp = preprocess_hp(x, fs, fc=0.5)
    yhp = preprocess_hp(y, fs, fc=0.5)
    zhp = preprocess_hp(z, fs, fc=0.5)

    # window-level labels via majority label inside window (here, constant per block)
    recs = []
    for (a, b) in blocks:
        lab = labels[a] if a < labels.size else ""
        for s, e in windows_from_block(a, b, win, hop, guard):
            sma = compute_sma(xhp[s:e], yhp[s:e], zhp[s:e])
            recs.append((lab, sma))

    if not recs:
        return
    d = pd.DataFrame(recs, columns=["label", "sma"])

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    bins = 30
    for lab, g in d.groupby("label"):
        c = ACTIVITY_COLORS.get(lab, "#777777")
        ax.hist(g["sma"].to_numpy(), bins=bins, alpha=0.45, label=lab, color=c, edgecolor="white", linewidth=0.5)
    ax.set_title("SMA distributions by activity (HP 0.5 Hz)")
    ax.set_xlabel("SMA")
    ax.set_ylabel("Count")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)

# ---------- IO / main ----------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["x", "y", "z", "active"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    return df

def run_for_file(csv_path: Path, person: str, out_root: Path,
                 fs: float, win: int, hop: int, guard: int, start_activity: str):
    outdir = out_root / person
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)
    active = df["active"].astype(int).to_numpy()
    blocks = find_active_blocks(active)

    # Rotate activity order so the first active block matches start_activity (if provided)
    order = ACTIVITY_ORDER[:]
    if start_activity:
        if start_activity in order:
            k = order.index(start_activity)
            order = order[k:] + order[:k]

    labels = assign_labels_to_blocks(len(active), blocks, order)

    stem = csv_path.stem
    plot_raw_with_labels(outdir / f"{stem}_raw.png", df, fs, labels, blocks)
    plot_sma_timeline(outdir / f"{stem}_sma.png", df, fs, labels, blocks, win, hop, guard)
    plot_peakfreq_timeline(outdir / f"{stem}_peakfreq.png", df, fs, labels, blocks, win, hop, guard)
    plot_sma_histograms(outdir / f"{stem}_sma_hist.png", df, fs, labels, blocks, win, hop, guard)
    print(f"[OK] Plotted: {csv_path.name} → {outdir}")

def main():
    ap = argparse.ArgumentParser(description="Plot raw accelerometry + labeled bands + feature timelines.")
    ap.add_argument("--person", required=True, help="Person folder under data/, e.g. --person veronica")
    ap.add_argument("--data_root", default="data", help="Root folder containing per-person subfolders")
    ap.add_argument("--fs", type=float, default=FS_DEFAULT, help="Sampling rate (Hz)")
    ap.add_argument("--win", type=int, default=WIN_DEFAULT, help="Window length (samples)")
    ap.add_argument("--hop", type=int, default=HOP_DEFAULT, help="Hop (samples)")
    ap.add_argument("--guard", type=int, default=GUARD_DEFAULT, help="Guard (samples) trimmed inside each block")
    ap.add_argument("--start_activity", type=str, default="", choices=["", "sit", "stand", "walk", "turn"],
                    help="Rotate label order so FIRST active block is this activity")
    ap.add_argument("--outdir", default="plots", help="Output root directory for plots")
    args = ap.parse_args()

    base = Path(args.data_root) / args.person
    csvs = sorted(base.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found for person '{args.person}' under {base}")

    out_root = Path(args.outdir)

    for csv_path in csvs:
        run_for_file(csv_path, args.person, out_root, args.fs, args.win, args.hop, args.guard, args.start_activity)

if __name__ == "__main__":
    main()
