"""
This file must:
(1) Load CSV (columns: tick,x,y,z,active)
(2) Detect contiguous 'active == 1' blocks and assign labels in fixed order:
sit->stand->walk->turn_cw-> (repeat)
(3) Build sliding windows (win_len=200, hop=100) ONLY if fully inside an activity block,
with guard margins trimmed from both block edges to avoid toggle jitter
(4) feature extraction (time + frequency); standardize
(5) establish an intensity-based pre-classifier to route static {sit,stand} vs dynamic {walk,turn}
(6) train/validate two SVMs (kernels)  -> activity_svm_static and activity_svm_dynamic
(7) Export to ONNX model; Zipmap disabled (so C++ gets a plain tensor); also preclassifier in json

CLI:
  python train_activity_svm.py --csv build/accel_calib_data.csv
                               --win 200 --hop 100 --guard 40
                               --fs 119 --outdir build

Dependencies:
  pip install numpy pandas scikit-learn scipy skl2onnx onnx
"""
# Import Libraries
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ONNX export
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

STATIC_CLASSES = {"sit", "stand"}
DYNAMIC_CLASSES = {"walk", "turn_cw"}
ACTIVITY_ORDER = ["sit", "stand", "walk", "turn_cw"]

# Predefined Training Order:
# sit -> stand -> walk -> turn CW
# So we need to go through data and when theres a sequence of ones, add column for walk, and next time u run into ones, u know its stand, etc

# FEATURE EXTRACTOR


# EXTRACT, THEN NORMALIZE

# Add margin around sliding windows to handle error in toggles/labels


# (1) DATA CLEANING/INIT


# (2) FEATURE EXTRACTOR -> TRAINING SET

# (3) KERNEL SVM CLASSIFIER TRAINING

# LABEL SEQUENCING & WINDOWING *****************************************************
def find_active_blocks(active: np.ndarray) -> List[Tuple[int,int]]:
    """Returns list of [start_idx, end_idx) for contiguous runs where active == 1
    Example: [0 1 1 0 1] -> [(1,2), (4,4)]"""
    blocks = []
    in_run = False
    start = 0
    for i,v in enumerate(active): # how do we know format of enumerate(active) and how do we know it's i,v -> can we name them more intuitive variables please
        # start run
        if v == 1 and not in_run:
            in_run = True
            start = i
        # end of run reached
        elif v == 0 and in_run:
            blocks.append((start,i))
            in_run = False
    # i dont understand why we have this last one outside of the loop
    if in_run:
        blocks.append((start, len(active)))
    return blocks

def apply_guard(blocks: List[Tuple[int,int]], guard: int, min_len:int):
    """Trim guard samples at both edges; drop if shorter than min_len after trimming"""
    kept = []
    for start, end in blocks:
        newStart, newEnd = start+guard, end-guard
        if newEnd - newStart > min_len:
            kept.append(newStart,newEnd)
    return kept

def window_indices(start: int, end: int, win_len: int, hop: int) -> List[Tuple[int,int]]:
    """Generate [w_start,w_end) windows fully inside [start,end)"""
    out = []
    i = start
    while i + win_len <= end:
        out.append((i,i+win_len))
        i+=hop
    return out

def assign_labels_to_blocks(n_blocks: int) -> List[str]:
    """Cycle through ACTIVITY_ORDER"""
    return [ACTIVITY_ORDER[i%len(ACTIVITY_ORDER)] for i in range(n_blocks)]

# FEATURE EXTRACTION *************************************************************


# TRAINING SET BUILDER ************************************************************


# TRAINING & EXPORT **************************************************************



# MAIN *********************************************************************
def main():

    # 1) Load

    # 2) Training set

    # 3) Training

    # 4) Export




