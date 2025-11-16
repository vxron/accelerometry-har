#!/usr/bin/env python3
"""
Export hierarchical HAR models (intensity pre-classifier + static/dynamic SVMs)
to ONNX along with a small JSON metadata file for use in C++.

Usage (example):

    from har_export import export_har_hierarchy

    export_har_hierarchy(
        (args...)
    )
"""

import json
from pathlib import Path
from typing import Sequence, Mapping

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def make_export_dir(mode: str, person: str | None) -> Path:
    """
    Return the correct output directory:
        models/general/
        models/<person>/
    """
    base = Path("models")
    if mode == "general":
        outdir = base / "general"
    else:
        if person is None:
            raise ValueError("person must be specified for per_person mode.")
        outdir = base / person

    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _export_pipeline_to_onnx(
    model,
    n_features: int,
    out_path: Path,
    input_name: str = "input",
    output_name: str | None = None,
    target_opset: int = 15,
):
    """
    Convert a sklearn Pipeline or estimator to ONNX.

    Assumes model expects a 2D float32 array of shape [N, n_features].
    """
    initial_type = [(input_name, FloatTensorType([None, n_features]))]
    onx = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=target_opset,
    )

    if output_name is not None:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"[ONNX] Saved → {out_path}")


def export_har_hierarchy(
    *,
    intensity_model,
    static_model,
    dynamic_model,
    feat_names: Sequence[str],
    int_sel_idx: Sequence[int],
    static_sel_idx: Sequence[int],
    dyn_sel_idx: Sequence[int],
    label_name_to_id: Mapping[str, int],
    intensity_filename: str = "har_intensity.onnx",
    static_filename: str = "har_static.onnx",
    dynamic_filename: str = "har_dynamic.onnx",
    meta_filename: str = "har_hierarchy_meta.json",
    mode: str = "per_person", # or general
    person: str,
):
    """
    Export the 3-model hierarchical HAR system to ONNX plus metadata JSON.

    Parameters
    ----------
    intensity_model : sklearn Pipeline/estimator
        Pre-classifier: outputs 0 (static) or 1 (dynamic).

    static_model : sklearn Pipeline/estimator
        Binary SVM for static branch: outputs 0 (sit) or 1 (stand).

    dynamic_model : sklearn Pipeline/estimator
        Binary SVM for dynamic branch: outputs 0 (turn) or 1 (walk).

    feat_names : list of str
        Global feature name order (matches Xf_* columns).

    *_sel_idx : list of int
        Column indices into feat_names used by each model.

    label_name_to_id : dict
        Global mapping, e.g. {"stand":0, "sit":1, "walk":2, "turn":3}.

    """
    out_dir = make_export_dir(mode, person)
    print(f"[EXPORT] Saving ONNX models to: {out_dir}")


    # 1) Export ONNX files
    int_path   = out_dir / intensity_filename
    static_path = out_dir / static_filename
    dyn_path    = out_dir / dynamic_filename

    _export_pipeline_to_onnx(
        intensity_model,
        n_features=len(int_sel_idx),
        out_path=int_path,
        input_name="input",
    )

    _export_pipeline_to_onnx(
        static_model,
        n_features=len(static_sel_idx),
        out_path=static_path,
        input_name="input",
    )

    _export_pipeline_to_onnx(
        dynamic_model,
        n_features=len(dyn_sel_idx),
        out_path=dyn_path,
        input_name="input",
    )

    # 2) Build metadata dictionary 
    # Ensure everything JSON serializable
    feat_names = list(feat_names)
    int_sel_idx   = [int(i) for i in int_sel_idx]
    static_sel_idx = [int(i) for i in static_sel_idx]
    dyn_sel_idx    = [int(i) for i in dyn_sel_idx]

    meta = {
        "feat_names": feat_names,
        "intensity": {
            "onnx_file": int_path.name,
            "feature_indices": int_sel_idx,
            "classes": {
                "static" : 0,
                "dynamic" : 1,
            },
        },
        "static_branch": {
            "onnx_file": static_path.name,
            "feature_indices": static_sel_idx,
            "branch_labels": {
                "sit" : 0,
                "stand" : 1,
            },
        },
        "dynamic_branch": {
            "onnx_file": dyn_path.name,
            "feature_indices": dyn_sel_idx,
            "branch_labels": {
                "turn" : 0,
                "walk" : 1,
            },
        },
        "label_name_to_id": {
            str(name): int(idx) for name, idx in label_name_to_id.items()
        },
    }

    meta_path = out_dir / meta_filename
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[META] Saved {meta_path}")

    return {
        "onnx_intensity": str(int_path),
        "onnx_static": str(static_path),
        "onnx_dynamic": str(dyn_path),
        "meta": str(meta_path),
    }
