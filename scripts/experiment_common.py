"""Shared parsing and metrics for the sensitivity analyses."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


def normalise_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def binary_prediction(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return float(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return float(value)
    text = str(value).strip()
    patterns = [
        r"(?is)(?:answer|output)\s*(?:is|:|=|\*|\s)*\s*([01])\b",
        r"(?m)^\s*([01])(?:\s*[.\-*]|\s*$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    standalone = re.findall(r"(?<!\d)([01])(?!\d)", text)
    return float(standalone[-1]) if standalone else np.nan


def load_predictions(path: Path, id_column: str = "id", prediction_column: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if id_column not in frame and "filename" in frame:
        frame[id_column] = frame["filename"].astype(str).map(lambda value: Path(value).stem)
    if id_column not in frame:
        raise ValueError(f"{path} has no {id_column!r} or 'filename' column")
    if prediction_column is None:
        candidates = [
            col for col in frame.columns
            if col not in {id_column, "filename", "fps", "model", "prompt_method", "status", "error"}
            and not col.lower().startswith("unnamed")
            and not col.startswith("elapsed_time_")
        ]
        preferred = [col for col in candidates if col == "response" or col.startswith("label_") or col == "videollama"]
        if len(preferred) != 1:
            raise ValueError(f"Could not uniquely infer prediction column in {path}: {preferred or candidates}")
        prediction_column = preferred[0]
    frame = frame.copy()
    frame["id"] = normalise_id(frame[id_column])
    frame["prediction"] = frame[prediction_column].map(binary_prediction)
    frame["source"] = str(path)
    if "model" not in frame:
        frame["model"] = prediction_column.removeprefix("label_")
    if "prompt_method" not in frame:
        parts = path.parts
        frame["prompt_method"] = next((part for part in parts if "prompt" in part), "unknown")
    return frame


def classification_metrics(frame: pd.DataFrame, truth: str = "true_label", prediction: str = "prediction") -> dict[str, float | int]:
    valid = frame.dropna(subset=[truth, prediction]).copy()
    if valid.empty:
        return {"n_total": len(frame), "n": 0, "n_invalid": len(frame), "accuracy": np.nan, "balanced_accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan,
                "tn": 0, "fp": 0, "fn": 0, "tp": 0}
    y_true = valid[truth].astype(int)
    y_pred = valid[prediction].astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_total": len(frame), "n": len(valid), "n_invalid": len(frame) - len(valid),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision, "recall": recall, "f1": f1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def correlation(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    return correlation_test(x, y, method)[0]


def correlation_test(x: pd.Series, y: pd.Series, method: str = "pearson") -> tuple[float, float]:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan, np.nan
    result = pearsonr(pair["x"], pair["y"]) if method == "pearson" else spearmanr(pair["x"], pair["y"])
    return float(result.statistic), float(result.pvalue)
