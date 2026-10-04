"""Data loading and signal helpers for Movesense ECG exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EcgData:
    path: Path
    metadata: dict[str, str]
    time_s: np.ndarray
    ecg_mv: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.time_s.size)

    @property
    def duration_s(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def start_s(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return float(self.time_s[0])

    @property
    def end_s(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return float(self.time_s[-1])

    @property
    def sample_rate_hz(self) -> float:
        if self.sample_count < 2:
            return 0.0
        spacing = np.median(np.diff(self.time_s[: min(self.sample_count, 10000)]))
        if spacing <= 0:
            return 0.0
        return float(1.0 / spacing)


def read_movesense_csv(path: Path) -> EcgData:
    metadata = read_metadata(path)
    frame = pd.read_csv(
        path,
        comment="#",
        usecols=[0, 1],
        dtype={"Elapsed time": "float64", "ECG": "float32"},
        memory_map=True,
    )
    frame = frame.dropna()

    required = {"Elapsed time", "ECG"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing expected column(s): {', '.join(sorted(missing))}")

    time_s = frame["Elapsed time"].to_numpy(dtype=np.float64, copy=True)
    ecg_mv = frame["ECG"].to_numpy(dtype=np.float32, copy=True)
    if time_s.size == 0:
        raise ValueError("CSV contains no ECG samples.")
    return EcgData(path=path, metadata=metadata, time_s=time_s, ecg_mv=ecg_mv)


def read_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if not body:
                continue
            key, _, value = body.partition(" ")
            metadata[key.lower()] = value.strip()
    return metadata


def downsample_envelope(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    bucket_count = max(1, max_points // 2)
    step = int(np.ceil(x.size / bucket_count))
    trim = (x.size // step) * step
    if trim == 0:
        return x, y

    x_view = x[:trim].reshape(-1, step)
    y_view = y[:trim].reshape(-1, step)
    min_idx = np.argmin(y_view, axis=1)
    max_idx = np.argmax(y_view, axis=1)

    first_idx = np.minimum(min_idx, max_idx)
    second_idx = np.maximum(min_idx, max_idx)
    rows = np.arange(y_view.shape[0])

    x_out = np.empty(y_view.shape[0] * 2, dtype=x.dtype)
    y_out = np.empty(y_view.shape[0] * 2, dtype=y.dtype)
    x_out[0::2] = x_view[rows, first_idx]
    y_out[0::2] = y_view[rows, first_idx]
    x_out[1::2] = x_view[rows, second_idx]
    y_out[1::2] = y_view[rows, second_idx]

    if trim < x.size:
        x_out = np.concatenate([x_out, x[trim:]])
        y_out = np.concatenate([y_out, y[trim:]])
    return x_out, y_out


def robust_y_limits(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return -1.0, 1.0
    low, high = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
    if low == high:
        low -= 0.5
        high += 0.5
    pad = max(0.02, float(high - low) * 0.15)
    return float(low - pad), float(high + pad)
