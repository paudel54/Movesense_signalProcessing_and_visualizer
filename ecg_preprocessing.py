"""ECG preprocessing and artifact-mask helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


MASK_VALUE = -1.0


@dataclass(frozen=True)
class PreprocessingSettings:
    apply_bandpass: bool = False
    low_cut_hz: float = 5.0
    high_cut_hz: float = 40.0
    filter_order: int = 4
    apply_zcr: bool = False
    zcr_window_sec: float = 2.0
    zcr_cutoff_hz: float = 30.0
    apply_motion: bool = False
    motion_window_sec: float = 2.0
    motion_step_sec: float = 0.5
    motion_std_threshold_mv: float = 1.0
    motion_ptp_threshold_mv: float = 3.0
    motion_percentile_low: float = 1.0
    motion_percentile_high: float = 99.0
    apply_rgap: bool = False
    max_r_peak_height: float = 0.55
    r_peak_height_scale: float = 0.42
    min_rr_sec: float = 0.4
    max_rr_sec: float = 1.5
    protect_sec: float = 0.2
    mask_value: float = MASK_VALUE


@dataclass
class PreprocessingResult:
    display_ecg: np.ndarray
    filtered_ecg: np.ndarray | None = None
    zcr_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    motion_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    rgap_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    combined_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    zcr_records: list[tuple[int, int, float, bool]] = field(default_factory=list)
    motion_records: list[tuple[int, int, float, float, bool]] = field(default_factory=list)
    r_peaks: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    rgap_intervals: list[tuple[int, int]] = field(default_factory=list)
    trailing_zcr_samples: int = 0
    trailing_motion_samples: int = 0
    message: str = "Raw ECG"

    def masked_seconds(self, fs: float) -> tuple[float, float, float, float]:
        if fs <= 0:
            return 0.0, 0.0, 0.0, 0.0
        zcr_sec = float(np.count_nonzero(self.zcr_mask) / fs)
        motion_sec = float(np.count_nonzero(self.motion_mask) / fs)
        rgap_sec = float(np.count_nonzero(self.rgap_mask) / fs)
        combined_sec = float(np.count_nonzero(self.combined_mask) / fs)
        return zcr_sec, motion_sec, rgap_sec, combined_sec

    def total_seconds(self, fs: float) -> float:
        if fs <= 0 or self.combined_mask.size == 0:
            return 0.0
        return float(self.combined_mask.size / fs)

    def usable_seconds(self, fs: float) -> float:
        if fs <= 0 or self.combined_mask.size == 0:
            return 0.0
        usable_samples = self.combined_mask.size - np.count_nonzero(self.combined_mask)
        return float(usable_samples / fs)

    def combined_percent(self) -> float:
        if self.combined_mask.size == 0:
            return 0.0
        return float(100.0 * np.count_nonzero(self.combined_mask) / self.combined_mask.size)

    def usable_percent(self) -> float:
        return 100.0 - self.combined_percent()


def apply_preprocessing(
    ecg_raw: np.ndarray,
    fs: float,
    settings: PreprocessingSettings,
) -> PreprocessingResult:
    raw = np.asarray(ecg_raw, dtype=float)
    empty_mask = np.zeros(len(raw), dtype=bool)

    if not settings.apply_bandpass:
        return PreprocessingResult(
            display_ecg=raw.copy(),
            zcr_mask=empty_mask.copy(),
            motion_mask=empty_mask.copy(),
            rgap_mask=empty_mask.copy(),
            combined_mask=empty_mask.copy(),
        )

    filtered = bandpass_ecg(
        raw,
        fs=fs,
        low=settings.low_cut_hz,
        high=settings.high_cut_hz,
        order=settings.filter_order,
    )

    zcr_mask = empty_mask.copy()
    zcr_records: list[tuple[int, int, float, bool]] = []
    trailing_zcr_samples = 0
    if settings.apply_zcr:
        zcr_mask, zcr_records, trailing_zcr_samples = zero_crossing_mask(
            filtered,
            fs=fs,
            window_sec=settings.zcr_window_sec,
            cutoff_hz=settings.zcr_cutoff_hz,
        )

    motion_mask = empty_mask.copy()
    motion_records: list[tuple[int, int, float, float, bool]] = []
    trailing_motion_samples = 0
    if settings.apply_motion:
        motion_mask, motion_records, trailing_motion_samples = motion_variability_mask(
            filtered,
            fs=fs,
            window_sec=settings.motion_window_sec,
            step_sec=settings.motion_step_sec,
            std_threshold_mv=settings.motion_std_threshold_mv,
            ptp_threshold_mv=settings.motion_ptp_threshold_mv,
            percentile_low=settings.motion_percentile_low,
            percentile_high=settings.motion_percentile_high,
        )

    rgap_mask = empty_mask.copy()
    r_peaks = np.array([], dtype=int)
    rgap_intervals: list[tuple[int, int]] = []
    if settings.apply_rgap:
        rgap_mask, r_peaks, rgap_intervals = long_rpeak_gap_mask(
            filtered,
            max_peak_height=settings.max_r_peak_height,
            fs=fs,
            height_scale=settings.r_peak_height_scale,
            min_rr_sec=settings.min_rr_sec,
            max_rr_sec=settings.max_rr_sec,
            protect_sec=settings.protect_sec,
        )

    combined_mask = zcr_mask | motion_mask | rgap_mask
    display = filtered.copy()
    display[combined_mask] = settings.mask_value
    message = "Bandpass filtered ECG"
    if settings.apply_zcr or settings.apply_motion or settings.apply_rgap:
        message = "Filtered ECG with artifact masks"

    return PreprocessingResult(
        display_ecg=display,
        filtered_ecg=filtered,
        zcr_mask=zcr_mask,
        motion_mask=motion_mask,
        rgap_mask=rgap_mask,
        combined_mask=combined_mask,
        zcr_records=zcr_records,
        motion_records=motion_records,
        r_peaks=r_peaks,
        rgap_intervals=rgap_intervals,
        trailing_zcr_samples=trailing_zcr_samples,
        trailing_motion_samples=trailing_motion_samples,
        message=message,
    )


def bandpass_ecg(
    ecg: np.ndarray,
    fs: float,
    low: float = 5.0,
    high: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Parameters use the project defaults from the preprocessing panel:
    low cutoff 5 Hz, high cutoff 40 Hz, and 4th-order Butterworth sections.
    The implementation designs second-order sections with scipy.signal.butter
    and applies zero-phase filtering with scipy.signal.sosfiltfilt.
    """
    x = np.asarray(ecg, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("ECG must be a finite, one-dimensional array.")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if order < 1:
        raise ValueError("Filter order must be at least 1.")
    if not 0 < low < high < fs / 2:
        raise ValueError("Require 0 < low < high < fs/2.")

    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def zero_crossing_mask(
    ecg_filtered: np.ndarray,
    fs: float,
    window_sec: float = 2.0,
    cutoff_hz: float = 30.0,
) -> tuple[np.ndarray, list[tuple[int, int, float, bool]], int]:
    x = np.asarray(ecg_filtered, dtype=float)
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    mask = np.zeros(len(x), dtype=bool)
    win = int(round(window_sec * fs))
    if win < 2:
        raise ValueError("ZCR windows must contain at least two samples.")

    records: list[tuple[int, int, float, bool]] = []
    for start in range(0, len(x) - win + 1, win):
        end = start + win
        window = x[start:end]
        crossings = int(np.count_nonzero(np.diff(np.signbit(window))))
        rate_hz = crossings / (len(window) - 1) * fs
        flagged = rate_hz > cutoff_hz
        if flagged:
            mask[start:end] = True
        records.append((start, end, float(rate_hz), bool(flagged)))

    trailing_samples = len(x) % win
    return mask, records, trailing_samples


def long_rpeak_gap_mask(
    ecg_filtered: np.ndarray,
    max_peak_height: float,
    fs: float,
    height_scale: float = 0.42,
    min_rr_sec: float = 0.4,
    max_rr_sec: float = 1.5,
    protect_sec: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if max_peak_height <= 0:
        raise ValueError("max_peak_height must be positive.")

    x = np.asarray(ecg_filtered, dtype=float)
    peaks, _ = find_peaks(
        x,
        height=height_scale * max_peak_height,
        distance=max(1, int(round(min_rr_sec * fs))),
    )

    mask = np.zeros(len(x), dtype=bool)
    intervals: list[tuple[int, int]] = []
    protect = int(round(protect_sec * fs))

    for left, right in zip(peaks[:-1], peaks[1:]):
        if (right - left) / fs > max_rr_sec:
            start = int(left + protect)
            end = int(right - protect)
            if start < end:
                mask[start:end] = True
                intervals.append((start, end))

    return mask, peaks, intervals


def motion_variability_mask(
    ecg_filtered: np.ndarray,
    fs: float,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    std_threshold_mv: float = 1.0,
    ptp_threshold_mv: float = 3.0,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> tuple[np.ndarray, list[tuple[int, int, float, float, bool]], int]:
    """Flag sliding windows with sustained high variability.

    Peak-to-peak is computed robustly as percentile_high - percentile_low
    on the filtered waveform, defaulting to P99 - P1.
    """
    x = np.asarray(ecg_filtered, dtype=float)
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if not 0 <= percentile_low < percentile_high <= 100:
        raise ValueError("Require 0 <= percentile_low < percentile_high <= 100.")
    if std_threshold_mv <= 0:
        raise ValueError("STD threshold must be positive.")
    if ptp_threshold_mv <= 0:
        raise ValueError("Peak-to-peak threshold must be positive.")

    win = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    if win < 2:
        raise ValueError("Motion windows must contain at least two samples.")
    if step < 1:
        raise ValueError("Motion step must contain at least one sample.")

    mask = np.zeros(len(x), dtype=bool)
    records: list[tuple[int, int, float, float, bool]] = []
    last_assessed_end = 0

    for start in range(0, len(x) - win + 1, step):
        end = start + win
        window = x[start:end]
        std_mv = float(np.std(window))
        lower = float(np.percentile(window, percentile_low))
        upper = float(np.percentile(window, percentile_high))
        ptp_mv = upper - lower
        flagged = std_mv > std_threshold_mv or ptp_mv > ptp_threshold_mv
        if flagged:
            mask[start:end] = True
        records.append((start, end, std_mv, ptp_mv, bool(flagged)))
        last_assessed_end = end

    trailing_samples = max(0, len(x) - last_assessed_end)
    return mask, records, trailing_samples
