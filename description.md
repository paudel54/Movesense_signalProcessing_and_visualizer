# Movesense ECG Visualizer Architecture

## Purpose

This project is a PyQt desktop ECG inspection tool for Movesense CSV exports. It preserves the original CSV timestamps, sample count, and ECG values while letting the user view raw, filtered, and masked display waveforms.

## File Structure

- `movesense_ecg_viewer.py`: command-line launcher.
- `ecg_viewer.py`: main PyQt window, controls, plot interaction, preprocessing UI, and image export.
- `movesense_data.py`: CSV metadata loading, ECG arrays, downsampling, and robust Y-axis limits.
- `ecg_preprocessing.py`: ECG preprocessing pipeline and artifact masks.
- `plot_helpers.py`: pyqtgraph setup and elapsed-time axis formatting.
- `app_style.py`: Qt stylesheet.
- `requirements.txt`: Python dependencies.
- `README.md`: run instructions and user-facing controls.

## Current Features

- Load Movesense ECG CSV files with metadata header support.
- Interactive ECG detail plot with pan, zoom, crosshair, quick window controls, and overview navigation.
- Export the current detail and overview plots as PNG or JPEG.
- Display raw ECG by default.
- Optional bandpass-filtered display waveform.
- Optional artifact masks that flatten flagged display samples to `-1` without changing the source data.
- Usable-signal reporting based on the combined mask.

## Preprocessing Pipeline

The raw ECG array is never modified. The preprocessing pipeline creates a separate display array with the same sample count and timestamps.

1. Bandpass filtering
   - Default low cutoff: `5 Hz`.
   - Default high cutoff: `40 Hz`.
   - Default filter order: `4`.
   - Implementation: Butterworth bandpass designed with `scipy.signal.butter(..., output="sos")` and applied with zero-phase `scipy.signal.sosfiltfilt`.
   - No existing `apply_bandpass_filter` helper was found in this project when the feature was added.

2. Zero-crossing noise detection
   - Requires bandpass filtering to be enabled.
   - Uses the filtered waveform before any samples are replaced by `-1`.
   - Divides the filtered ECG into full `2 second` non-overlapping windows.
   - Computes zero crossings per second as `sign changes / (samples - 1) * sample_rate`.
   - Flags full windows where the rate is strictly greater than the user cutoff.
   - Default cutoff: `30 crossings/second`.

3. Motion variance detection
   - Requires bandpass filtering to be enabled.
   - Uses the filtered waveform before any samples are replaced by `-1`.
   - Uses an adjustable sliding window.
   - Default window: `2.0 seconds`.
   - Default step: `0.5 seconds`.
   - Computes window standard deviation.
   - Computes robust peak-to-peak amplitude as `P99 - P1`, where `P99` is the 99th percentile and `P1` is the 1st percentile in the window.
   - Default standard deviation threshold: `1.0 mV`.
   - Default robust peak-to-peak threshold: `3.0 mV`.
   - Flags the whole sliding window when either threshold is strictly exceeded.
   - This detector is meant to catch sustained high-amplitude motion artifact while reducing sensitivity to isolated single-sample spikes.

4. Long R-peak gap detection
   - Requires bandpass filtering to be enabled.
   - Uses the filtered waveform before any samples are replaced by `-1`.
   - Peak threshold: `0.42 * configured maximum R-peak height`.
   - Minimum peak separation: `0.4 seconds`.
   - Flags gaps strictly greater than the configured gap threshold.
   - Default gap threshold: `1.5 seconds`.
   - Masks from `0.2 seconds` after the first peak to `0.2 seconds` before the next peak.
   - Long gaps indicate suspected artifacts or missed beats. They do not prove that the signal is noisy.

The ZCR mask, motion variance mask, and R-gap mask are combined with logical OR, then the combined mask is applied to a separate output display array by setting flagged samples to `-1`.

## Reporting

The preprocessing panel reports:

- ZCR masked duration.
- Motion variance masked duration.
- R-gap masked duration.
- Combined masked duration and percentage with overlapping samples counted once.
- Unassessed trailing samples from the full-window ZCR method.
- Initial available signal duration.
- Total masked signal duration.
- Usable signal duration and usable percentage.

Samples flattened to `-1` by the combined mask are treated as not usable. Usable percentage is calculated as:

```text
usable % = 100 * (total samples - combined masked samples) / total samples
```

## Change Log

- Added modular project structure.
- Added interactive pan, zoom, crosshair, overview selection, and image export.
- Removed CSV window export from the UI.
- Added preprocessing panel with bandpass filtering, zero-crossing artifact masking, and long R-peak gap masking.
- Added motion variance masking with sliding-window standard deviation and robust peak-to-peak thresholds.
- Added total available, total masked, usable duration, and usable percentage reporting.
