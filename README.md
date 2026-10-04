# Movesense ECG Visualizer

Standalone PyQt viewer for the Movesense ECG CSV in this folder.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Activate venv

```bash
source .venv/bin/activate
```

## Run

```bash
.venv/bin/python movesense_ecg_viewer.py
```

You can also pass a different CSV path:

```bash
.venv/bin/python movesense_ecg_viewer.py /path/to/file.csv
```

The app loads Movesense metadata, shows a full-recording overview, lets you inspect a selected time window, and can export the current view as an image.

## Interaction

- Drag the main ECG plot to pan through time.
- Use the mouse wheel over the main plot to zoom.
- Drag the highlighted region in the overview plot to jump through the recording.
- Click the overview plot to center the selected window around that time.
- Use Back, Forward, Reset, Fit Y, quick window buttons, or the Start and Window fields for precise navigation.
- Toggle Crosshair to read the nearest sample time and ECG value under the mouse.
- Use Export Image to save the current ECG detail plot and overview plot as a PNG or JPEG. The overview includes the selected region and center marker.
- Use Preprocessing controls to apply a 5-40 Hz bandpass filter, optional zero-crossing masks, optional motion variance masks, and optional long R-gap masks. Masked display samples are flattened to `-1`.
- Motion variance uses an adjustable sliding window, defaulting to 2 seconds, with adjustable step, standard deviation threshold, and robust peak-to-peak threshold (`P99 - P1`).
- Signal availability reports total signal, total masked signal, and usable percentage.

## Code layout

- `movesense_ecg_viewer.py`: small launcher script.
- `ecg_viewer.py`: main PyQt window and interaction logic.
- `movesense_data.py`: CSV metadata loading, ECG arrays, downsampling, and Y-axis limits.
- `ecg_preprocessing.py`: bandpass filtering, zero-crossing masks, motion variance masks, long R-gap masks, and combined masked display arrays.
- `plot_helpers.py`: plot formatting, time-axis labels, and pyqtgraph configuration.
- `app_style.py`: Qt stylesheet.
- `description.md`: project architecture, preprocessing parameters, feature notes, and change log.
