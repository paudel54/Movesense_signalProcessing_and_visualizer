#!/usr/bin/env python3
"""Main ECG viewer window for Movesense CSV exports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from app_style import APP_STYLESHEET
from ecg_preprocessing import PreprocessingResult, PreprocessingSettings, apply_preprocessing
from movesense_data import EcgData, downsample_envelope, read_movesense_csv, robust_y_limits
from plot_helpers import TimeAxisItem, configure_pyqtgraph, format_seconds

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - shown only when dependency is absent.
    raise SystemExit(
        "PyQt6 is required to run this viewer.\n"
        "Install it with: python3 -m pip install -r requirements.txt"
    ) from exc

try:
    import pyqtgraph as pg
except ImportError as exc:  # pragma: no cover - shown only when dependency is absent.
    raise SystemExit(
        "pyqtgraph is required for the interactive viewer.\n"
        "Install it with: python3 -m pip install -r requirements.txt"
    ) from exc


DEFAULT_CSV = "MovesenseECG-2026-09-15T19_20_59.226951Z.csv"
MAX_DETAIL_POINTS = 24000
MAX_OVERVIEW_POINTS = 10000
DEFAULT_WINDOW_SECONDS = 10.0


configure_pyqtgraph()


class EcgViewer(QtWidgets.QMainWindow):
    def __init__(self, initial_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Movesense ECG Visualizer")
        self.resize(1440, 900)

        self.data: EcgData | None = None
        self.display_ecg: np.ndarray = np.array([])
        self.preprocessing_result: PreprocessingResult | None = None
        self.visible_time: np.ndarray = np.array([])
        self.visible_ecg: np.ndarray = np.array([])
        self._updating_region = False
        self._updating_range = False
        self._updating_preprocessing_controls = False
        self._last_mouse_x: float | None = None

        self._build_ui()
        self._apply_style()

        if initial_path and initial_path.exists():
            self.load_csv(initial_path)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        root.addLayout(header)

        title = QtWidgets.QLabel("Movesense ECG Visualizer")
        title.setObjectName("Title")
        header.addWidget(title)

        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setObjectName("FileLabel")
        self.file_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self.file_label, stretch=1)

        self.open_button = QtWidgets.QPushButton("Open CSV")
        self.open_button.clicked.connect(self.choose_file)
        header.addWidget(self.open_button)

        self.export_image_button = QtWidgets.QPushButton("Export Image")
        self.export_image_button.clicked.connect(self.export_current_image)
        self.export_image_button.setEnabled(False)
        header.addWidget(self.export_image_button)

        controls = QtWidgets.QFrame()
        controls.setObjectName("Panel")
        controls_layout = QtWidgets.QGridLayout(controls)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setVerticalSpacing(8)
        root.addWidget(controls)

        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setRange(0, 0)
        self.start_spin.setKeyboardTracking(False)
        self.start_spin.setSuffix(" s")
        self.start_spin.valueChanged.connect(self.start_changed)

        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setDecimals(3)
        self.window_spin.setRange(0.050, 3600.0)
        self.window_spin.setValue(DEFAULT_WINDOW_SECONDS)
        self.window_spin.setKeyboardTracking(False)
        self.window_spin.setSuffix(" s")
        self.window_spin.valueChanged.connect(self.window_changed)

        self.position_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 10000)
        self.position_slider.valueChanged.connect(self.slider_changed)

        self.auto_y_check = QtWidgets.QCheckBox("Auto Y")
        self.auto_y_check.setChecked(True)
        self.auto_y_check.stateChanged.connect(self.refresh_detail_plot)

        self.crosshair_check = QtWidgets.QCheckBox("Crosshair")
        self.crosshair_check.setChecked(True)
        self.crosshair_check.stateChanged.connect(self.set_crosshair_visible)

        self.cursor_label = QtWidgets.QLabel("Cursor: n.a.")
        self.cursor_label.setMinimumWidth(260)
        self.cursor_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_view)

        self.fit_y_button = QtWidgets.QPushButton("Fit Y")
        self.fit_y_button.clicked.connect(self.fit_y_to_visible)

        self.prev_button = QtWidgets.QPushButton("Back")
        self.prev_button.clicked.connect(lambda: self.shift_window(-0.85))

        self.next_button = QtWidgets.QPushButton("Forward")
        self.next_button.clicked.connect(lambda: self.shift_window(0.85))

        controls_layout.addWidget(QtWidgets.QLabel("Start"), 0, 0)
        controls_layout.addWidget(self.start_spin, 0, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Window"), 0, 2)
        controls_layout.addWidget(self.window_spin, 0, 3)
        controls_layout.addWidget(self.prev_button, 0, 4)
        controls_layout.addWidget(self.next_button, 0, 5)
        controls_layout.addWidget(self.reset_button, 0, 6)
        controls_layout.addWidget(self.fit_y_button, 0, 7)
        controls_layout.addWidget(self.auto_y_check, 0, 8)
        controls_layout.addWidget(self.crosshair_check, 0, 9)
        controls_layout.addWidget(self.cursor_label, 0, 10)

        quick_windows = QtWidgets.QButtonGroup(self)
        quick_row = QtWidgets.QHBoxLayout()
        for seconds in (1, 5, 10, 30, 60, 300):
            button = QtWidgets.QPushButton(format_seconds(seconds))
            button.setProperty("quick", True)
            button.clicked.connect(lambda checked=False, value=float(seconds): self.set_window_seconds(value))
            quick_windows.addButton(button)
            quick_row.addWidget(button)
        quick_row.addStretch(1)

        controls_layout.addWidget(QtWidgets.QLabel("Position"), 1, 0)
        controls_layout.addWidget(self.position_slider, 1, 1, 1, 7)
        controls_layout.addLayout(quick_row, 1, 8, 1, 3)

        self._build_preprocessing_panel(root)

        self.stats_label = QtWidgets.QLabel("Load a Movesense CSV to begin.")
        self.stats_label.setObjectName("Stats")
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.stats_label)

        self.plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        root.addWidget(self.plot_splitter, stretch=1)

        self.detail_plot = pg.PlotWidget(axisItems={"bottom": TimeAxisItem(orientation="bottom")})
        self.detail_plot.setObjectName("DetailPlot")
        self.detail_plot.setLabel("left", "ECG (mV)")
        self.detail_plot.setLabel("bottom", "Elapsed time")
        self.detail_plot.getAxis("left").enableAutoSIPrefix(False)
        self.detail_plot.showGrid(x=True, y=True, alpha=0.22)
        self.detail_plot.setMouseEnabled(x=True, y=True)
        self.detail_plot.setMenuEnabled(False)
        self.detail_plot.hideButtons()
        self.detail_plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.detail_curve = self.detail_plot.plot(
            [],
            [],
            pen=pg.mkPen("#0f766e", width=1.25),
            connect="finite",
        )

        self.overview_plot = pg.PlotWidget(axisItems={"bottom": TimeAxisItem(orientation="bottom")})
        self.overview_plot.setObjectName("OverviewPlot")
        self.overview_plot.setLabel("left", "Overview")
        self.overview_plot.setLabel("bottom", "Elapsed time")
        self.overview_plot.getAxis("left").enableAutoSIPrefix(False)
        self.overview_plot.showGrid(x=True, y=True, alpha=0.18)
        self.overview_plot.setMouseEnabled(x=False, y=False)
        self.overview_plot.setMenuEnabled(False)
        self.overview_plot.hideButtons()
        self.overview_curve = self.overview_plot.plot(
            [],
            [],
            pen=pg.mkPen("#64748b", width=0.9),
            connect="finite",
        )
        self.region = pg.LinearRegionItem(
            values=[0, DEFAULT_WINDOW_SECONDS],
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(15, 118, 110, 44),
            movable=True,
        )
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self.region_changed)
        self.overview_plot.addItem(self.region)

        self.overview_center_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#f59e0b", width=2),
        )
        self.overview_center_line.setZValue(11)
        self.overview_plot.addItem(self.overview_center_line, ignoreBounds=True)

        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#334155", width=1))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#334155", width=1))
        self.detail_plot.addItem(self.crosshair_v, ignoreBounds=True)
        self.detail_plot.addItem(self.crosshair_h, ignoreBounds=True)
        self.set_crosshair_visible()

        self.detail_plot.getViewBox().sigRangeChanged.connect(self.detail_range_changed)
        self.detail_plot.scene().sigMouseMoved.connect(self.mouse_moved)
        self.overview_plot.scene().sigMouseClicked.connect(self.overview_clicked)

        self.plot_splitter.addWidget(self.detail_plot)
        self.plot_splitter.addWidget(self.overview_plot)
        self.plot_splitter.setSizes([680, 190])

        self.statusBar().showMessage("Ready")

    def _build_preprocessing_panel(self, root: QtWidgets.QVBoxLayout) -> None:
        panel = QtWidgets.QFrame()
        panel.setObjectName("Panel")
        layout = QtWidgets.QGridLayout(panel)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)
        root.addWidget(panel)

        title = QtWidgets.QLabel("Preprocessing")
        title.setObjectName("SectionTitle")
        layout.addWidget(title, 0, 0)

        self.bandpass_check = QtWidgets.QCheckBox("Bandpass")
        self.bandpass_check.stateChanged.connect(self.preprocessing_controls_changed)
        layout.addWidget(self.bandpass_check, 0, 1)

        self.low_cut_spin = QtWidgets.QDoubleSpinBox()
        self.low_cut_spin.setDecimals(2)
        self.low_cut_spin.setRange(0.01, 1000.0)
        self.low_cut_spin.setValue(5.0)
        self.low_cut_spin.setSuffix(" Hz")
        self.low_cut_spin.setKeyboardTracking(False)
        self.low_cut_spin.valueChanged.connect(self.preprocessing_controls_changed)

        self.high_cut_spin = QtWidgets.QDoubleSpinBox()
        self.high_cut_spin.setDecimals(2)
        self.high_cut_spin.setRange(0.02, 1000.0)
        self.high_cut_spin.setValue(40.0)
        self.high_cut_spin.setSuffix(" Hz")
        self.high_cut_spin.setKeyboardTracking(False)
        self.high_cut_spin.valueChanged.connect(self.preprocessing_controls_changed)

        self.filter_order_spin = QtWidgets.QSpinBox()
        self.filter_order_spin.setRange(1, 10)
        self.filter_order_spin.setValue(4)
        self.filter_order_spin.valueChanged.connect(self.preprocessing_controls_changed)

        layout.addWidget(QtWidgets.QLabel("Low"), 0, 2)
        layout.addWidget(self.low_cut_spin, 0, 3)
        layout.addWidget(QtWidgets.QLabel("High"), 0, 4)
        layout.addWidget(self.high_cut_spin, 0, 5)
        layout.addWidget(QtWidgets.QLabel("Order"), 0, 6)
        layout.addWidget(self.filter_order_spin, 0, 7)

        self.zcr_check = QtWidgets.QCheckBox("Zero crossing noise")
        self.zcr_check.stateChanged.connect(self.preprocessing_controls_changed)
        layout.addWidget(self.zcr_check, 1, 1)

        self.zcr_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.zcr_cutoff_spin.setDecimals(2)
        self.zcr_cutoff_spin.setRange(0.01, 500.0)
        self.zcr_cutoff_spin.setValue(30.0)
        self.zcr_cutoff_spin.setSuffix(" /s")
        self.zcr_cutoff_spin.setKeyboardTracking(False)
        self.zcr_cutoff_spin.valueChanged.connect(self.preprocessing_controls_changed)
        layout.addWidget(QtWidgets.QLabel("2 s windows, cutoff"), 1, 2)
        layout.addWidget(self.zcr_cutoff_spin, 1, 3)

        self.motion_check = QtWidgets.QCheckBox("Motion variance")
        self.motion_check.stateChanged.connect(self.preprocessing_controls_changed)
        layout.addWidget(self.motion_check, 1, 4)

        self.motion_window_spin = QtWidgets.QDoubleSpinBox()
        self.motion_window_spin.setDecimals(2)
        self.motion_window_spin.setRange(0.10, 30.0)
        self.motion_window_spin.setValue(2.00)
        self.motion_window_spin.setSuffix(" s")
        self.motion_window_spin.setKeyboardTracking(False)
        self.motion_window_spin.valueChanged.connect(self.preprocessing_controls_changed)

        self.motion_std_spin = QtWidgets.QDoubleSpinBox()
        self.motion_std_spin.setDecimals(3)
        self.motion_std_spin.setRange(0.001, 1000.0)
        self.motion_std_spin.setValue(1.000)
        self.motion_std_spin.setSuffix(" mV")
        self.motion_std_spin.setKeyboardTracking(False)
        self.motion_std_spin.valueChanged.connect(self.preprocessing_controls_changed)

        self.motion_ptp_spin = QtWidgets.QDoubleSpinBox()
        self.motion_ptp_spin.setDecimals(3)
        self.motion_ptp_spin.setRange(0.001, 1000.0)
        self.motion_ptp_spin.setValue(3.000)
        self.motion_ptp_spin.setSuffix(" mV")
        self.motion_ptp_spin.setKeyboardTracking(False)
        self.motion_ptp_spin.valueChanged.connect(self.preprocessing_controls_changed)

        self.motion_step_spin = QtWidgets.QDoubleSpinBox()
        self.motion_step_spin.setDecimals(2)
        self.motion_step_spin.setRange(0.05, 10.0)
        self.motion_step_spin.setValue(0.50)
        self.motion_step_spin.setSuffix(" s")
        self.motion_step_spin.setKeyboardTracking(False)
        self.motion_step_spin.valueChanged.connect(self.preprocessing_controls_changed)

        layout.addWidget(QtWidgets.QLabel("Window"), 1, 5)
        layout.addWidget(self.motion_window_spin, 1, 6)
        layout.addWidget(QtWidgets.QLabel("Step"), 1, 7)
        layout.addWidget(self.motion_step_spin, 1, 8)
        layout.addWidget(QtWidgets.QLabel("STD >"), 2, 5)
        layout.addWidget(self.motion_std_spin, 2, 6)
        layout.addWidget(QtWidgets.QLabel("P99-P1 >"), 2, 7)
        layout.addWidget(self.motion_ptp_spin, 2, 8)

        self.rgap_check = QtWidgets.QCheckBox("Long R-gap")
        self.rgap_check.stateChanged.connect(self.preprocessing_controls_changed)
        layout.addWidget(self.rgap_check, 3, 1)

        self.max_r_peak_spin = QtWidgets.QDoubleSpinBox()
        self.max_r_peak_spin.setDecimals(3)
        self.max_r_peak_spin.setRange(0.001, 1000.0)
        self.max_r_peak_spin.setValue(0.550)
        self.max_r_peak_spin.setSuffix(" mV")
        self.max_r_peak_spin.setKeyboardTracking(False)
        self.max_r_peak_spin.valueChanged.connect(self.preprocessing_controls_changed)

        self.max_rr_spin = QtWidgets.QDoubleSpinBox()
        self.max_rr_spin.setDecimals(2)
        self.max_rr_spin.setRange(0.10, 10.0)
        self.max_rr_spin.setValue(1.50)
        self.max_rr_spin.setSuffix(" s")
        self.max_rr_spin.setKeyboardTracking(False)
        self.max_rr_spin.valueChanged.connect(self.preprocessing_controls_changed)

        layout.addWidget(QtWidgets.QLabel("Max R peak"), 3, 2)
        layout.addWidget(self.max_r_peak_spin, 3, 3)
        layout.addWidget(QtWidgets.QLabel("Gap >"), 3, 4)
        layout.addWidget(self.max_rr_spin, 3, 5)

        self.preprocessing_label = QtWidgets.QLabel("Preprocessing off. Raw ECG is displayed.")
        self.preprocessing_label.setObjectName("Subtle")
        self.preprocessing_label.setWordWrap(True)
        self.preprocessing_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.preprocessing_label, 4, 1, 1, 10)

        self._update_preprocessing_dependency_state()

    def _apply_style(self) -> None:
        self.setStyleSheet(APP_STYLESHEET)

    def preprocessing_controls_changed(self) -> None:
        if self._updating_preprocessing_controls:
            return
        self._update_preprocessing_dependency_state()
        self.apply_current_preprocessing()

    def _update_preprocessing_dependency_state(self) -> None:
        bandpass_enabled = self.bandpass_check.isChecked()
        dependent_widgets = (
            self.zcr_check,
            self.zcr_cutoff_spin,
            self.motion_check,
            self.motion_window_spin,
            self.motion_std_spin,
            self.motion_ptp_spin,
            self.motion_step_spin,
            self.rgap_check,
            self.max_r_peak_spin,
            self.max_rr_spin,
        )
        self._updating_preprocessing_controls = True
        if not bandpass_enabled:
            self.zcr_check.setChecked(False)
            self.motion_check.setChecked(False)
            self.rgap_check.setChecked(False)
        for widget in dependent_widgets:
            widget.setEnabled(bandpass_enabled)
        self._updating_preprocessing_controls = False

    def _configure_filter_ranges(self) -> None:
        if not self.data:
            return
        nyquist = self.data.sample_rate_hz / 2.0
        high_max = max(0.02, nyquist - 0.001)
        self.low_cut_spin.blockSignals(True)
        self.high_cut_spin.blockSignals(True)
        self.low_cut_spin.setRange(0.01, high_max)
        self.high_cut_spin.setRange(0.02, high_max)
        if self.high_cut_spin.value() >= high_max:
            self.high_cut_spin.setValue(high_max)
        if self.low_cut_spin.value() >= self.high_cut_spin.value():
            self.low_cut_spin.setValue(max(0.01, self.high_cut_spin.value() / 2.0))
        self.low_cut_spin.blockSignals(False)
        self.high_cut_spin.blockSignals(False)

    def current_preprocessing_settings(self) -> PreprocessingSettings:
        bandpass_enabled = self.bandpass_check.isChecked()
        return PreprocessingSettings(
            apply_bandpass=bandpass_enabled,
            low_cut_hz=self.low_cut_spin.value(),
            high_cut_hz=self.high_cut_spin.value(),
            filter_order=self.filter_order_spin.value(),
            apply_zcr=bandpass_enabled and self.zcr_check.isChecked(),
            zcr_cutoff_hz=self.zcr_cutoff_spin.value(),
            apply_motion=bandpass_enabled and self.motion_check.isChecked(),
            motion_window_sec=self.motion_window_spin.value(),
            motion_step_sec=self.motion_step_spin.value(),
            motion_std_threshold_mv=self.motion_std_spin.value(),
            motion_ptp_threshold_mv=self.motion_ptp_spin.value(),
            apply_rgap=bandpass_enabled and self.rgap_check.isChecked(),
            max_r_peak_height=self.max_r_peak_spin.value(),
            max_rr_sec=self.max_rr_spin.value(),
        )

    def apply_current_preprocessing(self) -> None:
        if not self.data:
            return
        settings = self.current_preprocessing_settings()
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        QtWidgets.QApplication.processEvents()
        try:
            result = apply_preprocessing(
                self.data.ecg_mv,
                fs=self.data.sample_rate_hz,
                settings=settings,
            )
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.preprocessing_label.setText(f"Preprocessing error: {exc}")
            self.statusBar().showMessage("Preprocessing failed")
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        self.preprocessing_result = result
        self.set_display_waveform(result.display_ecg, processed=settings.apply_bandpass)
        self.preprocessing_label.setText(self.preprocessing_summary(settings, result))
        self.refresh_detail_plot()

    def set_display_waveform(self, waveform: np.ndarray, *, processed: bool) -> None:
        if not self.data:
            return
        self.display_ecg = np.asarray(waveform, dtype=float)
        detail_color = "#2563eb" if processed else "#0f766e"
        overview_color = "#7c3aed" if processed else "#64748b"
        self.detail_curve.setPen(pg.mkPen(detail_color, width=1.25))
        self.overview_curve.setPen(pg.mkPen(overview_color, width=0.9))

        overview_x, overview_y = downsample_envelope(
            self.data.time_s,
            self.display_ecg,
            MAX_OVERVIEW_POINTS,
        )
        self.overview_curve.setData(overview_x, overview_y)
        self.overview_plot.setXRange(self.data.start_s, self.data.end_s, padding=0.0)
        self.fit_overview_y()

    def preprocessing_summary(
        self,
        settings: PreprocessingSettings,
        result: PreprocessingResult,
    ) -> str:
        if not settings.apply_bandpass:
            total_sec = self.data.sample_count / self.data.sample_rate_hz if self.data else 0.0
            return (
                "Preprocessing off. Raw ECG is displayed. "
                f"Initial available signal: {total_sec:.2f} s. "
                "Masked: 0.00 s. Usable: 100.00%."
            )

        parts = [
            (
                f"Bandpass {settings.low_cut_hz:.2f}-{settings.high_cut_hz:.2f} Hz, "
                f"Butterworth order {settings.filter_order}."
            )
        ]
        fs = self.data.sample_rate_hz if self.data else 0
        zcr_sec, motion_sec, rgap_sec, combined_sec = result.masked_seconds(fs)
        if settings.apply_zcr:
            flagged_windows = sum(1 for _, _, _, flagged in result.zcr_records if flagged)
            parts.append(
                f"ZCR masked {zcr_sec:.2f} s across {flagged_windows} full 2 s windows "
                f"(cutoff > {settings.zcr_cutoff_hz:.2f}/s)."
            )
            if result.trailing_zcr_samples:
                parts.append(f"ZCR trailing samples unassessed: {result.trailing_zcr_samples}.")
        if settings.apply_motion:
            flagged_windows = sum(1 for _, _, _, _, flagged in result.motion_records if flagged)
            parts.append(
                f"Motion variance masked {motion_sec:.2f} s across {flagged_windows} sliding windows "
                f"({settings.motion_window_sec:.2f} s window, {settings.motion_step_sec:.2f} s step, "
                f"STD > {settings.motion_std_threshold_mv:.3f} mV or "
                f"P99-P1 > {settings.motion_ptp_threshold_mv:.3f} mV)."
            )
            if result.trailing_motion_samples:
                parts.append(f"Motion trailing samples unassessed: {result.trailing_motion_samples}.")
        if settings.apply_rgap:
            parts.append(
                f"R-gap masked {rgap_sec:.2f} s from {len(result.rgap_intervals)} gaps "
                f"(peak threshold {settings.r_peak_height_scale:.2f} x {settings.max_r_peak_height:.3f} mV, "
                f"gap > {settings.max_rr_sec:.2f} s)."
            )
            parts.append("Long gaps are suspected artifacts or missed beats, not proof of noisy signal.")
        if settings.apply_zcr or settings.apply_motion or settings.apply_rgap:
            parts.append(
                f"Combined mask {combined_sec:.2f} s ({result.combined_percent():.2f}%), "
                "with overlaps counted once."
            )
        total_sec = result.total_seconds(fs)
        usable_sec = result.usable_seconds(fs)
        parts.append(
            f"Signal availability: total {total_sec:.2f} s, masked {combined_sec:.2f} s, "
            f"usable {usable_sec:.2f} s ({result.usable_percent():.2f}%)."
        )
        return " ".join(parts)

    def choose_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Movesense CSV",
            str(Path.cwd()),
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            self.load_csv(Path(path))

    def load_csv(self, path: Path) -> None:
        self.statusBar().showMessage(f"Loading {path.name} ...")
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        QtWidgets.QApplication.processEvents()

        try:
            data = read_movesense_csv(path)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Could not load CSV", str(exc))
            self.statusBar().showMessage("Load failed")
            return

        QtWidgets.QApplication.restoreOverrideCursor()
        self.data = data
        self.display_ecg = data.ecg_mv.astype(float, copy=True)
        self.preprocessing_result = None
        self.file_label.setText(str(data.path))
        self.export_image_button.setEnabled(True)
        self._configure_filter_ranges()

        self.start_spin.blockSignals(True)
        self.window_spin.blockSignals(True)
        self.position_slider.blockSignals(True)
        self.start_spin.setRange(data.start_s, data.end_s)
        self.window_spin.setRange(0.050, max(0.050, data.duration_s))
        self.window_spin.setValue(min(DEFAULT_WINDOW_SECONDS, max(0.050, data.duration_s)))
        self.start_spin.setValue(data.start_s)
        self.position_slider.setValue(0)
        self.start_spin.blockSignals(False)
        self.window_spin.blockSignals(False)
        self.position_slider.blockSignals(False)

        self.apply_current_preprocessing()
        self.set_region(data.start_s, min(data.start_s + self.window_spin.value(), data.end_s))
        self.set_detail_range(data.start_s, min(data.start_s + self.window_spin.value(), data.end_s))
        self.statusBar().showMessage(
            f"Loaded {data.sample_count:,} samples from {data.path.name}"
        )

    def start_changed(self, value: float) -> None:
        if not self.data:
            return
        start, end = self.clamp_window(value, value + self.window_spin.value())
        self.set_region(start, end)
        self.set_detail_range(start, end)

    def window_changed(self, value: float) -> None:
        if not self.data:
            return
        start, end = self.current_window()
        center = (start + end) / 2.0
        new_start = center - value / 2.0
        new_end = center + value / 2.0
        start, end = self.clamp_window(new_start, new_end)
        self.set_region(start, end)
        self.set_detail_range(start, end)

    def slider_changed(self, value: int) -> None:
        if not self.data:
            return
        width = self.window_spin.value()
        start = self.data.start_s + (value / 10000.0) * max(0.0, self.data.duration_s - width)
        start, end = self.clamp_window(start, start + width)
        self.set_region(start, end)
        self.set_detail_range(start, end)

    def region_changed(self) -> None:
        if self._updating_region or not self.data:
            return
        start, end = self.clamp_window(*self.region.getRegion())
        self.set_controls_from_window(start, end)
        self.set_detail_range(start, end)

    def detail_range_changed(self, view_box: pg.ViewBox, view_range: list[list[float]]) -> None:
        if self._updating_range or not self.data:
            return
        start, end = view_range[0]
        start, end = self.clamp_window(start, end)
        self.set_controls_from_window(start, end)
        self.set_region(start, end)
        self.refresh_detail_plot()

    def overview_clicked(self, event: object) -> None:
        if not self.data:
            return
        if not hasattr(event, "button") or event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        scene_pos = event.scenePos()
        if not self.overview_plot.sceneBoundingRect().contains(scene_pos):
            return
        plot_point = self.overview_plot.plotItem.vb.mapSceneToView(scene_pos)
        width = self.window_spin.value()
        start, end = self.clamp_window(plot_point.x() - width / 2.0, plot_point.x() + width / 2.0)
        self.set_region(start, end)
        self.set_detail_range(start, end)

    def mouse_moved(self, scene_pos: QtCore.QPointF) -> None:
        if not self.data or not self.crosshair_check.isChecked():
            return
        if not self.detail_plot.sceneBoundingRect().contains(scene_pos):
            return
        point = self.detail_plot.plotItem.vb.mapSceneToView(scene_pos)
        x = float(point.x())
        y = float(point.y())
        self._last_mouse_x = x
        self.crosshair_v.setPos(x)
        self.crosshair_h.setPos(y)
        nearest = self.nearest_sample(x)
        if nearest is None:
            self.cursor_label.setText("Cursor: n.a.")
            return
        sample_time, sample_value = nearest
        self.cursor_label.setText(f"Cursor: {sample_time:.3f} s, {sample_value:.4f} mV")

    def set_crosshair_visible(self) -> None:
        visible = self.crosshair_check.isChecked()
        self.crosshair_v.setVisible(visible)
        self.crosshair_h.setVisible(visible)
        if not visible:
            self.cursor_label.setText("Cursor: n.a.")

    def set_window_seconds(self, seconds: float) -> None:
        self.window_spin.setValue(seconds)

    def shift_window(self, fraction: float) -> None:
        if not self.data:
            return
        start, end = self.current_window()
        width = end - start
        shifted_start = start + width * fraction
        shifted_end = end + width * fraction
        start, end = self.clamp_window(shifted_start, shifted_end)
        self.set_region(start, end)
        self.set_detail_range(start, end)

    def reset_view(self) -> None:
        if not self.data:
            return
        start = self.data.start_s
        end = min(self.data.end_s, start + min(DEFAULT_WINDOW_SECONDS, self.data.duration_s))
        self.window_spin.blockSignals(True)
        self.window_spin.setValue(end - start)
        self.window_spin.blockSignals(False)
        self.set_region(start, end)
        self.set_detail_range(start, end)
        self.fit_overview_y()

    def fit_y_to_visible(self) -> None:
        if self.visible_ecg.size == 0:
            return
        low, high = robust_y_limits(self.visible_ecg)
        was_updating = self._updating_range
        self._updating_range = True
        self.detail_plot.setYRange(low, high, padding=0.0)
        self._updating_range = was_updating

    def fit_overview_y(self) -> None:
        if not self.data or self.display_ecg.size == 0:
            return
        low, high = robust_y_limits(self.display_ecg)
        self.overview_plot.setYRange(low, high, padding=0.0)

    def set_region(self, start: float, end: float) -> None:
        self._updating_region = True
        self.region.setRegion((start, end))
        self.overview_center_line.setPos((start + end) / 2.0)
        self._updating_region = False

    def set_detail_range(self, start: float, end: float) -> None:
        self._updating_range = True
        self.detail_plot.setXRange(start, end, padding=0.0)
        self._updating_range = False
        self.refresh_detail_plot()

    def refresh_detail_plot(self) -> None:
        if not self.data:
            return
        start, end = self.current_window()
        left = np.searchsorted(self.data.time_s, start, side="left")
        right = np.searchsorted(self.data.time_s, end, side="right")
        self.visible_time = self.data.time_s[left:right]
        self.visible_ecg = self.display_ecg[left:right]

        plot_time, plot_ecg = downsample_envelope(
            self.visible_time,
            self.visible_ecg,
            MAX_DETAIL_POINTS,
        )
        self.detail_curve.setData(plot_time, plot_ecg)

        if self.auto_y_check.isChecked():
            self.fit_y_to_visible()

        if self._last_mouse_x is not None:
            nearest = self.nearest_sample(self._last_mouse_x)
            if nearest is not None:
                self.cursor_label.setText(f"Cursor: {nearest[0]:.3f} s, {nearest[1]:.4f} mV")
        self.update_stats(start, end)

    def update_stats(self, start: float, end: float) -> None:
        if not self.data:
            return
        segment = self.visible_ecg
        duration_min = self.data.duration_s / 60.0
        if segment.size:
            peak_to_peak = float(np.max(segment) - np.min(segment))
            text = (
                f"Samples: {self.data.sample_count:,} | "
                f"Duration: {duration_min:.2f} min | "
                f"Rate: {self.data.sample_rate_hz:.1f} Hz | "
                f"Window: {start:.3f}-{end:.3f} s ({segment.size:,} samples) | "
                f"Mean: {np.mean(segment):.4f} mV | "
                f"Min/Max: {np.min(segment):.4f}/{np.max(segment):.4f} mV | "
                f"Peak-to-peak: {peak_to_peak:.4f} mV"
            )
        else:
            text = "No samples in the selected window."

        meta_bits = []
        for key in ("created", "device", "serial", "bandwidth", "age", "gender"):
            if key in self.data.metadata:
                meta_bits.append(f"{key}: {self.data.metadata[key]}")
        if meta_bits:
            text = f"{text}\nMetadata: " + " | ".join(meta_bits)
        if self.preprocessing_result is not None:
            text = f"{text}\nDisplay: {self.preprocessing_result.message}"
            total_sec = self.preprocessing_result.total_seconds(self.data.sample_rate_hz)
            masked_sec = self.preprocessing_result.masked_seconds(self.data.sample_rate_hz)[-1]
            usable_sec = self.preprocessing_result.usable_seconds(self.data.sample_rate_hz)
            text = (
                f"{text}\nSignal availability: total {total_sec:.2f} s | "
                f"masked {masked_sec:.2f} s | usable {usable_sec:.2f} s "
                f"({self.preprocessing_result.usable_percent():.2f}%)"
            )
        self.stats_label.setText(text)

    def set_controls_from_window(self, start: float, end: float) -> None:
        if not self.data:
            return
        width = max(0.001, end - start)
        self.start_spin.blockSignals(True)
        self.window_spin.blockSignals(True)
        self.position_slider.blockSignals(True)
        self.start_spin.setValue(start)
        self.window_spin.setValue(width)
        denominator = max(0.001, self.data.duration_s - width)
        slider_value = int(round(((start - self.data.start_s) / denominator) * 10000))
        self.position_slider.setValue(max(0, min(10000, slider_value)))
        self.start_spin.blockSignals(False)
        self.window_spin.blockSignals(False)
        self.position_slider.blockSignals(False)

    def current_window(self) -> tuple[float, float]:
        if not self.data:
            return 0.0, 0.0
        view_range = self.detail_plot.getViewBox().viewRange()[0]
        start, end = view_range[0], view_range[1]
        if end <= start:
            start = self.start_spin.value()
            end = start + self.window_spin.value()
        return self.clamp_window(start, end)

    def clamp_window(self, start: float, end: float) -> tuple[float, float]:
        if not self.data:
            return start, end
        width = max(0.001, end - start)
        width = min(width, max(0.001, self.data.duration_s))
        start = max(self.data.start_s, min(start, self.data.end_s - width))
        end = start + width
        return float(start), float(end)

    def nearest_sample(self, time_s: float) -> tuple[float, float] | None:
        if not self.data or self.data.sample_count == 0:
            return None
        idx = int(np.searchsorted(self.data.time_s, time_s))
        idx = max(0, min(self.data.sample_count - 1, idx))
        waveform = self.display_ecg if self.display_ecg.size == self.data.sample_count else self.data.ecg_mv
        return float(self.data.time_s[idx]), float(waveform[idx])

    def export_current_image(self) -> None:
        if not self.data:
            return
        start, end = self.current_window()
        default_name = self.data.path.with_name(
            f"{self.data.path.stem}_{start:.3f}s_to_{end:.3f}s.png"
        )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export current ECG image",
            str(default_name),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg)",
        )
        if not path:
            return

        output_path = Path(path)
        if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            output_path = output_path.with_suffix(".png")

        QtWidgets.QApplication.processEvents()
        pixmap = self.plot_splitter.grab()
        if pixmap.isNull() or not pixmap.save(str(output_path)):
            QtWidgets.QMessageBox.critical(
                self,
                "Image export failed",
                f"Could not save image to {output_path}",
            )
            return

        self.statusBar().showMessage(f"Exported image to {output_path.name}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Movesense ECG CSV data.")
    parser.add_argument(
        "csv",
        nargs="?",
        default=DEFAULT_CSV,
        help="Path to the Movesense CSV file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    app = QtWidgets.QApplication(sys.argv)
    viewer = EcgViewer(csv_path if csv_path.exists() else None)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
