"""Plot formatting helpers for the ECG viewer."""

from __future__ import annotations

import pyqtgraph as pg


def configure_pyqtgraph() -> None:
    pg.setConfigOptions(antialias=True, background="w", foreground="#334155")


class TimeAxisItem(pg.AxisItem):
    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        return [format_seconds(value) for value in values]


def format_seconds(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value < 60:
        return f"{sign}{value:.3g}s"
    minutes, seconds = divmod(value, 60.0)
    if minutes < 60:
        return f"{sign}{int(minutes):02d}:{seconds:04.1f}"
    hours, minutes = divmod(minutes, 60.0)
    return f"{sign}{int(hours):02d}:{int(minutes):02d}:{seconds:04.1f}"
