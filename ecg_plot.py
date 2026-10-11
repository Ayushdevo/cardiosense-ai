"""Render each ECG plot with its own figure and detached image buffer."""
from io import BytesIO

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image


def render_ecg_plot(signal, peaks, heart_rate):
    figure = Figure(figsize=(8, 3))
    axis = figure.subplots()
    window = signal[:500]
    axis.plot(window, color="royalblue", label="ECG Signal")
    visible_peaks = [peak for peak in peaks if 0 <= peak < len(window)]
    if visible_peaks:
        axis.plot(visible_peaks, [signal[peak] for peak in visible_peaks], "ro", markersize=5)
    bpm_label = f"{heart_rate} BPM" if heart_rate else "-- BPM"
    axis.set_title(f"ECG Signal ({bpm_label})", fontsize=13)
    axis.set_xlabel("Sample")
    axis.set_ylabel("Amplitude")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    try:
        with BytesIO() as output:
            FigureCanvasAgg(figure).print_png(output)
            output.seek(0)
            with Image.open(output) as image:
                return image.copy()
    finally:
        figure.clear()
