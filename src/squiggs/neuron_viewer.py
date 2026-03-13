"""
neuron_viewer.py

The neuron viewer handles the logic for generating
sliding plots across units.

Author: Stellina X. Ao
Created: 2026-02-26
Last Modified: 2026-02-27
Python Version: >= 3.10.4
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# from PyQt6.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout
#     QSlider, QLineEdit
# )
# from PyQt6.QtCore import Qt
# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class NeuronViewer:
    def __init__(
        self,
        num_units,
        render_func,
        ymin=None,
        ymax=None,
        ncols=1,
        nrows=1,
        title="Neuron Viewer",
    ):
        plt.close("all")
        self.num_units = num_units
        self.render_func = render_func

        if ncols == 1 and nrows == 1:
            self.fig, self.axes = plt.subplots(figsize=(2, 2))
            self.axes = [self.axes]
        else:
            self.fig, self.axes = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                figsize=(2.5 * ncols, 2.5 * nrows),
                sharey=True,
            )

        self.fig.subplots_adjust(
            left=0.2,
            right=0.9,
            top=0.8,
            bottom=0.2,  # leave space for slider
            hspace=0.4,  # vertical spacing between rows
            wspace=0.3,  # horizontal spacing between columns
        )

        plt.subplots_adjust(bottom=0.3)

        self.current_idx = 0
        self.render_func(self.current_idx, self.fig, self.axes)

        # slider axis
        slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(
            slider_ax, "Unit", 0, self.num_units - 1, valinit=0, valstep=1
        )

        self.slider.on_changed(self.update)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def update(self, val):
        idx = int(self.slider.val)
        self.render_func(idx, self.fig, self.axes)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        step = 1
        if event.key == "right":
            idx = self.slider.val + step
            if idx > self.num_units - 1:
                return
            self.slider.set_val(idx)
        elif event.key == "left":
            idx = self.slider.val - step
            self.slider.set_val(idx)
