import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class NeuronViewer:
    def __init__(self, num_units, render_func, ymin=None, ymax=None, ncols=1, nrows=1, title="Neuron Viewer"):
        plt.close("all")
        self.num_units = num_units
        self.render_func = render_func

        if ncols == 1 and nrows == 1: 
            self.fig, self.axes = plt.subplots()
            self.axes = [self.axes]
        else:
            self.fig, self.axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(2.5*ncols, 2.5*nrows), sharey=True)
        plt.subplots_adjust(bottom=0.25)

        self.current_idx = 0
        self.render_func(self.current_idx, self.fig, self.axes)

        # slider axis
        slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(
            slider_ax,
            "Unit",
            0,
            self.num_units - 1,
            valinit=0,
            valstep=1
        )
        
        self.global_ylim = ymin is not None and ymax is not None
        if self.global_ylim:
            padding = 0.05 * (ymax-ymin)
            self.ymin = ymin - padding
            self.ymax = ymax + padding
            self.axes[0].set_ylim(self.ymin, self.ymax)

        self.slider.on_changed(self.update)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.slider.on_changed(self.update)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def update(self, val):
        idx = int(self.slider.val)
        self.render_func(idx, self.fig, self.axes)
        if self.global_ylim:
            self.axes[0].set_ylim(self.ymin, self.ymax)
        else:
            self.axes[0].relim()
            self.axes[0].autoscale_view()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        step = 1
        if event.key == 'right':
            idx = self.slider.val + step
            if idx > self.num_units-1: 
                return
            self.slider.set_val(idx)
        elif event.key == 'left':
            idx = self.slider.val - step
            self.slider.set_val(idx)
            
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from DAMN.damn.alignment import construct_timebins

class PETHViewer:
    def __init__(self, peth, peth_a, peth_b, trial_data, pres, posts, binwidth_s, label_a="", label_b="", mode="cond"):
        plt.close('all')
        
        self.peth   = peth
        self.peth_a = peth_a
        self.peth_b = peth_b
        
        self.mode = mode
        self.num_units = peth.shape[0]
        self.num_tbins = peth.shape[2]
        
        self.all_means = self.peth.mean(axis=1)
        self.all_stds  = self.peth.std(axis=1)
        
        self.all_means_a = self.peth_a.mean(axis=1)
        self.all_means_b = self.peth_b.mean(axis=1)
        self.all_stds_a  = self.peth_a.std(axis=1)
        self.all_stds_b  = self.peth_b.std(axis=1)
        
        self.times, _, _ = construct_timebins(pres, posts, binwidth_s)

        # create figure
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        # init data (unit 0)
        self.current_idx = 0
        
        if self.mode=="grand":
            mean = self.all_means[self.current_idx]
            std  = self.all_stds[self.current_idx]

            # plot mean
            self.trace_mean, = self.ax.plot(self.times, mean, color="#261B49")

            # plot std
            self.trace_std = self.ax.fill_between(
                self.times,
                mean - std,
                mean + std,
                alpha=0.5,
                color="#5C5EA1"
            )
            
            
            ymin = np.min(self.all_means - self.all_stds)
            ymax = np.max(self.all_means + self.all_stds)
        
        elif self.mode=="cond":
            mean_a = self.all_means_a[self.current_idx]
            mean_b = self.all_means_b[self.current_idx]
            std_a  = self.all_stds_a[self.current_idx]
            std_b  = self.all_stds_b[self.current_idx]
            
            self.label_a = label_a
            self.label_b = label_b
            
            # plot means
            self.trace_mean_a, = self.ax.plot(self.times, mean_a, color="#29723E", label=label_a)
            self.trace_mean_b, = self.ax.plot(self.times, mean_b, color="#672982", label=label_b)
            
            # plot stds
            self.trace_std_a = self.ax.fill_between(
                self.times,
                mean_a - std_a,
                mean_a + std_a,
                alpha=0.5,
                color="#6FCD77"
            )
            
            self.trace_std_b = self.ax.fill_between(
                self.times,
                mean_b - std_b,
                mean_b + std_b,
                alpha=0.5, 
                color="#9F5DBCFF"
            )
            
            self.fig.legend()
            
            ymin = np.min((np.min(self.all_means_a - self.all_stds_a), np.min(self.all_means_b - self.all_stds_b)))
            ymax = np.max((np.max(self.all_means_a + self.all_stds_a), np.max(self.all_means_b + self.all_stds_b)))
            
        else:
            raise NotImplementedError("accepted modes are \'grand\' and \'cond\'")
            
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Firing Rate (Hz)")
        self.ax.set_title(f"PETH, Unit 0")

        # slider axis
        slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(
            slider_ax,
            "Unit",
            0,
            self.num_units - 1,
            valinit=0,
            valstep=1
        )

        padding = 0.05 * (ymax - ymin)
        self.ax.set_ylim(ymin - padding, ymax + padding)

        self.slider.on_changed(self.update)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def update(self, val):
        idx = int(self.slider.val)
        if idx > self.num_units-1:
            return

        if self.mode=="grand":
            mean = self.all_means[idx]
            std  = self.all_stds[idx]

            # update mean line
            self.trace_mean.set_ydata(mean)

            # remove and redraw shaded region
            self.trace_std.remove()
            self.trace_std = self.ax.fill_between(
                np.arange(self.num_tbins),
                mean - std,
                mean + std,
                alpha=0.5,
                color="#5C5EA1"
            )
        elif self.mode=="cond":
            mean_a = self.all_means_a[idx]
            mean_b = self.all_means_b[idx]
            std_a  = self.all_stds_a[idx]
            std_b  = self.all_stds_b[idx]
            
            # plot means
            self.trace_mean_a.set_ydata(mean_a)
            self.trace_mean_b.set_ydata(mean_b)
            
            # plot stds
            self.trace_std_a.remove()
            self.trace_std_b.remove()
            
            self.trace_std_a = self.ax.fill_between(
                np.arange(self.num_tbins),
                mean_a - std_a,
                mean_a + std_a,
                alpha=0.5,
                color="#6FCD77"
            )
            
            self.trace_std_b = self.ax.fill_between(
                np.arange(self.num_tbins),
                mean_b - std_b,
                mean_b + std_b,
                alpha=0.5, 
                color="#9F5DBCFF"
            )
        else:
            raise NotImplementedError("accepted modes are \'grand\' and \'cond\'")

        self.ax.set_title(f"PETH, Unit {idx}")

        self.fig.canvas.draw_idle()

    def on_key(self, event):
        step = 1
        if event.key == 'right':
            idx = self.slider.val + step
            if idx > self.num_units-1: 
                return
            self.slider.set_val(idx)
        elif event.key == 'left':
            idx = self.slider.val - step
            self.slider.set_val(idx)