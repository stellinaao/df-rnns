"""
renderers.py

Renderer objects handle a plotting functionality for 
a single neuron and are designed to be used in conjunction
with a NeuronViewer() object.

Author: Stellina X. Ao
Created: 2026-02-26
Last Modified: 2026-02-27
Python Version: >= 3.10.4
"""

import numpy as np
from scipy.stats import sem
from DAMN.damn.alignment import construct_timebins

class PETHRenderer:
    def __init__(self, peth=None, pres=1, posts=2, binwidth_s=0.1, peth_a=None, peth_b=None, color="#261B49", color_a="#29723E", color_b="#9F5DBC", mode="grand", label_a="", label_b="", do_sem=True, relim=True):
        """
        Parameters
        ----------
        mode : "grand" or "cond"
            grand -> single mean/std
            cond  -> separate a/b condition mean/std
            
        Example
        ----------
        >> renderer_grand = PETHRenderer(peth, pres, posts, binwidth_s, mode="grand")
        >> viewer1 = NeuronViewer(num_units=peth.shape[0], render_func=renderer_grand, ymin=renderer_grand.ymin, ymax=renderer_grand.ymax)


        >> renderer_cond = PETHRenderer(
            peth_a=peth_l,
            peth_b=peth_r,
            mode="cond",
            label_a="left",
            label_b="right"
        )
        >> viewer2 = NeuronViewer(num_units=peth.shape[0], render_func=renderer_cond, ymin=renderer_cond.ymin, ymax=renderer_cond.ymax)
        """
        
        self.mode=mode
        
        if self.mode=="grand":
            self.peth = peth
            
            self.all_means = self.peth.mean(axis=1)
            self.all_stds  = sem(self.peth, axis=1) if do_sem else self.peth.std(axis=1)
            
            self.ymin = np.min(self.all_means - self.all_stds, axis=1)
            self.ymax = np.max(self.all_means + self.all_stds, axis=1)
            
            self.color = color
        
        elif self.mode=="cond":
            self.peth_a = peth_a
            self.peth_b = peth_b
            
            self.label_a = label_a
            self.label_b = label_b
            
            self.all_means_a = self.peth_a.mean(axis=1)
            self.all_means_b = self.peth_b.mean(axis=1)
            self.all_stds_a  = sem(self.peth_a, axis=1) if do_sem else self.peth_a.std(axis=1)
            self.all_stds_b  = sem(self.peth_b, axis=1) if do_sem else self.peth_b.std(axis=1)
            
            self.ymin = np.min((np.min(self.all_means_a - self.all_stds_a, axis=1), np.min(self.all_means_b - self.all_stds_b, axis=1)), axis=0)
            self.ymax = np.max((np.max(self.all_means_a + self.all_stds_a, axis=1), np.max(self.all_means_b + self.all_stds_b, axis=1)), axis=0)
            
            self.color_a = color_a
            self.color_b = color_b
         
        else:
            raise NotImplementedError("Valid modes are 'grand' and 'cond.'")
        
        self.relim = relim
        if not self.relim:
            self.ymin_g = np.min(self.ymin)
            self.ymax_g = np.max(self.ymax)
            padding = 0.05 * (self.ymax_g-self.ymin_g)
            self.ymin_g -= padding
            self.ymax_g += padding
            
        self.times, _, _ = construct_timebins(pres, posts, binwidth_s)
        self.times = np.arange(peth.shape[2])

    def __call__(self, idx, fig, axes):
        ax = axes[0]
        ax.clear()
        if self.mode == "grand":
            mean = self.all_means[idx]
            std  = self.all_stds[idx]
            print("BEAUTEOUS")
            ax.plot(self.times, mean, color=self.color)
            ax.fill_between(self.times, mean-std, mean+std, alpha=0.3, color=self.color)
            
            ax.axvline(x=0, color="#666666", linewidth=0.5, linestyle="--")
        elif self.mode == "cond":
            mean_a = self.all_means_a[idx]
            std_a  = self.all_stds_a[idx]
            mean_b = self.all_means_b[idx]
            std_b  = self.all_stds_b[idx]

            ax.plot(self.times, mean_a, color=self.color_a, label=self.label_a)
            ax.plot(self.times, mean_b, color=self.color_b, label=self.label_b)
            ax.fill_between(self.times, mean_a-std_a, mean_a+std_a, alpha=0.3, color=self.color_a)
            ax.fill_between(self.times, mean_b-std_b, mean_b+std_b, alpha=0.3, color=self.color_b)
            
            ax.axvline(x=0, color="#666666", linewidth=0.5, linestyle="--")
            ax.legend()
        else:
            raise ValueError("Mode must be 'grand' or 'cond'.")


        if self.relim:
            padding = 0.05*(self.ymax[idx]-self.ymin[idx])
            ax.set_ylim(self.ymin[idx]-padding, self.ymax[idx]+padding)
        else:
            ax.set_ylim(self.ymin_g, self.ymax_g)
            
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_title(f"Unit {idx}")
        
class FitRenderer:
    def __init__(self, model=None, x=None, y=None):
        from scipy.stats import pearsonr as r
        self.model = model
        self.x = x
        self.y = y
        self.yhat = self.model(self.x).detach().numpy()
        self.rsquared = r(self.y, self.yhat, axis=0).statistic**2
        
        self.cids = self.model.cids
    
    def __call__(self, idx, fig, axes):
        for ax in axes: 
            ax.clear()
            
        ax = axes[0]
        ax.plot(self.y[:,self.cids][:,idx], color="#666666", alpha=0.5, label="observed")
        ax.plot(self.yhat[:,idx], color="#5C2392", alpha=0.5, label="predicted")
        
        # ax.legend()
        ax.set_xlabel("Trials")
        ax.set_ylabel("Spike Counts")
        ax.set_title(f"$r^2$={self.rsquared[idx]:.3f}")
class KernelRenderer:
    def __init__(self, model=None, dmat=None, bias=None):
        """
        Parameters
        ----------
        mode : "grand" or "cond"
            grand -> single mean/std
            cond  -> separate a/b condition mean/std
            
        Example
        ----------
        >> renderer_grand = PETHRenderer(peth, pres, posts, binwidth_s, mode="grand")
        >> viewer1 = NeuronViewer(num_units=peth.shape[0], render_func=renderer_grand, ymin=renderer_grand.ymin, ymax=renderer_grand.ymax)


        >> renderer_cond = PETHRenderer(
            peth_a=peth_l,
            peth_b=peth_r,
            mode="cond",
            label_a="left",
            label_b="right"
        )
        >> viewer2 = NeuronViewer(num_units=peth.shape[0], render_func=renderer_cond, ymin=renderer_cond.ymin, ymax=renderer_cond.ymax)
        """
        self.linkfunc = model.estimators_[0]._base_loss.link.inverse
        
        # get the unique tags from dmat
        self.all_tags = []
        for _, reg in dmat.regressors.items():
            self.all_tags.extend(reg.tags)
        self.all_tags = np.unique(self.all_tags)
        self.all_tags = [t for t in self.all_tags if t not in ['task', 'interaction', 'hmm', 'behavior']]
        
        self.model = model
        self.dmat = dmat
        self.bias = bias
        
        self.cache = {}
        ymin = np.inf
        ymax = -np.inf
        
        for tag in self.all_tags:
            self.cache[tag] = {}
            regs = self.dmat.select(tag=tag)
            
            for r, reg in regs.items():
                k_all, t = reg.reconstruct_kernel()
                self.cache[tag][f"{reg}_t"] = t
                self.cache[tag][f"{reg}_k"] = np.zeros((len(bias), t.shape[0]))
                
                for idx in range(len(bias)):
                    k = k_all[:, idx]
                    k = self.linkfunc(k + bias[idx])
                    
                    max_curr = np.max(k)
                    min_curr = np.min(k)
                    
                    if max_curr > ymax:
                        ymax = max_curr
                    if min_curr < ymin:
                        ymin = min_curr

                    self.cache[tag][f"{reg}_k"][idx] = k
        self.ymin = ymin
        self.ymax = ymax

    def __call__(self, idx, fig, axes):
        for ax in axes: 
            ax.clear()
        
        for i, tag in enumerate(self.all_tags):
            regs = self.dmat.select(tag=tag)
            for r, reg in regs.items():
                axes[i].plot(self.cache[tag][f"{reg}_t"], self.cache[tag][f"{reg}_k"][idx], label=reg.name)
            axes[i].axvline(x=0, linewidth=0.5, linestyle="--", color="#333333")
            axes[i].set_title(tag)
            if tag not in ['history', 'dlc', 'video']:
                axes[i].legend()
            axes[i].set_xlabel("Time (s)")
        
        axes[0].set_ylabel("Weight")
        fig.suptitle(f"Unit {idx}")

