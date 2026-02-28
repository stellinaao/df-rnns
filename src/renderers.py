import numpy as np
from DAMN.damn.alignment import construct_timebins

class PETHRenderer:
    def __init__(self, peth=None, pres=1, posts=2, binwidth_s=0.1, peth_a=None, peth_b=None, mode="grand", label_a="", label_b=""):
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
            self.all_stds  = self.peth.std(axis=1)
            
            self.ymin = np.min(self.all_means - self.all_stds)
            self.ymax = np.max(self.all_means + self.all_stds)
        
        elif self.mode=="cond":
            self.peth_a = peth_a
            self.peth_b = peth_b
            
            self.label_a = label_a
            self.label_b = label_b
            
            self.all_means_a = self.peth_a.mean(axis=1)
            self.all_means_b = self.peth_b.mean(axis=1)
            self.all_stds_a  = self.peth_a.std(axis=1)
            self.all_stds_b  = self.peth_b.std(axis=1)
            
            self.ymin = np.min((np.min(self.all_means_a - self.all_stds_a), np.min(self.all_means_b - self.all_stds_b)))
            self.ymax = np.max((np.max(self.all_means_a + self.all_stds_a), np.max(self.all_means_b + self.all_stds_b)))
        
        else:
            raise NotImplementedError("Valid modes are 'grand' and 'cond.'")
            
        self.times, _, _ = construct_timebins(pres, posts, binwidth_s)

    def __call__(self, idx, fig, axes):
        ax = axes[0]
        ax.clear()
        if self.mode == "grand":
            mean = self.all_means[idx]
            std  = self.all_stds[idx]
            ax.plot(self.times, mean, color="#261B49")
            ax.fill_between(self.times, mean-std, mean+std, alpha=0.3, color="#5C5EA1")
        elif self.mode == "cond":
            mean_a = self.all_means_a[idx]
            std_a  = self.all_stds_a[idx]
            mean_b = self.all_means_b[idx]
            std_b  = self.all_stds_b[idx]

            ax.plot(self.times, mean_a, color="#29723E", label=self.label_a)
            ax.plot(self.times, mean_b, color="#672982", label=self.label_b)
            ax.fill_between(self.times, mean_a-std_a, mean_a+std_a, alpha=0.3, color="#6FCD77")
            ax.fill_between(self.times, mean_b-std_b, mean_b+std_b, alpha=0.3, color="#9F5DBCFF")
            ax.legend()
        else:
            raise ValueError("Mode must be 'grand' or 'cond'.")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_title(f"Unit {idx}")
        
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
        self.all_tags = [t for t in self.all_tags if t not in ['task', 'interaction', 'hmm']]
        
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

