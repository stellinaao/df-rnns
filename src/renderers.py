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

    def __call__(self, idx, ax):
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
    def __init__(self, model, dmat, bias):

        self.model = model
        self.dmat = dmat
        self.bias = bias
        self.linkfunc = model.estimators_[0]._base_loss.link.inverse
        self.n_units = len(bias)

        # --------------------
        # collect tags
        # --------------------
        all_tags = []
        for _, reg in self.dmat.regressors.items():
            all_tags.extend(reg.tags)

        all_tags = np.unique(all_tags)
        all_tags = [t for t in all_tags if t not in ["task", "interaction"]]
        self.tags = list(all_tags)

        # --------------------
        # cache kernels
        # --------------------
        self.cache = {}

        global_min = np.inf
        global_max = -np.inf

        for tag in self.tags:
            self.cache[tag] = {}
            regs = self.dmat.select(tag=tag)

            for _, reg in regs.items():
                k_raw, t = reg.reconstruct_kernel()
                k = self.linkfunc(k_raw + self.bias[np.newaxis, :])

                self.cache[tag][reg.name] = {
                    "t": t,
                    "k": k
                }

                global_min = min(global_min, np.min(k))
                global_max = max(global_max, np.max(k))

        self.ymin = global_min
        self.ymax = global_max

        self._axes = []
        self._initialized = False   # 🔑 important flag

    def __call__(self, idx, ax):

        fig = ax.figure

        # --------------------------------------------------
        # FIRST CALL (during NeuronViewer init)
        # Do NOT remove ax or it will crash viewer.
        # --------------------------------------------------
        if not self._initialized:
            self._initialized = True
            ax.clear()
            ax.text(
                0.5, 0.5,
                "Initializing Kernel Viewer...",
                ha="center",
                va="center"
            )
            return

        # --------------------------------------------------
        # Subsequent calls (safe to rebuild layout)
        # --------------------------------------------------

        # remove previous subplot axes
        for a in self._axes:
            a.remove()
        self._axes = []

        # remove the placeholder axis
        ax.remove()

        n_tags = len(self.tags)
        gs = fig.add_gridspec(1, n_tags)

        for i, tag in enumerate(self.tags):

            subax = fig.add_subplot(gs[0, i])
            self._axes.append(subax)

            for regname, data in self.cache[tag].items():
                t = data["t"]
                k = data["k"][:, idx]
                subax.plot(t, k, label=regname)

            subax.set_title(tag)
            subax.set_ylim(self.ymin, self.ymax)
            subax.set_xlabel("Time (s)")

            if i == 0:
                subax.set_ylabel("Predicted rate")

            if tag not in ["history", "dlc", "video"]:
                subax.legend()

        fig.tight_layout()