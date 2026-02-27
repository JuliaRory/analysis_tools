from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt 
import numpy as np 

from mne.viz import plot_topomap

viridisBig = cm.get_cmap('jet')
newcmp = ListedColormap(viridisBig(np.linspace(0, 1, 15)))

# топографические карты
def plot_topoplot(X, positions, vmin=None, vmax=None, ch_labels=None, axes=None):
        im, cn = plot_topomap(X, positions,  image_interp='cubic', ch_type='eeg', names =ch_labels,
                size=5, show=False, contours=4, sphere=0.5, 
                cmap=newcmp, extrapolate='head', axes=axes, vlim=[vmin, vmax])
        return im

def plot_components(projForward, xy, ch_labels, gs=None, row_ind=None, idxs=None):
        ims = []
        if idxs is None:
                idxs = [0, 1, 2, 3, -4, -3, -2, -1]
        vmin, vmax = np.min(projForward[:, idxs]), np.max(projForward[:, idxs])
        vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))
        
        for i, idx in enumerate(idxs):
                ax_map = plt.subplot(gs[row_ind, i+1])
                im = plot_topoplot(projForward[:, idx], xy, axes=ax_map, 
                        vmin=vmin, vmax=vmax)
                comp_number = idx if idx >= 0 else len(ch_labels)+idx
                ax_map.set_title(f"CSP #{comp_number+1}")
                ims.append(im)
        
        return ims, vmin, vmax

def plot_eigenvalues(eigvals, ax):
        ax.plot(eigvals, "k")
        ax.scatter(range(len(eigvals)), eigvals, marker="o", s=20)
        ax.set_ylim(0, 1)
        ax.set_title("Eigenvalues")

        ediff = np.diff(eigvals)
        ok_steps = np.where(ediff > np.median(ediff) * 5)[0]
        ok_evalLow_inds = np.arange(np.max(ok_steps[ok_steps < 10]))
        ok_evalHigh_inds = np.arange(len(eigvals)-1, np.min(ok_steps[ok_steps > 30]), -1)
        
        ax.scatter(ok_evalLow_inds, eigvals[ok_evalLow_inds],  label='ERD OK')
        ax.scatter(ok_evalHigh_inds, eigvals[ok_evalHigh_inds], label='ERS OK')



def plot_CSP_components(eigvals, A, positions, ch_labels, row_idx, gs, fig):
        # первый график: линия eigenvalues

        ax0 = plt.subplot(gs[row_idx, 0])
        plot_eigenvalues(eigvals, ax0)
        
        # топоплоты
        ims, vmin, vmax = plot_components(A, positions, ch_labels, gs, row_idx)
        
        # общий colorbar справа от последнего topomap
        ax_map = plt.subplot(gs[row_idx, len(ims)])
        divider = make_axes_locatable(ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(ims[-1], cax=cax)
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
        cbar.ax.yaxis.set_tick_params(labelsize=10)