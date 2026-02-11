from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap


import matplotlib.pyplot as plt 
import numpy as np 

from mne.viz import plot_topomap

viridisBig = cm.get_cmap('jet')
newcmp = ListedColormap(viridisBig(np.linspace(0, 1, 15)))


def plot_CSP_components(eigvals, A, positions, ch_labels, row_idx, gs, fig):
        # первый график: линия eigenvalues
        ax0 = plt.subplot(gs[row_idx, 0])
        ax0.plot(eigvals, "k")
        ax0.scatter(range(len(eigvals)), eigvals, marker="o", s=20)
        ax0.set_ylim(0, 1)
        ax0.set_title("Eigenvalues")
        
        # топоплоты
        ims = []
        idxs = [0, 1, 2, 3, -4, -3, -2, -1]
        vmin = np.min(A[idxs])
        vmax = np.max(A[idxs])
        vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))
        
         # топографические карты
        def plot_topoplot(X, positions, vmin=None, vmax=None, ch_labels=None, axes=None):
                im, cn = plot_topomap(X, positions,  image_interp='cubic', ch_type='eeg', names =ch_labels,
                        size=5, show=False, contours=4, sphere=0.5, 
                        cmap=newcmp, extrapolate='head', axes=axes, vlim=[vmin, vmax])
                return im
        
        for i, idx in enumerate(idxs):
                ax_map = plt.subplot(gs[row_idx, i+1])
                im = plot_topoplot(A[idx], positions, axes=ax_map, 
                        vmin=vmin, vmax=vmax)
                comp_number = idx if idx >= 0 else len(ch_labels)+idx
                ax_map.set_title(f"CSP #{comp_number+1}")
                ims.append(im)

        # общий colorbar справа от последнего topomap
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(ims[-1], cax=cax)
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
        cbar.ax.yaxis.set_tick_params(labelsize=10)