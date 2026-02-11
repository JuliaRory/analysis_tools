import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils.parse_h5df import load_h5df
from src.utils.spectral_analysis import bandpass_filter, compute_psd_welch, compute_windowed_fft
from src.utils.montage_processing import find_ch_idx
from src.utils.rereferencing import rereference_eeg
from src.visualization.plot_signal import plot_signal
from src.visualization.spectrogram import plot_spectrogram
from src.utils.transformations import unit_to_db
from src.visualization.check_alpha_rhythm import plot_spectr, plot_alpha_spectr

# ch_labels = get_channel_names(CED_FILE)
# positions = get_topo_positions(CED_FILE)


# idx_Fz = find_ch_idx("Fz", CED_FILE)

# DATA_FOLDER = r".\data"
# RECORD = "01-open-closed-eyes.hdf"

DATA_FOLDER = r"R:\data\dry_gel"
RECORD = "opened_closed_eyes.hdf"

EEG_CHANNELS = np.arange(12)

Fs = 1000 # Hz
s_to_idx = lambda x: int(x * Fs)
ms_to_idx = lambda x: int(x // 1000 * Fs)

CED_FILE = r"./resources/mks64_standard.ced"
labels_ROA = ["FC5", "FC3", "FC1", "C1", "CP1", "CP3", "CP5", "C5", "C3"] # mu rhythm
labels_ROA = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Fz", "FCz", "AF3", "AF4"] # occipital lobe and frontal electrodes

# =================================
# ==========DRY GEL TEST===========
# =================================
CED_FILE = r"./resources/mks10.ced"
labels_ROA = ["PO3", "POz", "PO4", "O1", "Oz", "O2", "Fz", "Cz", "P5", "P6"] # occipital lobe and frontal electrodes

idxs_ROA = [find_ch_idx(ch, CED_FILE) for ch in labels_ROA]

plt.ion() 

do_referencing = True
REFERENT_CHANNELS = [10, 11]    # or NONE

# == load dataset ==

data, _ = load_h5df(os.path.join(DATA_FOLDER, RECORD))
print("Data shape: {}".format(data.shape))

# == preprocessing == 

# bandpass filter
print("---filtering---")
raw_eeg = data[:-1, EEG_CHANNELS] * 1E6 # uV
filt_eeg = bandpass_filter(raw_eeg, fs=Fs, low=0.5, high=40)


# re-referencing Fz
print("---re referencing---")
if REFERENT_CHANNELS is None:
    idx_Fz = find_ch_idx("Fz", CED_FILE)
    reref_eeg = rereference_eeg(filt_eeg, idx_Fz)
else:
    # idx_Fz = find_ch_idx("Fz", CED_FILE)
    idx_ref = REFERENT_CHANNELS
    reref_eeg = rereference_eeg(filt_eeg, idx_ref)

signal = filt_eeg if not do_referencing else reref_eeg

# == check signal == 
start_s, end_s = 0, raw_eeg.shape[0] // 1000 
plot_signal(start_s, end_s, signal, s_to_idx, plot=True)  # plot all channels

# === spectral analysis ===
print("---psd calculation---")
# freq, psd = compute_psd_welch(signal, fs=Fs, fmin=0.5, fmax=40, freq_res=.5)
# psd_db = unit_to_db(psd)

# print(np.min(psd_db), np.max(psd_db))
# plot_spectr(freq, psd[idxs_ROA], labels_ROA, plot_mean=True, freq_min = 0, freq_max=30, y_min=np.min(psd), y_max=np.max(psd), to_db=False)
# plot_spectr(freq, psd_db[idxs_ROA], labels_ROA, plot_mean=True, freq_min = 0, freq_max=30, y_min=np.min(psd_db), y_max=np.max(psd_db), to_db=True)

idx_half = len(raw_eeg) // 2
eeg_opened = signal[:idx_half, :]       # opened eyes
eeg_closed = signal[idx_half:, :]       # closed eyes

freq, psd_opened = compute_psd_welch(eeg_opened, fs=Fs, fmin=0.5, fmax=40, freq_res=.5)
freq, psd_closed = compute_psd_welch(eeg_closed, fs=Fs, fmin=0.5, fmax=40, freq_res=.5)

max_psd = max(np.max(psd_opened), np.max(psd_closed))
min_psd =0
fig = plot_alpha_spectr(freq, psd_opened[idxs_ROA], psd_closed[idxs_ROA], labels_ROA,  plot_mean=True, y_min=min_psd,  y_max=max_psd, 
                 freq_min = 0, freq_max=20, to_db=False)


f, t, S = compute_windowed_fft(reref_eeg, fs=Fs)

fig, ax = plot_spectrogram(f, t, S, 
                           average=True, channels=idxs_ROA,
                           fmin=1, fmax=40, 
                           title=f"EEG Spectrogram\n(average of channels: {', '.join(labels_ROA)})",
                           symmetric=True)
ax.axvline(idx_half / Fs, color='white')
ax.text(20, 37, "Opened eyes", color='white', fontsize=20)
ax.text(160, 37, "Closed eyes", color='white', fontsize=20)

plt.ioff()     
plt.show()     
