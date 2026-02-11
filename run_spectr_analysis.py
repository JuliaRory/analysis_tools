import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils.parse_h5df import load_h5df
from src.utils.spectral_analysis import bandpass_filter, compute_psd_welch
from src.utils.montage_processing import find_ch_idx
from src.utils.rereferencing import rereference_eeg
from src.visualization.plot_signal import plot_signal
from src.utils.transformations import unit_to_db
from src.visualization.check_alpha_rhythm import plot_spectr

# ch_labels = get_channel_names(CED_FILE)
# positions = get_topo_positions(CED_FILE)


# idx_Fz = find_ch_idx("Fz", CED_FILE)

DATASET = "pilot/01_calib_sessions"
SESSION = "02ES_ses10.12.2025_calib_v1.1"
DATA_FOLDER = os.path.join(r".\data\raw\pilot\01_calib_sessions", SESSION, "data")
RECORD = "01-open-closed-eyes.hdf"

EEG_CHANNELS = np.arange(64)
Fs = 1000 # Hz
s_to_idx = lambda x: int(x * Fs)
ms_to_idx = lambda x: int(x // 1000 * Fs)

CED_FILE = r"./resources/mks64_standard.ced"
labels_ROA = ["FC5", "FC3", "FC1", "C1", "CP1", "CP3", "CP5", "C5", "C3"] # mu rhythm
labels_ROA = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Fz", "FCz", "AF3", "AF4"] # occipital lobe and frontal electrodes
idxs_ROA = [find_ch_idx(ch, CED_FILE) for ch in labels_ROA]

plt.ion() 

do_referencing = False

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
idx_Fz = find_ch_idx("Fz", CED_FILE)
reref_eeg = rereference_eeg(filt_eeg, idx_Fz)

signal = filt_eeg if not do_referencing else reref_eeg

# == check signal == 
start_s, end_s = 0, raw_eeg.shape[0] // 1000 
plot_signal(start_s, end_s, signal, s_to_idx, plot=True)  # plot all channels

# === spectral analysis ===
print("---psd calculation---")
freq, psd = compute_psd_welch(signal, fs=Fs, fmin=0.5, fmax=40, freq_res=.5)
psd_db = unit_to_db(psd)

print(np.min(psd_db), np.max(psd_db))
plot_spectr(freq, psd[idxs_ROA], labels_ROA, plot_mean=True, freq_min = 0, freq_max=30, y_min=np.min(psd), y_max=np.max(psd), to_db=False)
plot_spectr(freq, psd_db[idxs_ROA], labels_ROA, plot_mean=True, freq_min = 0, freq_max=30, y_min=np.min(psd_db), y_max=np.max(psd_db), to_db=True)


plt.ioff()     
plt.show()     
