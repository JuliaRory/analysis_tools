# === project setup ===
from pathlib import Path
import sys

# === imports ===
from h5py import File 
import numpy as np 

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


from src.utils.parse_h5df import * 
from src.utils.fb_quasi_parse_events import reparse_trigger_v1_1, trigger_to_event_v1_1
from src.utils.events import * 
from src.utils.spectral_analysis import *
from src.utils.transformations import unit_to_db
from src.utils.montage_processing import *
from src.utils.rereferencing import *

from src.utils.CSP import calculate_CSP
from src.visualization.plot_csp_components import plot_CSP_components



# === config ===


EEG_CHANNELS = np.arange(64)
CED_FILE = r"./resources/mks64_standard.ced"

ch_labels = get_channel_names(CED_FILE)
positions = get_topo_positions(CED_FILE)

labels_ROA = ["FC5", "FC3", "FC1", "C1", "CP1", "CP3", "CP5", "C5", "C3"] 
idxs_ROA = [find_ch_idx(ch, CED_FILE) for ch in labels_ROA]
idx_Fz = find_ch_idx("Fz", CED_FILE)

Fs = 1000 # Hz
s_to_idx = lambda x: int(x * Fs)
ms_to_idx = lambda x: int(x // 1000 * Fs)

def receive_csp_components(data_folder):
    for record in os.listdir(data_folder):
        if record == "01-open-closed-eyes.hdf":
            continue
        print(f"======================")
        print(f"======={record}=======")
        data, _ = load_h5df(os.path.join(data_folder, record))
        raw_eeg = data[:, EEG_CHANNELS] * 1E6 # uV

        trigger = reverse_trigger(ttl2binary(data[:, -1], bit_index=0))

        events, trigger_sum = trigger_to_event_v1_1(trigger, window_size=600)        # 1 - motor, 2 - rest
        idx_motor = receive_epochs(events, event_code=1)
        idx_rest = receive_epochs(events, event_code=2)

        bands = [[8, 30], [8, 12], [9, 13], [10, 14], [11, 15]]

        fig = plt.figure(figsize=(22, 3 * len(bands)))
        gs = gridspec.GridSpec(len(bands), 9, height_ratios=[1]*len(bands), wspace=0.3)

        for row_idx, (low_f, high_f) in enumerate(bands): 
            filt_eeg = bandpass_filter(raw_eeg, fs=Fs, low=low_f, high=high_f)
            epochs_motor, epochs_rest = slice_epochs(filt_eeg, idx_motor), slice_epochs(filt_eeg, idx_rest)
            eigvals, eigvecs, A = calculate_CSP(epochs_motor , epochs_rest)
            
            plot_CSP_components(eigvals, A, positions, ch_labels, row_idx, gs, fig)

            # Добавляем название полосы **над всей строкой**
            ax0 = plt.subplot(gs[row_idx, 0])
            pos = ax0.get_position()  # BBox

            # Добавляем название полосы
            fig.text(0.5, pos.y1 + 0.01, f"Bandpass filter: {low_f}-{high_f} Hz",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
            print(f"{low_f}-{high_f} Hz -- done.")
        plt.suptitle(record, y=0.93)
        plt.show()
        

