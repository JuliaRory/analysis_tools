from numpy import asarray, eye, newaxis, ones, mean, integer, ndarray


def rereference_eeg(eeg_data, ref_idx):
    """
    Re-reference EEG data relative to one or several reference electrodes.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_samples, n_channels)
        Original EEG signal.
    ref_idx : int or sequence of ints
        Index (or indices) of reference electrode(s) (0-based).

    Returns
    -------
    eeg_reref : ndarray, shape (n_samples, n_channels)
        EEG signal re-referenced to the given electrode(s).
    """
    eeg_data = asarray(eeg_data)
    n_channels = eeg_data.shape[1]

    # Приводим ref_idx к массиву индексов
    if isinstance(ref_idx, (int, integer)):
        ref_idx = [ref_idx]
    elif isinstance(ref_idx, (list, tuple, ndarray)):
        ref_idx = list(ref_idx)
    else:
        raise TypeError("ref_idx must be int or a sequence of ints.")

    # Проверка границ
    # for idx in ref_idx:
    #     if idx < 0 or idx >= n_channels:
    #         raise ValueError(
    #             f"ref_idx ({idx}) is out of bounds for {n_channels} channels."
    #         )

    # Вычисляем средний референсный сигнал
    ref_signal = eeg_data[:, ref_idx].mean(axis=1, keepdims=True)

    # Вычитаем его из всех каналов
    eeg_reref = eeg_data - ref_signal

    return eeg_reref

def rereference_eeg_matrix(eeg_data, ref_idx):
    """
    Re-reference EEG data relative to a specific reference electrode using a matrix.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_samples, n_channels)
        Original EEG signal.
    ref_idx : int
        Index of the reference electrode (0-based).

    Returns
    -------
    eeg_reref : ndarray, shape (n_samples, n_channels)
        EEG signal re-referenced to the given electrode.
    """
    n_samples, n_channels = eeg_data.shape
    
    if ref_idx < 0 or ref_idx >= n_channels:
        raise ValueError(f"ref_idx ({ref_idx}) is out of bounds for {n_channels} channels.")
    
    R = eye(n_channels) - eye(n_channels)[:, ref_idx][:, newaxis] 
    
    # умножение по каналам
    eeg_reref = eeg_data @ R.T  # shape: (n_samples, n_channels)
    
    return eeg_reref

def rereference_eeg_simple(eeg_data, ref_idx):
    """
    Re-reference EEG data relative to a specific reference electrode.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_samples, n_channels)
        Original EEG signal.
    ref_idx : int
        Index of the reference electrode (0-based).

    Returns
    -------
    eeg_reref : ndarray, shape (n_samples, n_channels)
        EEG signal re-referenced to the given electrode.
    """
    eeg_data = asarray(eeg_data)
    
    if ref_idx < 0 or ref_idx >= eeg_data.shape[1]:
        raise ValueError(f"ref_idx ({ref_idx}) is out of bounds for {eeg_data.shape[1]} channels.")
    
    # вычитаем сигнал опорного электрода из всех каналов
    eeg_reref = eeg_data - eeg_data[:, [ref_idx]]
    
    return eeg_reref

def apply_car(eeg_data, exclude_channels_idx=None):
    """
    Apply Common Average Reference (CAR) to EEG data.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_samples, n_channels)
        Raw EEG signal.
    exclude_channels_idx : list or ndarray, optional
        Indices of channels to exclude from CAR computation
        (e.g. bad channels or reference electrode).

    Returns
    -------
    eeg_car : ndarray, shape (n_samples, n_channels)
        EEG data after CAR re-referencing.
    """
    eeg_data = asarray(eeg_data)

    if exclude_channels_idx is None:
        exclude_channels_idx = []

    # каналы, участвующие в вычислении среднего
    include_mask = ones(eeg_data.shape[1], dtype=bool)
    include_mask[exclude_channels_idx] = False

    # общее среднее по каналам
    car = mean(eeg_data[:, include_mask], axis=1, keepdims=True)

    # вычитаем CAR
    eeg_car = eeg_data - car

    return eeg_car