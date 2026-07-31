import mne
import numpy as np
from scipy.signal import welch

from logger_init import logger

from config import (
    EEG_CHANNEL_TYPES,
    FEATURES_TO_EXTRACT,
    FREQUENCY_BANDS,
    VAD_COLUMNS,
    WINDOW_LENGTH_SEC,
    WINDOW_STEP_SEC,
)
from features_tools.epoching import generate_sliding_windows, match_stimulus_events


def extract_band_power(data, sfreq, bands):
    """
    Mean PSD (Welch) per channel, integrated over each frequency band.
    `data` shape: (n_channels, n_samples). Returns a flat array ordered
    channel-major then band-minor, length n_channels * len(bands).
    """
    nperseg = min(data.shape[-1], int(sfreq))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)

    band_powers = []
    for channel_psd in psd:
        for low, high in bands.values():
            mask = (freqs >= low) & (freqs <= high)
            band_powers.append(
                np.trapz(channel_psd[mask], freqs[mask]) if mask.any() else 0.0
            )
    return np.array(band_powers)


def extract_hjorth(data):
    """
    Hjorth mobility and complexity per channel.
    `data` shape: (n_channels, n_samples). Returns a flat array:
    [mobility_ch0..N, complexity_ch0..N].
    """
    var0 = np.var(data, axis=-1)
    d1 = np.diff(data, axis=-1)
    var1 = np.var(d1, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var2 = np.var(d2, axis=-1)

    mobility = np.sqrt(np.divide(var1, var0, out=np.zeros_like(var0), where=var0 != 0))
    mobility_d1 = np.sqrt(
        np.divide(var2, var1, out=np.zeros_like(var1), where=var1 != 0)
    )
    complexity = np.divide(
        mobility_d1, mobility, out=np.zeros_like(mobility), where=mobility != 0
    )
    return np.concatenate([mobility, complexity])


_EXTRACTORS = {
    "band_power": lambda data, sfreq: extract_band_power(data, sfreq, FREQUENCY_BANDS),
    "hjorth": lambda data, sfreq: extract_hjorth(data),
}


def extract_features_for_window(data, sfreq):
    blocks = []
    for name in FEATURES_TO_EXTRACT:
        if name not in _EXTRACTORS:
            raise ValueError(f"Unknown feature extractor '{name}' in FEATURES_TO_EXTRACT")
        blocks.append(_EXTRACTORS[name](data, sfreq))
    return np.concatenate(blocks)


def extract_features(fif_path, events_csv_path, subject_id):
    """
    Build (X, y, metadata) for one subject: sliding-window EEG features
    over each labeled stimulus trial, paired with its valence/arousal/
    dominance target.
    """
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    sfreq = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, **{ch_type: True for ch_type in EEG_CHANNEL_TYPES})

    trials = match_stimulus_events(raw, events_csv_path)
    logger.info(f"Matched {len(trials)} labeled stimulus trials in {fif_path}")

    X, y = [], []
    metadata = {
        "subject_id": [],
        "trial_index": [],
        "window_start": [],
        "window_end": [],
        "media_filename": [],
    }

    for trial_index, trial in enumerate(trials):
        windows = generate_sliding_windows(
            trial["onset"], trial["duration"], WINDOW_LENGTH_SEC, WINDOW_STEP_SEC
        )
        for start, end in windows:
            start_sample, stop_sample = raw.time_as_index([start, end])
            data = raw.get_data(picks=picks, start=start_sample, stop=stop_sample)

            X.append(extract_features_for_window(data, sfreq))
            y.append([trial[col] for col in VAD_COLUMNS])
            metadata["subject_id"].append(subject_id)
            metadata["trial_index"].append(trial_index)
            metadata["window_start"].append(start)
            metadata["window_end"].append(end)
            metadata["media_filename"].append(trial["media_filename"])

    X = np.array(X)
    y = np.array(y)
    metadata = {k: np.array(v) for k, v in metadata.items()}
    return X, y, metadata
