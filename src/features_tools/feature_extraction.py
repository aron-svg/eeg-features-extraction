import mne
import numpy as np

from logger_init import logger

from config import (
    ASYMMETRIC_CHANNEL_PAIRS,
    BAND_RATIOS,
    EEG_CHANNEL_TYPES,
    EXCLUDED_CHANNEL_NAMES,
    FEATURES_TO_EXTRACT,
    FREQUENCY_BANDS,
    PERIPHERAL_CHANNEL_TYPES,
    VAD_COLUMNS,
    WINDOW_LENGTH_SEC,
    WINDOW_SKIP_SEC,
    WINDOW_STEP_SEC,
)
from features_tools.epoching import generate_sliding_windows, match_stimulus_events
from features_tools.features.asymmetry import (
    extract_differential_asymmetry,
    extract_rational_asymmetry,
    resolve_channel_pairs,
)
from features_tools.features.band_power import extract_band_power
from features_tools.features.band_ratios import extract_band_ratios
from features_tools.features.differential_entropy import extract_differential_entropy
from features_tools.features.hjorth import extract_hjorth
from features_tools.features.peripheral_stats import extract_peripheral_stats, resolve_picks
from features_tools.features.spectral_entropy import extract_spectral_entropy

_EXTRACTORS = {
    "band_power": lambda channel_data, sfreq, pair_indices: extract_band_power(
        channel_data["eeg"], sfreq, FREQUENCY_BANDS
    ),
    "hjorth": lambda channel_data, sfreq, pair_indices: extract_hjorth(channel_data["eeg"]),
    "differential_entropy": lambda channel_data, sfreq, pair_indices: extract_differential_entropy(
        channel_data["eeg"], sfreq, FREQUENCY_BANDS
    ),
    "differential_asymmetry": lambda channel_data, sfreq, pair_indices: extract_differential_asymmetry(
        channel_data["eeg"], sfreq, FREQUENCY_BANDS, pair_indices
    ),
    "rational_asymmetry": lambda channel_data, sfreq, pair_indices: extract_rational_asymmetry(
        channel_data["eeg"], sfreq, FREQUENCY_BANDS, pair_indices
    ),
    "band_ratios": lambda channel_data, sfreq, pair_indices: extract_band_ratios(
        channel_data["eeg"], sfreq, FREQUENCY_BANDS, BAND_RATIOS
    ),
    "spectral_entropy": lambda channel_data, sfreq, pair_indices: extract_spectral_entropy(
        channel_data["eeg"], sfreq, FREQUENCY_BANDS
    ),
    "peripheral_stats": lambda channel_data, sfreq, pair_indices: extract_peripheral_stats(
        channel_data["peripheral"]
    ),
}


def extract_features_for_window(channel_data, sfreq, pair_indices):
    blocks = []
    for name in FEATURES_TO_EXTRACT:
        if name not in _EXTRACTORS:
            raise ValueError(f"Unknown feature extractor '{name}' in FEATURES_TO_EXTRACT")
        blocks.append(_EXTRACTORS[name](channel_data, sfreq, pair_indices))
    return np.concatenate(blocks)


def extract_features(fif_path, events_csv_path, subject_id):
    """
    Build (X, y, metadata) for one subject: sliding-window EEG features
    over each labeled stimulus trial, paired with its valence/arousal/
    dominance target.
    """
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    sfreq = raw.info["sfreq"]
    eeg_picks = mne.pick_types(raw.info, **{ch_type: True for ch_type in EEG_CHANNEL_TYPES})

    pair_indices = []
    if "differential_asymmetry" in FEATURES_TO_EXTRACT or "rational_asymmetry" in FEATURES_TO_EXTRACT:
        eeg_ch_names = [raw.ch_names[p] for p in eeg_picks]
        pair_indices = resolve_channel_pairs(eeg_ch_names, ASYMMETRIC_CHANNEL_PAIRS)

    peripheral_picks = []
    if "peripheral_stats" in FEATURES_TO_EXTRACT:
        raw_peripheral_picks = mne.pick_types(
            raw.info, **{ch_type: True for ch_type in PERIPHERAL_CHANNEL_TYPES}
        )
        peripheral_ch_names = [raw.ch_names[p] for p in raw_peripheral_picks]
        keep = resolve_picks(peripheral_ch_names, EXCLUDED_CHANNEL_NAMES)
        peripheral_picks = [raw_peripheral_picks[i] for i in keep]

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
            trial["onset"],
            trial["duration"],
            WINDOW_LENGTH_SEC,
            WINDOW_STEP_SEC,
            skip=WINDOW_SKIP_SEC,
        )
        for start, end in windows:
            start_sample, stop_sample = raw.time_as_index([start, end])
            channel_data = {
                "eeg": raw.get_data(picks=eeg_picks, start=start_sample, stop=stop_sample),
                "peripheral": raw.get_data(
                    picks=peripheral_picks, start=start_sample, stop=stop_sample
                ),
            }

            X.append(extract_features_for_window(channel_data, sfreq, pair_indices))
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
