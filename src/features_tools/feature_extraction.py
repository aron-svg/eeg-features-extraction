import mne
import numpy as np
from scipy.signal import welch

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
    WINDOW_STEP_SEC,
)
from features_tools.epoching import generate_sliding_windows, match_stimulus_events

# Floor applied before log()/division, to avoid log(0) or divide-by-zero on
# a silent/flat band.
_POWER_FLOOR = 1e-12


def _welch_psd(data, sfreq):
    nperseg = min(data.shape[-1], int(sfreq))
    return welch(data, fs=sfreq, nperseg=nperseg, axis=-1)


def _band_power_matrix(data, sfreq, bands):
    """
    Mean PSD (Welch) per channel, integrated over each frequency band.
    `data` shape: (n_channels, n_samples). Returns shape (n_channels, n_bands).
    """
    freqs, psd = _welch_psd(data, sfreq)

    powers = np.zeros((psd.shape[0], len(bands)))
    for ch, channel_psd in enumerate(psd):
        for b, (low, high) in enumerate(bands.values()):
            mask = (freqs >= low) & (freqs <= high)
            powers[ch, b] = (
                np.trapz(channel_psd[mask], freqs[mask]) if mask.any() else 0.0
            )
    return powers


def _differential_entropy_matrix(data, sfreq, bands):
    """
    Differential entropy per channel/band, assuming the band-limited signal
    is approximately Gaussian: DE = 0.5 * log(2*pi*e*power).
    Returns shape (n_channels, n_bands).
    """
    power = np.maximum(_band_power_matrix(data, sfreq, bands), _POWER_FLOOR)
    return 0.5 * np.log(2 * np.pi * np.e * power)


def extract_band_power(data, sfreq, bands):
    """
    Flat array ordered channel-major then band-minor, length
    n_channels * len(bands).
    """
    return _band_power_matrix(data, sfreq, bands).flatten()


def extract_differential_entropy(data, sfreq, bands):
    """
    Flat array ordered channel-major then band-minor, length
    n_channels * len(bands).
    """
    return _differential_entropy_matrix(data, sfreq, bands).flatten()


def extract_differential_asymmetry(data, sfreq, bands, pair_indices):
    """
    DE(left) - DE(right) for each homologous electrode pair and band.
    `pair_indices`: list of (left_row, right_row) indices into `data`.
    Flat array ordered pair-major then band-minor, length
    len(pair_indices) * len(bands).
    """
    de = _differential_entropy_matrix(data, sfreq, bands)
    if not pair_indices:
        return np.array([])
    return np.array([de[left] - de[right] for left, right in pair_indices]).flatten()


def extract_rational_asymmetry(data, sfreq, bands, pair_indices):
    """
    DE(left) / DE(right) for each homologous electrode pair and band -
    companion ratio to differential_asymmetry (SEED/DEAP "RASM" feature).
    Flat array ordered pair-major then band-minor, length
    len(pair_indices) * len(bands).
    """
    de = _differential_entropy_matrix(data, sfreq, bands)
    if not pair_indices:
        return np.array([])
    ratios = []
    for left, right in pair_indices:
        denom = de[right]
        ratios.append(np.divide(de[left], denom, out=np.zeros_like(denom), where=denom != 0))
    return np.array(ratios).flatten()


def extract_band_ratios(data, sfreq, bands, ratios):
    """
    Named band-power ratios per channel (e.g. beta/alpha "engagement index").
    `ratios`: {name: (numerator band names, denominator band names)}.
    Flat array ordered channel-major then ratio-minor, length
    n_channels * len(ratios).
    """
    power = _band_power_matrix(data, sfreq, bands)
    band_names = list(bands.keys())

    values = []
    for _, (num_bands, denom_bands) in ratios.items():
        num_idx = [band_names.index(b) for b in num_bands]
        denom_idx = [band_names.index(b) for b in denom_bands]
        numerator = power[:, num_idx].sum(axis=1)
        denominator = power[:, denom_idx].sum(axis=1)
        values.append(
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > _POWER_FLOOR,
            )
        )
    # stack as (n_channels, n_ratios) then flatten channel-major
    return np.stack(values, axis=1).flatten()


def extract_spectral_entropy(data, sfreq, bands):
    """
    Shannon entropy of the normalized PSD per channel, restricted to the
    overall frequency range covered by `bands` and normalized to [0, 1] by
    log(n_freq_bins). Measures how "flat"/irregular the spectrum is,
    independent of the band power extractors.
    `data` shape: (n_channels, n_samples). Returns shape (n_channels,).
    """
    freqs, psd = _welch_psd(data, sfreq)
    low = min(b[0] for b in bands.values())
    high = max(b[1] for b in bands.values())
    mask = (freqs >= low) & (freqs <= high)
    psd = psd[:, mask]

    psd_norm = psd / np.maximum(psd.sum(axis=-1, keepdims=True), _POWER_FLOOR)
    entropy = -(psd_norm * np.log(psd_norm + _POWER_FLOOR)).sum(axis=-1)
    return entropy / np.log(psd.shape[-1])


def resolve_channel_pairs(ch_names, configured_pairs):
    """
    Map channel-name pairs from config to row indices into a `picks`-ordered
    data array. Pairs whose channel(s) aren't present are skipped (with a
    warning) rather than raising, so a subject missing a few electrodes
    doesn't break the whole extraction.
    """
    name_to_index = {name: i for i, name in enumerate(ch_names)}
    pair_indices = []
    for left, right in configured_pairs:
        if left in name_to_index and right in name_to_index:
            pair_indices.append((name_to_index[left], name_to_index[right]))
        else:
            logger.warning(
                f"Skipping asymmetry pair ({left}, {right}): not both present "
                f"in recording channels."
            )
    return pair_indices


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


def extract_peripheral_stats(data):
    """
    Generic time-domain descriptors per channel: mean, std, min, max, slope.
    Meant for slow/non-oscillatory peripheral signals (ECG, EOG, GSR,
    respiration, temperature) where EEG-style band power isn't meaningful.
    `data` shape: (n_channels, n_samples). Returns a flat array ordered
    channel-major then stat-minor, length n_channels * 5.
    """
    if data.shape[0] == 0:
        return np.array([])

    n_samples = data.shape[-1]
    t = np.arange(n_samples)
    stats = []
    for channel in data:
        slope = np.polyfit(t, channel, 1)[0] if n_samples > 1 else 0.0
        stats.extend([channel.mean(), channel.std(), channel.min(), channel.max(), slope])
    return np.array(stats)


def resolve_picks(ch_names, excluded_names):
    """
    Row indices (into a `picks`-ordered data array) to keep after dropping
    any channel name in `excluded_names` (e.g. a non-signal "timestamp"
    channel bundled under a real channel type).
    """
    return [i for i, name in enumerate(ch_names) if name not in excluded_names]


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
            trial["onset"], trial["duration"], WINDOW_LENGTH_SEC, WINDOW_STEP_SEC
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
