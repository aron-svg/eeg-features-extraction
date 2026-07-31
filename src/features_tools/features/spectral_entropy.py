import numpy as np

from features_tools.features.spectral import POWER_FLOOR, welch_psd


def extract_spectral_entropy(data, sfreq, bands):
    """
    Shannon entropy of the normalized PSD per channel, restricted to the
    overall frequency range covered by `bands` and normalized to [0, 1] by
    log(n_freq_bins). Measures how "flat"/irregular the spectrum is,
    independent of the band power extractors.
    `data` shape: (n_channels, n_samples). Returns shape (n_channels,).
    """
    freqs, psd = welch_psd(data, sfreq)
    low = min(b[0] for b in bands.values())
    high = max(b[1] for b in bands.values())
    mask = (freqs >= low) & (freqs <= high)
    psd = psd[:, mask]

    psd_norm = psd / np.maximum(psd.sum(axis=-1, keepdims=True), POWER_FLOOR)
    entropy = -(psd_norm * np.log(psd_norm + POWER_FLOOR)).sum(axis=-1)
    return entropy / np.log(psd.shape[-1])
