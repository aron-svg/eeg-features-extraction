"""
Shared low-level spectral utilities, reused by several feature extractors
(band_power, differential_entropy, asymmetry, band_ratios, spectral_entropy).
"""
import numpy as np
from scipy.signal import welch

# Floor applied before log()/division, to avoid log(0) or divide-by-zero on
# a silent/flat band.
POWER_FLOOR = 1e-12


def welch_psd(data, sfreq):
    nperseg = min(data.shape[-1], int(sfreq))
    return welch(data, fs=sfreq, nperseg=nperseg, axis=-1)


def band_power_matrix(data, sfreq, bands):
    """
    Mean PSD (Welch) per channel, integrated over each frequency band.
    `data` shape: (n_channels, n_samples). Returns shape (n_channels, n_bands).
    """
    freqs, psd = welch_psd(data, sfreq)

    powers = np.zeros((psd.shape[0], len(bands)))
    for ch, channel_psd in enumerate(psd):
        for b, (low, high) in enumerate(bands.values()):
            mask = (freqs >= low) & (freqs <= high)
            powers[ch, b] = (
                np.trapz(channel_psd[mask], freqs[mask]) if mask.any() else 0.0
            )
    return powers


def differential_entropy_matrix(data, sfreq, bands):
    """
    Differential entropy per channel/band, assuming the band-limited signal
    is approximately Gaussian: DE = 0.5 * log(2*pi*e*power).
    Returns shape (n_channels, n_bands).
    """
    power = np.maximum(band_power_matrix(data, sfreq, bands), POWER_FLOOR)
    return 0.5 * np.log(2 * np.pi * np.e * power)
