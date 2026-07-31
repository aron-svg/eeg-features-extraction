from features_tools.features.spectral import band_power_matrix


def extract_band_power(data, sfreq, bands):
    """
    Flat array ordered channel-major then band-minor, length
    n_channels * len(bands).
    """
    return band_power_matrix(data, sfreq, bands).flatten()
