from features_tools.features.spectral import differential_entropy_matrix


def extract_differential_entropy(data, sfreq, bands):
    """
    Flat array ordered channel-major then band-minor, length
    n_channels * len(bands).
    """
    return differential_entropy_matrix(data, sfreq, bands).flatten()
