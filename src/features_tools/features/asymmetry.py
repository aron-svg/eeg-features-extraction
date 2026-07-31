import numpy as np

from logger_init import logger

from features_tools.features.spectral import differential_entropy_matrix


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


def extract_differential_asymmetry(data, sfreq, bands, pair_indices):
    """
    DE(left) - DE(right) for each homologous electrode pair and band.
    `pair_indices`: list of (left_row, right_row) indices into `data`.
    Flat array ordered pair-major then band-minor, length
    len(pair_indices) * len(bands).
    """
    de = differential_entropy_matrix(data, sfreq, bands)
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
    de = differential_entropy_matrix(data, sfreq, bands)
    if not pair_indices:
        return np.array([])
    ratios = []
    for left, right in pair_indices:
        denom = de[right]
        ratios.append(np.divide(de[left], denom, out=np.zeros_like(denom), where=denom != 0))
    return np.array(ratios).flatten()
