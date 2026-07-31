import numpy as np

from features_tools.features.spectral import POWER_FLOOR, band_power_matrix


def extract_band_ratios(data, sfreq, bands, ratios):
    """
    Named band-power ratios per channel (e.g. beta/alpha "engagement index").
    `ratios`: {name: (numerator band names, denominator band names)}.
    Flat array ordered channel-major then ratio-minor, length
    n_channels * len(ratios).
    """
    power = band_power_matrix(data, sfreq, bands)
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
                where=denominator > POWER_FLOOR,
            )
        )
    # stack as (n_channels, n_ratios) then flatten channel-major
    return np.stack(values, axis=1).flatten()
