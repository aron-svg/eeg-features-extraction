import numpy as np


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
