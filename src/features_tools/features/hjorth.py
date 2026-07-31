import numpy as np


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
