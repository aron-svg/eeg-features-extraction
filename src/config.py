"""
Configuration variables for the EEG features extraction pipeline.
"""
import os
#######################################
# Data directories
#######################################

#project root: parent directory of this file's directory (src/)
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_INPUT_DIR = os.path.join(DATA_DIR, "data", "input")
DATA_OUTPUT_DIR = os.path.join(DATA_DIR, "data", "output")

#######################################
# Stimulus / labels matching
#######################################

# Annotation description in the .fif marking a labeled trial
STIMULUS_DESCRIPTION = "stimulus"

# events.csv columns holding the regression targets
VAD_COLUMNS = ["valence", "arousal", "dominance"]

# Max allowed gap (seconds) between a .fif annotation onset and its
# matching events.csv row onset. Guards against silent desync on a
# future subject rather than pairing rows positionally.
ONSET_MATCH_TOLERANCE_SEC = 0.05

#######################################
# Sliding window
#######################################

WINDOW_LENGTH_SEC = 4.0
WINDOW_STEP_SEC = 2.0

#######################################
# Channels
#######################################

# Channel types used for the EEG-specific extractors (band power, hjorth,
# differential entropy/asymmetry).
EEG_CHANNEL_TYPES = ["eeg"]

# Other recorded channel types worth extracting: ECG (heart activity),
# EOG (eye movement/blinks), misc (bundles temperature/respiration/GSR in
# this recording - see EXCLUDED_CHANNEL_NAMES for the non-signal one).
PERIPHERAL_CHANNEL_TYPES = ["ecg", "eog", "misc"]

# Channel names to drop regardless of type - e.g. "timestamp" is a time
# reference channel bundled under "misc", not a physiological signal.
EXCLUDED_CHANNEL_NAMES = ["timestamp"]

# Left/right homologous electrode pairs (10-20 convention: odd=left,
# even=right), used by the "differential_asymmetry" extractor. Pairs
# whose channels aren't found in a given recording are skipped.
ASYMMETRIC_CHANNEL_PAIRS = [
    ("Fp1", "Fp2"),
    ("AF3", "AF4"),
    ("F7", "F8"),
    ("F3", "F4"),
    ("FC1", "FC2"),
    ("FC5", "FC6"),
    ("T7", "T8"),
    ("C3", "C4"),
    ("CP1", "CP2"),
    ("CP5", "CP6"),
    ("P7", "P8"),
    ("P3", "P4"),
    ("PO3", "PO4"),
    ("O1", "O2"),
]

#######################################
# Features
#######################################

FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Named band-power ratios, used by the "band_ratios" extractor:
# {name: (numerator band names, denominator band names)}. Numerator/
# denominator powers are summed when more than one band is listed.
BAND_RATIOS = {
    "beta_alpha": (["beta"], ["alpha"]),  # engagement index
    "theta_alpha_beta": (["theta", "alpha"], ["beta"]),  # cognitive load index
}

# Catalog of feature extractors implemented in feature_extraction.py
AVAILABLE_FEATURES = [
    "band_power",
    "hjorth",
    "differential_entropy",
    "differential_asymmetry",
    "rational_asymmetry",
    "band_ratios",
    "spectral_entropy",
    "peripheral_stats",
]

# Subset actually computed - edit this list to enable/disable extractors
# without touching any code.
FEATURES_TO_EXTRACT = [
    "band_power",
    "hjorth",
    "differential_entropy",
    "differential_asymmetry",
    "rational_asymmetry",
    "band_ratios",
    "spectral_entropy",
    "peripheral_stats",
]