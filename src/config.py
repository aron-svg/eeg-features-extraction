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

# Channel types included in feature extraction (mne picks), e.g. drop
# ecg/eog/misc channels present in the recording.
EEG_CHANNEL_TYPES = ["eeg"]

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

# Catalog of feature extractors implemented in feature_extraction.py
AVAILABLE_FEATURES = ["band_power", "hjorth"]

# Subset actually computed - edit this list to enable/disable extractors
# without touching any code.
FEATURES_TO_EXTRACT = ["band_power", "hjorth"]