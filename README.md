# EEG Features Extraction

Extraction pipeline that turns preprocessed EEG recordings (MNE `.fif`) into ML-ready feature arrays for predicting **valence / arousal / dominance (VAD)**.

## What it does

For each subject folder in `data/input/`, the pipeline:

1. Loads the preprocessed `.fif` recording and matches its `stimulus` annotations to the corresponding rows in `events.csv` (by nearest onset, within a configurable tolerance) to recover the valence/arousal/dominance rating of each trial.
2. Slides a fixed-length window (default 4s, step 2s) across each labeled trial.
3. Extracts a configurable set of EEG (and optionally peripheral) features per window.
4. Saves one `.npz` per subject in `data/output/` containing `X` (features), `y` (VAD labels), and per-window metadata.

## Project structure

```
src/
├── config.py                    # All pipeline parameters (paths, window size, feature catalog...)
├── main.py                      # Entry point: loops over data/input/<subject>/, extracts, saves .npz
├── features_tools/
│   ├── epoching.py              # Matches .fif annotations <-> events.csv rows, builds sliding windows
│   └── feature_extraction.py    # All feature extractors + dispatch
├── logger_init.py / logging_config.py
scripts/
└── inspect_features.py          # Standalone viewer for the *_features.npz files (no pipeline import)
data/
├── input/<subject_id>/          # preprocessed_*.fif + events.csv (gitignored)
└── output/<subject_id>_features.npz  # (gitignored)
```

## Expected input layout

Each subject is a subfolder of `data/input/`:

```
data/input/HZO024/
├── preprocessed_data_v3_HZO024.fif   # required - the preprocessed EEG recording
└── events.csv                        # required - one "stimulus" row per labeled trial,
                                       # with valence/arousal/dominance columns
```

Other files that may be present alongside these (e.g. `HAPI.json`, `PL_GAZE*.json`) are not used by this pipeline - the LSL sync they provide is already reflected in the `.fif` annotation onsets.

## Running it

```bash
uv sync
uv run python src/main.py
```

This processes every subject subfolder found in `DATA_INPUT_DIR` and writes `data/output/<subject_id>_features.npz`.

## Configuration (`src/config.py`)

All parameters are centralized here - nothing else needs to be touched to change behavior:

| Setting | Purpose |
|---|---|
| `WINDOW_LENGTH_SEC` / `WINDOW_STEP_SEC` | Sliding window size/step within each labeled trial (default 4s / 2s) |
| `ONSET_MATCH_TOLERANCE_SEC` | Max allowed gap when pairing a `.fif` annotation to its `events.csv` row |
| `EEG_CHANNEL_TYPES` | Channel types used by the EEG-specific extractors |
| `PERIPHERAL_CHANNEL_TYPES` / `EXCLUDED_CHANNEL_NAMES` | Non-EEG channels included (ECG/EOG/misc), with non-signal channels (e.g. `timestamp`) dropped by name |
| `ASYMMETRIC_CHANNEL_PAIRS` | Left/right electrode pairs used by the hemispheric asymmetry extractors |
| `FREQUENCY_BANDS` / `BAND_RATIOS` | EEG frequency bands and named band-power ratios |
| `AVAILABLE_FEATURES` | Catalog of implemented extractors |
| `FEATURES_TO_EXTRACT` | Subset actually computed - edit this list to enable/disable features without touching any code |

### Feature catalog

| Feature | Channels | Description |
|---|---|---|
| `band_power` | EEG | Welch PSD integrated per frequency band |
| `hjorth` | EEG | Mobility + complexity per channel |
| `differential_entropy` | EEG | `0.5*log(2*pi*e*power)` per band - Gaussian approximation, standard in EEG-emotion literature (SEED/DEAP) |
| `differential_asymmetry` | EEG | DE(left) - DE(right) per hemispheric pair/band (DASM) |
| `rational_asymmetry` | EEG | DE(left) / DE(right) per hemispheric pair/band (RASM) |
| `band_ratios` | EEG | Named ratios between band powers (e.g. beta/alpha "engagement index") |
| `spectral_entropy` | EEG | Shannon entropy of the normalized PSD, bounded to [0, 1] |
| `peripheral_stats` | ECG / EOG / temp / resp / GSR | Mean, std, min, max, slope per channel |

## Output format

One `.npz` per subject, loadable with `np.load(path, allow_pickle=True)`:

- `X`: `(n_windows, n_features)` - feature matrix
- `y`: `(n_windows, 3)` - valence/arousal/dominance, constant across all windows of the same trial
- `subject_id`, `trial_index`, `window_start`, `window_end`, `media_filename`: per-window metadata

Windows overlap within a trial (they are not independent samples) - group any train/test split by `trial_index` (and `subject_id`, once multiple subjects are combined) to avoid data leakage.

## Inspecting a result

```bash
uv run python scripts/inspect_features.py data/output/HZO024_features.npz [--show]
```

Prints a text summary (shapes, VAD ranges, windows per trial) and saves a 4-panel overview figure next to the input file: standardized feature heatmap, PCA projection colored by stimulus emotion category, valence/arousal plane, and windows-per-trial distribution.

## Setup

```bash
uv sync
```

Installs Python, `mne`, `numpy`, `scipy`, `matplotlib`, and dev tooling (`black`/`flake8`) from `pyproject.toml`.

## Docker

```bash
docker-compose up
```

Builds the image and runs `python src/main.py` against the `data/` folder mounted from the host.

## Git

`data/input/`, `data/output/`, `__pycache__/`, `.venv/`, and `logs/` are gitignored - raw recordings and extracted feature arrays are never committed.
