import csv

from logger_init import logger

from config import (
    ONSET_MATCH_TOLERANCE_SEC,
    STIMULUS_DESCRIPTION,
    VAD_COLUMNS,
)


def _read_events_csv(events_csv_path):
    with open(events_csv_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def match_stimulus_events(raw, events_csv_path):
    """
    Pair each `STIMULUS_DESCRIPTION` annotation in `raw` with its labeled
    row in events.csv, matched by nearest onset within
    `ONSET_MATCH_TOLERANCE_SEC`.

    Returns a list of dicts: onset, duration, media_filename, and the
    VAD_COLUMNS values (as floats).
    """
    rows = _read_events_csv(events_csv_path)
    candidate_rows = [
        r
        for r in rows
        if r.get("description") == STIMULUS_DESCRIPTION
        and all(r.get(col) not in ("", None) for col in VAD_COLUMNS)
    ]

    trials = []
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        if description != STIMULUS_DESCRIPTION:
            continue

        best_row = min(
            candidate_rows,
            key=lambda r: abs(float(r["onset"]) - onset),
            default=None,
        )
        if best_row is None:
            raise ValueError(
                f"No events.csv row with description="
                f"'{STIMULUS_DESCRIPTION}' left to match annotation at "
                f"onset={onset:.3f}s"
            )

        gap = abs(float(best_row["onset"]) - onset)
        if gap > ONSET_MATCH_TOLERANCE_SEC:
            raise ValueError(
                f"Closest events.csv row for annotation at onset="
                f"{onset:.3f}s is {gap:.3f}s away, exceeding tolerance "
                f"{ONSET_MATCH_TOLERANCE_SEC}s"
            )

        candidate_rows.remove(best_row)
        trials.append(
            {
                "onset": onset,
                "duration": duration,
                "media_filename": best_row.get("media_filename"),
                **{col: float(best_row[col]) for col in VAD_COLUMNS},
            }
        )

    return trials


def generate_sliding_windows(onset, duration, window_length, step):
    """
    Fixed-length sliding windows (start, end) covering [onset, onset+duration],
    stepping by `step`. Only full windows are kept (no partial tail window).
    """
    if duration < window_length:
        logger.warning(
            f"Trial at onset={onset:.3f}s has duration={duration:.3f}s "
            f"shorter than window_length={window_length}s, skipping."
        )
        return []

    windows = []
    start = onset
    end_limit = onset + duration
    while start + window_length <= end_limit + 1e-9:
        windows.append((start, start + window_length))
        start += step

    return windows
