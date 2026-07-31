import glob
import os
import sys

import numpy as np

from logger_init import logger
from config import DATA_INPUT_DIR, DATA_OUTPUT_DIR
from features_tools.feature_extraction import extract_features


def check_input_files(input_dir):
    """
    Check if the input directory contains any files.
    If not, log an error and exit the program.
    """
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist.")
        sys.exit(1)

    if not os.listdir(input_dir):
        logger.error(f"No input files found in {input_dir}. Please add files to process.")
        sys.exit(1)
    else:
        logger.info(f"Found {len(os.listdir(input_dir))} input files in {input_dir}.")


def find_subject_files(subject_dir):
    """
    Locate the preprocessed .fif recording and events.csv for one subject
    subfolder (e.g. data/input/HZO024/).
    """
    fif_files = glob.glob(os.path.join(subject_dir, "*.fif"))
    if not fif_files:
        raise FileNotFoundError(f"No .fif file found in {subject_dir}")

    events_csv_path = os.path.join(subject_dir, "events.csv")
    if not os.path.exists(events_csv_path):
        raise FileNotFoundError(f"No events.csv found in {subject_dir}")

    return fif_files[0], events_csv_path


def save_subject_features(subject_id, X, y, metadata):
    """
    Save one subject's extracted (X, y, metadata) to a single .npz in
    DATA_OUTPUT_DIR.
    """
    if not os.path.exists(DATA_OUTPUT_DIR):
        os.makedirs(DATA_OUTPUT_DIR)
        logger.info(f"Created output directory {DATA_OUTPUT_DIR}.")

    out_path = os.path.join(DATA_OUTPUT_DIR, f"{subject_id}_features.npz")
    np.savez(out_path, X=X, y=y, **metadata)
    logger.info(
        f"Saved {X.shape[0]} windows ({X.shape[1]} features each) to {out_path}."
    )


if __name__ == "__main__":
    logger.info("Starting the main process")
    check_input_files(DATA_INPUT_DIR)

    for subject_id in sorted(os.listdir(DATA_INPUT_DIR)):
        subject_dir = os.path.join(DATA_INPUT_DIR, subject_id)
        if not os.path.isdir(subject_dir):
            continue

        fif_path, events_csv_path = find_subject_files(subject_dir)
        logger.info(f"Extracting features for subject {subject_id}")
        X, y, metadata = extract_features(fif_path, events_csv_path, subject_id)
        save_subject_features(subject_id, X, y, metadata)
