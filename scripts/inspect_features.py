"""
Standalone viewer for the *_features.npz files produced by src/main.py.
Independent of the extraction pipeline: only reads the .npz array names
(X, y, trial_index, window_start, window_end, media_filename), no import
of config/features_tools.

Usage:
    python scripts/inspect_features.py data/output/HZO024_features.npz
    python scripts/inspect_features.py data/output/HZO024_features.npz --show
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_emotion(media_filename):
    parts = os.path.normpath(str(media_filename)).split(os.sep)
    return parts[-2] if len(parts) >= 2 else "unknown"


def pca_2d(X):
    Xc = X - X.mean(axis=0)
    std = Xc.std(axis=0)
    std[std == 0] = 1.0
    Xc = Xc / std
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :2] * S[:2]


def print_summary(data):
    X, y, trial_index = data["X"], data["y"], data["trial_index"]
    windows_per_trial = np.bincount(trial_index)

    print(f"windows: {X.shape[0]}, features per window: {X.shape[1]}")
    print(f"trials: {len(windows_per_trial)}")
    print(
        "windows per trial: "
        f"min={windows_per_trial.min()}, max={windows_per_trial.max()}, "
        f"mean={windows_per_trial.mean():.1f}"
    )
    for name, col in zip(["valence", "arousal", "dominance"], y.T):
        print(f"{name}: min={col.min():.1f}, max={col.max():.1f}, mean={col.mean():.2f}")


def plot_overview(data, subject_id):
    X, y, trial_index = data["X"], data["y"], data["trial_index"]
    emotions = np.array([parse_emotion(m) for m in data["media_filename"]])
    unique_emotions = sorted(set(emotions.tolist()))
    cmap = plt.get_cmap("tab10")
    color_map = {e: cmap(i % 10) for i, e in enumerate(unique_emotions)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Feature overview — {subject_id}", fontsize=14)

    # 1. standardized feature heatmap
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xz = (X - X.mean(axis=0)) / std
    im = axes[0, 0].imshow(Xz, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    axes[0, 0].set_title("Standardized features (windows × features)")
    axes[0, 0].set_xlabel("feature index")
    axes[0, 0].set_ylabel("window index")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8)

    # 2. PCA projection colored by media emotion category
    proj = pca_2d(X)
    for e in unique_emotions:
        mask = emotions == e
        axes[0, 1].scatter(
            proj[mask, 0], proj[mask, 1], label=e, color=color_map[e], s=25, alpha=0.8
        )
    axes[0, 1].set_title("PCA of features, colored by media emotion")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    axes[0, 1].legend(fontsize=7, loc="best", ncol=2)

    # 3. valence/arousal circumplex, colored by dominance
    sc = axes[1, 0].scatter(y[:, 0], y[:, 1], c=y[:, 2], cmap="viridis", s=25)
    axes[1, 0].axvline(5, color="gray", linestyle="--", linewidth=0.8)
    axes[1, 0].axhline(5, color="gray", linestyle="--", linewidth=0.8)
    axes[1, 0].set_title("Valence vs Arousal (color = Dominance)")
    axes[1, 0].set_xlabel("Valence")
    axes[1, 0].set_ylabel("Arousal")
    axes[1, 0].set_xlim(0, 10)
    axes[1, 0].set_ylim(0, 10)
    fig.colorbar(sc, ax=axes[1, 0], shrink=0.8, label="Dominance")

    # 4. windows per trial
    windows_per_trial = np.bincount(trial_index)
    axes[1, 1].bar(range(len(windows_per_trial)), windows_per_trial, color="steelblue")
    axes[1, 1].set_title("Windows per trial")
    axes[1, 1].set_xlabel("trial index")
    axes[1, 1].set_ylabel("n windows")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", help="Path to a *_features.npz file")
    parser.add_argument(
        "--output", help="Where to save the overview figure (default: alongside the input file)"
    )
    parser.add_argument(
        "--show", action="store_true", help="Also display the figure interactively"
    )
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    subject_id = os.path.basename(args.npz_path).replace("_features.npz", "")

    print_summary(data)
    fig = plot_overview(data, subject_id)

    output_path = args.output or os.path.splitext(args.npz_path)[0] + "_overview.png"
    fig.savefig(output_path, dpi=150)
    print(f"saved figure to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
