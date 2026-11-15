from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


RANDOM_STATE = 42
N_TRAIN = 500
N_TEST = 500
N_PROJECTIONS = 5
BIN_COUNTS = (1, 3, 5, 10, 20, 50, 100, 200)
N_COLUMNS = 4
RANGE_PADDING = 1.0
PROBABILITY_FLOOR = 1 / N_TRAIN


def random_projections(
    rng: np.random.Generator,
    n_projections: int
) -> np.ndarray:
    vectors = rng.multivariate_normal(
        mean=(0, 0), cov=np.eye(2), size=n_projections
    )
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def fit_histograms(
    X: np.ndarray,
    projections: np.ndarray,
    n_bins: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    histograms = []
    for projected in (X @ projections.T).T:
        value_range = (
            projected.min() - RANGE_PADDING,
            projected.max() + RANGE_PADDING,
        )
        counts, edges = np.histogram(projected, bins=n_bins, range=value_range)
        histograms.append((counts / counts.sum(), edges))
    return histograms


def mean_bin_probability(
    X: np.ndarray,
    projections: np.ndarray,
    histograms: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    probabilities = np.zeros((len(X), len(histograms)))
    for index, (projected, (bin_probabilities, edges)) in enumerate(
        zip((X @ projections.T).T, histograms)
    ):
        bin_indices = np.digitize(projected, edges) - 1
        inside = (bin_indices >= 0) & (bin_indices < len(bin_probabilities))
        probabilities[inside, index] = bin_probabilities[bin_indices[inside]]
    return probabilities.mean(axis=1)


def anomaly_scores(mean_probabilities: np.ndarray) -> np.ndarray:
    return -np.log(mean_probabilities + PROBABILITY_FLOOR)


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    
    X_train, _ = make_blobs(
        n_samples=N_TRAIN,
        n_features=2,
        centers=[(0, 0)],
        cluster_std=1.0,
        random_state=RANDOM_STATE,
    )
    X_test = rng.uniform(low=-3, high=3, size=(N_TEST, 2))
    projections = random_projections(rng, N_PROJECTIONS)

    n_rows = int(np.ceil(len(BIN_COUNTS) / N_COLUMNS))
    figure, axes = plt.subplots(
        n_rows, N_COLUMNS, figsize=(5 * N_COLUMNS, 4.5 * n_rows), squeeze=False
    )
    for axes_item, n_bins in zip(axes.flat, BIN_COUNTS):
        histograms = fit_histograms(X_train, projections, n_bins)
        scores = anomaly_scores(
            mean_bin_probability(X_test, projections, histograms)
        )
        points = axes_item.scatter(
            X_test[:, 0], X_test[:, 1], c=scores, cmap="YlOrRd", s=18
        )
        figure.colorbar(
            points, ax=axes_item, label="Anomaly score = -log(mean bin probability)"
        )
        axes_item.set_title(f"{n_bins} bins")
        axes_item.set_aspect("equal")
        axes_item.grid(alpha=0.25)
    figure.suptitle(f"Simple LODA, {N_PROJECTIONS} random projections")
    figure.tight_layout()
    figure.savefig(Path(__file__).with_name("ex1.png"), dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
