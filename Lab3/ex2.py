import warnings
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.dif import DIF
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from sklearn.datasets import make_blobs


RANDOM_STATE = 42
CONTAMINATION = 0.02
N_PER_CLUSTER = 500
N_TEST = 1000
TEST_RANGE = (-10, 20)
CENTERS_2D = [(10, 0), (0, 10)]
CENTERS_3D = [(0, 10, 0), (10, 0, 10)]
DIF_HIDDEN_LAYERS = ([16], [64, 32], [500, 100])
LODA_BIN_COUNTS = (5, 10, 50)

ModelFactory = Callable[[], object]

warnings.filterwarnings("ignore", message=".*pin_memory.*")


def make_data(
    centers: list[tuple[int, ...]],
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    X_train, _ = make_blobs(
        n_samples=[N_PER_CLUSTER] * len(centers),
        n_features=len(centers[0]),
        centers=centers,
        cluster_std=1.0,
        random_state=RANDOM_STATE,
    )
    X_test = rng.uniform(*TEST_RANGE, size=(N_TEST, len(centers[0])))
    return X_train, X_test


def default_models() -> dict[str, ModelFactory]:
    return {
        "IForest": lambda: IForest(
            contamination=CONTAMINATION, random_state=RANDOM_STATE
        ),
        "DIF": lambda: DIF(
            contamination=CONTAMINATION, random_state=RANDOM_STATE
        ),
        "LODA": lambda: LODA(
            contamination=CONTAMINATION, random_state=RANDOM_STATE
        ),
    }


def variation_models() -> dict[str, ModelFactory]:
    models: dict[str, ModelFactory] = {}
    
    for hidden in DIF_HIDDEN_LAYERS:
        models[f"DIF hidden={hidden}"] = lambda hidden=hidden: DIF(
            contamination=CONTAMINATION,
            hidden_neurons=hidden,
            random_state=RANDOM_STATE,
        )
    
    for n_bins in LODA_BIN_COUNTS:
        models[f"LODA n_bins={n_bins}"] = lambda n_bins=n_bins: LODA(
            contamination=CONTAMINATION,
            n_bins=n_bins,
            random_state=RANDOM_STATE
        )
    return models


def score_models(
    models: dict[str, ModelFactory],
    X_train: np.ndarray,
    X_test: np.ndarray
) -> dict[str, np.ndarray]:
    scores = {}
    for name, factory in models.items():
        model = factory()
        model.fit(X_train)
        scores[name] = model.decision_function(X_test)
    return scores


def plot_scores(
    scores: dict[str, np.ndarray],
    X_train: np.ndarray,
    X_test: np.ndarray,
    output_name: str,
    title: str,
    n_columns: int = 3,
) -> None:
    is_3d = X_test.shape[1] == 3
    n_rows = int(np.ceil(len(scores) / n_columns))
    figure = plt.figure(figsize=(6 * n_columns, 5 * n_rows))
    
    for index, (name, values) in enumerate(scores.items(), start=1):
        if is_3d:
            axes_item = figure.add_subplot(n_rows, n_columns, index, projection="3d")
            points = axes_item.scatter(
                *X_test.T, c=values, cmap="viridis", s=10, alpha=0.8
            )
            axes_item.scatter(*X_train.T, color="black", s=1, alpha=0.15)
        else:
            axes_item = figure.add_subplot(n_rows, n_columns, index)
            points = axes_item.scatter(*X_test.T, c=values, cmap="viridis", s=14)
            axes_item.scatter(*X_train.T, color="black", s=2, alpha=0.2)
            axes_item.set_aspect("equal")
            axes_item.grid(alpha=0.25)
        figure.colorbar(points, ax=axes_item, label="Anomaly score", shrink=0.8)
        axes_item.set_title(name)
    
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(Path(__file__).with_name(output_name), dpi=150)
    plt.close(figure)


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    X_train, X_test = make_data(CENTERS_2D, rng)
    plot_scores(
        score_models(default_models(), X_train, X_test),
        X_train,
        X_test,
        "ex2_2d.png",
        "2D anomaly scores (black: training data)",
    )
    plot_scores(
        score_models(variation_models(), X_train, X_test),
        X_train,
        X_test,
        "ex2_variations.png",
        "DIF hidden layers (top) and LODA bin counts (bottom)",
    )

    X_train_3d, X_test_3d = make_data(CENTERS_3D, rng)
    plot_scores(
        score_models(default_models(), X_train_3d, X_test_3d),
        X_train_3d,
        X_test_3d,
        "ex2_3d.png",
        "3D anomaly scores (black: training data)",
    )


if __name__ == "__main__":
    main()
