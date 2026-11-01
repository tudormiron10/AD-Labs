from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.datasets import make_blobs


CONTAMINATION = 0.07
NEIGHBORS = (5, 15, 30)


def main() -> None:
    X, _ = make_blobs(
        n_samples=(200, 100),
        centers=[(-10, -10), (10, 10)],
        cluster_std=[2, 6],
        n_features=2,
        random_state=42,
    )
    figure, axes = plt.subplots(
        len(NEIGHBORS),
        2,
        figsize=(12, 5 * len(NEIGHBORS)),
        squeeze=False,
    )
    for row, n_neighbors in enumerate(NEIGHBORS):
        for column, model_type in enumerate((KNN, LOF)):
            model = model_type(
                contamination=CONTAMINATION,
                n_neighbors=n_neighbors,
            )
            model.fit(X)
            predictions = model.predict(X)
            axes_item = axes[row, column]
            axes_item.scatter(
                X[predictions == 0, 0],
                X[predictions == 0, 1],
                color="tab:blue",
                label="Inlier",
                alpha=0.6,
            )
            axes_item.scatter(
                X[predictions == 1, 0],
                X[predictions == 1, 1],
                color="tab:red",
                label="Outlier",
                marker="x",
            )
            axes_item.set_title(f"{model_type.__name__} (k={n_neighbors})")
            axes_item.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(Path(__file__).with_name("ex3.png"), dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
