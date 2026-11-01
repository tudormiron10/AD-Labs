from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.utils.data import generate_data_clusters
from sklearn.metrics import balanced_accuracy_score


CONTAMINATION = 0.1
N_TRAIN = 400
N_TEST = 200
N_FEATURES = 2
NEIGHBORS = (1, 3, 5)


def main() -> None:
    X_train, X_test, y_train, y_test = generate_data_clusters(
        n_train=N_TRAIN,
        n_test=N_TEST,
        n_clusters=2,
        n_features=N_FEATURES,
        contamination=CONTAMINATION,
        random_state=42,
    )
    models = [KNN(contamination=CONTAMINATION, n_neighbors=k) for k in NEIGHBORS]
    for model in models:
        model.fit(X_train)

    figure, axes = plt.subplots(
        len(NEIGHBORS),
        4,
        figsize=(16, 5 * len(NEIGHBORS)),
        squeeze=False,
    )
    for row, (model, n_neighbors) in enumerate(zip(models, NEIGHBORS)):
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        plots = (
            (X_train, y_train, "Training ground truth"),
            (X_train, train_predictions, "Training prediction"),
            (X_test, y_test, "Test ground truth"),
            (X_test, test_predictions, "Test prediction"),
        )
        for column, (data, labels, title) in enumerate(plots):
            axes_item = axes[row, column]
            axes_item.scatter(
                data[labels == 0, 0],
                data[labels == 0, 1],
                color="tab:blue",
                label="Inlier",
                alpha=0.6,
            )
            axes_item.scatter(
                data[labels == 1, 0],
                data[labels == 1, 1],
                color="tab:red",
                label="Outlier",
                marker="x",
            )
            axes_item.set_title(f"{title} (k={n_neighbors})")
            axes_item.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.tight_layout()
    output_path = Path(__file__).with_name("ex2.png")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    metrics_figure, metrics_axes = plt.subplots(figsize=(7, 4))
    train_metrics = [
        balanced_accuracy_score(y_train, model.predict(X_train)) for model in models
    ]
    test_metrics = [
        balanced_accuracy_score(y_test, model.predict(X_test)) for model in models
    ]
    metrics_axes.plot(NEIGHBORS, train_metrics, marker="o", label="Train")
    metrics_axes.plot(NEIGHBORS, test_metrics, marker="o", label="Test")
    metrics_axes.set_xlabel("n_neighbors")
    metrics_axes.set_ylabel("Balanced accuracy")
    metrics_axes.set_title("KNN parameter comparison")
    metrics_axes.set_xticks(NEIGHBORS)
    metrics_axes.legend()
    metrics_axes.grid(alpha=0.25)
    metrics_figure.tight_layout()
    metrics_figure.savefig(
        Path(__file__).with_name("ex2_metrics.png"),
        dpi=150,
    )
    plt.close(metrics_figure)


if __name__ == "__main__":
    main()
