from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.combination import average, maximization
from pyod.models.knn import KNN
from pyod.utils.utility import standardizer
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


DATASET_PATH = Path(__file__).parent / "cardio.mat"
DATASET_SHAPE = (1831, 21)
CONTAMINATION = 176 / 1831
NEIGHBORS = np.linspace(30, 120, 10, dtype=int)


def load_cardio_dataset() -> tuple[np.ndarray, np.ndarray]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = loadmat(DATASET_PATH)
    if "X" not in data or "y" not in data:
        raise KeyError("cardio.mat must contain variables named 'X' and 'y'.")

    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"]).ravel().astype(int)
    if X.shape != DATASET_SHAPE or y.shape != (DATASET_SHAPE[0],):
        raise ValueError(
            "The cardio dataset must have X shape "
            f"{DATASET_SHAPE} and y shape ({DATASET_SHAPE[0]},)."
        )
    return X, y


def fit_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_scores = []
    test_scores = []
    train_metrics = []
    test_metrics = []

    for n_neighbors in NEIGHBORS:
        model = KNN(
            contamination=CONTAMINATION,
            n_neighbors=int(n_neighbors),
        )
        model.fit(X_train)
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        train_scores.append(model.decision_function(X_train))
        test_scores.append(model.decision_function(X_test))
        train_metrics.append(
            balanced_accuracy_score(y_train, train_predictions)
        )
        test_metrics.append(
            balanced_accuracy_score(y_test, test_predictions)
        )

    return (
        np.column_stack(train_scores),
        np.column_stack(test_scores),
        np.array(train_metrics),
        np.array(test_metrics),
    )


def evaluate_ensemble(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, tuple[float, float, float]]:
    train_scores, test_scores = standardizer(train_scores, test_scores)
    ensemble_scores = {
        "Average": (average(train_scores), average(test_scores)),
        "Maximum": (maximization(train_scores), maximization(test_scores)),
    }
    results = {}
    for name, (train_score, test_score) in ensemble_scores.items():
        threshold = np.quantile(train_score, 1 - CONTAMINATION)
        train_predictions = (train_score > threshold).astype(int)
        test_predictions = (test_score > threshold).astype(int)
        results[name] = (
            float(threshold),
            balanced_accuracy_score(y_train, train_predictions),
            balanced_accuracy_score(y_test, test_predictions),
        )
    return results


def main() -> None:
    X, y = load_cardio_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    X_train, X_test = standardizer(X_train, X_test)
    train_scores, test_scores, train_metrics, test_metrics = fit_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    ensemble_results = evaluate_ensemble(
        train_scores,
        test_scores,
        y_train,
        y_test,
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(
        NEIGHBORS,
        train_metrics,
        marker="o",
        label="Train",
    )
    axes[0].plot(
        NEIGHBORS,
        test_metrics,
        marker="o",
        label="Test",
    )
    axes[0].set_title("Individual KNN models")
    axes[0].set_xlabel("n_neighbors")
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_xticks(NEIGHBORS)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].axis("off")
    axes[1].set_title("Evaluation results")
    table_data = [
        [
            str(n_neighbors),
            f"{train_metric:.4f}",
            f"{test_metric:.4f}",
        ]
        for n_neighbors, train_metric, test_metric in zip(
            NEIGHBORS,
            train_metrics,
            test_metrics,
        )
    ]
    individual_table = axes[1].table(
        cellText=table_data,
        colLabels=["n_neighbors", "Train BA", "Test BA"],
        cellLoc="center",
        bbox=[0.02, 0.42, 0.96, 0.48],
    )
    individual_table.auto_set_font_size(False)
    individual_table.set_fontsize(8)
    ensemble_table = axes[1].table(
        cellText=[
            [
                name,
                f"{threshold:.4f}",
                f"{train_metric:.4f}",
                f"{test_metric:.4f}",
            ]
            for name, (threshold, train_metric, test_metric) in ensemble_results.items()
        ],
        colLabels=["Strategy", "Threshold", "Train BA", "Test BA"],
        cellLoc="center",
        bbox=[0.02, 0.08, 0.96, 0.22],
    )
    ensemble_table.auto_set_font_size(False)
    ensemble_table.set_fontsize(8)

    figure.tight_layout()
    figure.savefig(Path(__file__).with_name("ex4.png"), dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
