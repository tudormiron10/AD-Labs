from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


CONTAMINATION = 0.1
N_TRAIN = 1000
N_FEATURES = 1


def generate_data(
    n_train: int,
    n_features: int,
    contamination: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_outliers = round(n_train * contamination)
    n_inliers = n_train - n_outliers

    inliers = rng.normal(0, 1, size=(n_inliers, n_features))
    outliers = rng.normal(5, 1, size=(n_outliers, n_features))
    values = np.vstack((inliers, outliers))
    labels = np.concatenate((np.zeros(n_inliers, dtype=int), np.ones(n_outliers, dtype=int)))

    order = rng.permutation(n_train)
    return values[order], labels[order]


def main() -> None:
    X_train, y_train = generate_data(
        n_train=N_TRAIN,
        n_features=N_FEATURES,
        contamination=CONTAMINATION,
        random_state=42,
    )
    values = X_train[:, 0]

    mean = np.mean(values)
    standard_deviation = np.std(values)
    z_scores = np.abs((values - mean) / standard_deviation)
    threshold = np.quantile(z_scores, 1 - CONTAMINATION)
    predictions = (z_scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_train, predictions, labels=[0, 1]).ravel()
    balanced_accuracy = balanced_accuracy_score(y_train, predictions)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    data_axes, results_axes = axes
    sample_indices = np.arange(N_TRAIN)
    data_axes.scatter(
        sample_indices[y_train == 0],
        values[y_train == 0],
        color="tab:blue",
        label="Normal",
        alpha=0.55,
        s=18,
    )
    data_axes.scatter(
        sample_indices[y_train == 1],
        values[y_train == 1],
        color="tab:red",
        label="True outlier",
        marker="x",
        s=30,
    )
    data_axes.scatter(
        sample_indices[predictions == 1],
        values[predictions == 1],
        facecolors="none",
        edgecolors="black",
        label="Z-score prediction",
        s=65,
    )
    data_axes.set_xlabel("Sample index")
    data_axes.set_ylabel("Value")
    data_axes.grid(alpha=0.25)

    results_axes.axis("off")
    results_axes.set_title("Evaluation results")
    results_axes.text(
        0.08,
        0.88,
        (
            f"Absolute Z-score threshold: {threshold:.3f}\n"
            f"Balanced accuracy: {balanced_accuracy:.3f}\n\n"
            "Confusion matrix\n"
            f"TN={tn}   FP={fp}\n"
            f"FN={fn}   TP={tp}"
        ),
        va="top",
        fontsize=12,
    )
    handles, labels = data_axes.get_legend_handles_labels()
    results_axes.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.05, 0.05),
        frameon=True,
    )
    figure.tight_layout()
    output_path = Path(__file__).with_name("ex3.png")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

if __name__ == "__main__":
    main()
