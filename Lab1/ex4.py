from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


CONTAMINATION = 0.1
N_SAMPLES = 1000
N_FEATURES = 2
MEAN = np.array([2.0, -1.0])
COVARIANCE = np.array([[2.0, 1.0], [1.0, 2.0]])


def main() -> None:
    rng = np.random.default_rng(42)
    n_outliers = int(N_SAMPLES * CONTAMINATION)
    n_normal = N_SAMPLES - n_outliers

    standard_normal_samples = rng.standard_normal((n_normal, N_FEATURES))
    cholesky_factor = np.linalg.cholesky(COVARIANCE)
    normal_samples = standard_normal_samples @ cholesky_factor.T + MEAN

    outlier_samples = rng.uniform(
        low=MEAN - 10,
        high=MEAN + 10,
        size=(n_outliers, N_FEATURES),
    )
    X = np.vstack((normal_samples, outlier_samples))
    y = np.concatenate((np.zeros(n_normal, dtype=int), np.ones(n_outliers, dtype=int)))

    centered = X - MEAN
    whitened = np.linalg.solve(cholesky_factor, centered.T).T
    MD = np.linalg.norm(whitened, axis=1)
    threshold = np.quantile(MD, 1 - CONTAMINATION)
    predictions = (MD > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    balanced_accuracy = balanced_accuracy_score(y, predictions)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    data_axes, results_axes = axes
    data_axes.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color="tab:blue",
        label="Normal",
        alpha=0.45,
        s=18,
    )
    data_axes.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color="tab:red",
        label="True outlier",
        marker="x",
        s=30,
    )
    data_axes.scatter(
        X[predictions == 1, 0],
        X[predictions == 1, 1],
        facecolors="none",
        edgecolors="black",
        label="Mahalanobis prediction",
        s=65,
    )
    data_axes.set_xlabel("Feature 1")
    data_axes.set_ylabel("Feature 2")
    data_axes.grid(alpha=0.25)

    results_axes.axis("off")
    results_axes.set_title("Evaluation results")
    results_axes.text(
        0.08,
        0.88,
        (
            f"Mahalanobis threshold: {threshold:.3f}\n"
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
    output_path = Path(__file__).with_name("ex4.png")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

if __name__ == "__main__":
    main()
