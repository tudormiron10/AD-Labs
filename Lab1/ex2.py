from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


DATA_CONTAMINATION = 0.1
N_TRAIN = 400
N_TEST = 100
N_FEATURES = 2
MODEL_CONTAMINATIONS = (0.05, 0.1, 0.2)


def evaluate_model(
    X_train,
    y_train,
    X_test,
    y_test,
    contamination: float,
) -> dict[str, float | int]:
    model = KNN(contamination=contamination)
    model.fit(X_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix( y_test, test_predictions, labels=[0, 1]).ravel()
    test_scores = model.decision_function(X_test)

    return {
        "contamination": contamination,
        "train_outliers": int(train_predictions.sum()),
        "test_outliers": int(test_predictions.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_predictions),
        "auc": roc_auc_score(y_test, test_scores),
    }


def main() -> None:
    X_train, X_test, y_train, y_test = generate_data(
        n_train=N_TRAIN,
        n_test=N_TEST,
        n_features=N_FEATURES,
        contamination=DATA_CONTAMINATION,
        random_state=42,
    )

    results = [
        evaluate_model(X_train, y_train, X_test, y_test, contamination)
        for contamination in MODEL_CONTAMINATIONS
    ]
    baseline = results[MODEL_CONTAMINATIONS.index(DATA_CONTAMINATION)]

    model = KNN(contamination=DATA_CONTAMINATION)
    model.fit(X_train)
    test_predictions = model.predict(X_test)
    test_scores = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, test_scores)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    roc_axes, results_axes = axes
    roc_axes.plot(
        fpr,
        tpr,
        label=f"KNN (AUC = {baseline['auc']:.3f})",
        color="tab:blue",
    )
    roc_axes.plot(
        [0, 1],
        [0, 1],
        "--",
        color="tab:gray",
        label="Random classifier",
    )
    roc_axes.set_title("KNN ROC curve")
    roc_axes.set_xlabel("False positive rate")
    roc_axes.set_ylabel("True positive rate")
    roc_axes.legend()
    roc_axes.grid(alpha=0.25)

    results_axes.axis("off")
    results_axes.set_title("Evaluation results")
    results_axes.text(
        0.05,
        0.92,
        (
            f"Baseline contamination: {DATA_CONTAMINATION:.2f}\n"
            f"Balanced accuracy: {baseline['balanced_accuracy']:.3f}\n"
            f"AUC: {baseline['auc']:.3f}\n\n"
            f"Confusion matrix\n"
            f"TN={baseline['tn']}   FP={baseline['fp']}\n"
            f"FN={baseline['fn']}   TP={baseline['tp']}\n\n"
            "Effect of model contamination"
        ),
        va="top",
        fontsize=11,
    )
    results_axes.table(
        cellText=[
            [
                f"{result['contamination']:.2f}",
                f"{result['balanced_accuracy']:.3f}",
                f"{result['auc']:.3f}",
            ]
            for result in results
        ],
        colLabels=["Contamination", "BA", "AUC"],
        cellLoc="center",
        bbox=[0.05, 0.15, 0.9, 0.3],
    )
    figure.tight_layout()
    output_path = Path(__file__).with_name("ex2.png")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

if __name__ == "__main__":
    main()
