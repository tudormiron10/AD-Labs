import time
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
from pyod.utils.utility import standardizer
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATASET_PATH = Path(__file__).parent / "shuttle.mat"
TEST_SIZE = 0.4
N_SPLITS = 10

DIF_ENSEMBLE_SIZE = 10
warnings.filterwarnings("ignore", message=".*pin_memory.*")


def load_shuttle_dataset() -> tuple[np.ndarray, np.ndarray]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    data = loadmat(DATASET_PATH)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"]).ravel().astype(int)
    return X, y


def model_factories(
    contamination: float,
    seed: int
) -> dict[str, Callable[[], object]]:
    return {
        "IForest": lambda: IForest(
            contamination=contamination,
            random_state=seed
        ),
        "LODA": lambda: LODA(
            contamination=contamination,
            random_state=seed
        ),
        "DIF": lambda: DIF(
            contamination=contamination,
            n_ensemble=DIF_ENSEMBLE_SIZE,
            random_state=seed,
        ),
    }


def evaluate_split(
    X: np.ndarray,
    y: np.ndarray,
    seed: int
) -> dict[str, tuple[float, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=seed, 
        stratify=y
    )
    X_train, X_test = standardizer(X_train, X_test)
    contamination = float(y_train.mean())

    results = {}
    for name, factory in model_factories(contamination, seed).items():
        model = factory()
        model.fit(X_train)
        balanced_accuracy = balanced_accuracy_score(y_test, model.predict(X_test))
        roc_auc = roc_auc_score(y_test, model.decision_function(X_test))
        results[name] = (balanced_accuracy, roc_auc)
    return results


def plot_results(results: dict[str, np.ndarray]) -> None:
    names = list(results)
    metrics = np.stack([results[name] for name in names])
    means = metrics.mean(axis=1)
    stds = metrics.std(axis=1)

    positions = np.arange(len(names))
    width = 0.35
    figure, axes = plt.subplots(figsize=(8, 5))
    for offset, (label, column) in zip((-width / 2, width / 2), (("BA", 0), ("ROC AUC", 1))):
        bars = axes.bar(
            positions + offset,
            means[:, column],
            width,
            yerr=stds[:, column],
            capsize=4,
            label=label,
        )
        axes.bar_label(bars, fmt="%.3f", padding=3)
    
    axes.set_xticks(positions, names)
    axes.set_ylim(0, 1.1)
    axes.set_ylabel("Mean over test splits")
    axes.set_title(f"Shuttle: mean over {N_SPLITS} train/test splits (±1 std)")
    axes.legend(loc="lower right")
    axes.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(Path(__file__).with_name("ex3.png"), dpi=150)
    plt.close(figure)


def main() -> None:
    X, y = load_shuttle_dataset()
    print(f"shuttle: X {X.shape}, outlier rate {y.mean():.4f}")

    per_split: dict[str, list[tuple[float, float]]] = {}
    for seed in range(N_SPLITS):
        start = time.perf_counter()
        split_results = evaluate_split(X, y, seed)
        
        for name, values in split_results.items():
            per_split.setdefault(name, []).append(values)
        
        summary = ", ".join(
            f"{name} BA={ba:.3f} AUC={auc:.3f}"
            for name, (ba, auc) in split_results.items()
        )
        print(f"split {seed + 1:2d} ({time.perf_counter() - start:5.1f}s): {summary}")

    results = {name: np.array(values) for name, values in per_split.items()}
    print(f"\nMean over {N_SPLITS} splits:")
    print(f"{'Model':<8} {'BA':>14} {'ROC AUC':>14}")
    for name, values in results.items():
        mean, std = values.mean(axis=0), values.std(axis=0)
        print(
            f"{name:<8} {mean[0]:.4f} ± {std[0]:.4f} {mean[1]:.4f} ± {std[1]:.4f}"
        )
    plot_results(results)


if __name__ == "__main__":
    main()
