from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_data(
    n_train: int,
    n_test: int,
    n_features: int,
    contamination: float,
    random_state: int,
):
    rng = np.random.default_rng(random_state)

    def make_split(size: int):
        n_outliers = round(size * contamination)
        n_normal = size - n_outliers
        normal = rng.normal(0, 1, (n_normal, n_features))
        outliers = rng.uniform(-6, 6, (n_outliers, n_features))
        features = np.vstack((normal, outliers))
        labels = np.concatenate(
            (np.zeros(n_normal, dtype=int), np.ones(n_outliers, dtype=int))
        )
        order = rng.permutation(size)
        return features[order], labels[order]

    X_train, y_train = make_split(n_train)
    X_test, y_test = make_split(n_test)
    return X_train, X_test, y_train, y_test


CONTAMINATION = 0.1
N_TRAIN = 400
N_TEST = 100
N_FEATURES = 2


def main() -> None:
    X_train, _, y_train, _ = generate_data(
        n_train=N_TRAIN,
        n_test=N_TEST,
        n_features=N_FEATURES,
        contamination=CONTAMINATION,
        random_state=42,
    )

    normal = y_train == 0
    outlier = y_train == 1

    figure, axes = plt.subplots(figsize=(6, 6))
    axes.scatter(
        X_train[normal, 0],
        X_train[normal, 1],
        c="tab:blue",
        label="Normal",
        alpha=0.7,
    )
    
    axes.scatter(
        X_train[outlier, 0],
        X_train[outlier, 1],
        c="tab:red",
        label="Outlier",
        alpha=0.7,
        marker="x",
    )
    axes.set_xlabel("Feature 1")
    axes.set_ylabel("Feature 2")
    axes.grid(alpha=0.25)

    output_path = Path(__file__).with_name("ex1.png")
    figure.savefig(output_path)


if __name__ == "__main__":
    main()
