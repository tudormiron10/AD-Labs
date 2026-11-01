from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RANDOM_STATE = 42
NOISE_VARIANCES = (0.25, 1.0, 4.0)
N_SAMPLES = 100
CASE_COLORS = ("tab:blue", "tab:orange", "tab:green")


def leverage_scores(features: np.ndarray) -> np.ndarray:
    hat_matrix = features @ np.linalg.pinv(features.T @ features) @ features.T
    return np.diag(hat_matrix)


def plot_case(
    axes: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    title: str,
    color: str,
    label: str,
) -> None:
    scores = leverage_scores(features)
    highest = np.argsort(scores)[-5:]
    axes.scatter(x, y, color=color, s=16, alpha=0.65, label=label)
    axes.scatter(
        x[highest],
        y[highest],
        facecolors="none",
        edgecolors="black",
        s=75,
        linewidths=1.2,
    )
    axes.set_title(title)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.grid(alpha=0.25)


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    figures = []

    x = np.linspace(-5, 5, N_SAMPLES)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    cases = (
        ("Regular", 0.0, 0.0),
        ("High variance on x", 1.0, 0.0),
        ("High variance on y", 0.0, 1.0),
        ("High variance on x and y", 1.0, 1.0),
    )
    for axes_item, (title, x_noise, y_noise) in zip(axes.flat, cases):
        for variance, color in zip(NOISE_VARIANCES, CASE_COLORS):
            x_case = x + rng.normal(0, x_noise * np.sqrt(variance), N_SAMPLES)
            y_case = x + 1 + rng.normal(0, y_noise * np.sqrt(variance), N_SAMPLES)
            features = np.column_stack((x_case, np.ones(N_SAMPLES)))
            
            plot_case(
                axes_item,
                x_case,
                y_case,
                features,
                title,
                color,
                f"Variance {variance}",
            )
    axes[0, 0].legend()
    figure.tight_layout()
    figures.append((figure, Path(__file__).with_name("ex1_1d.png")))

    x1 = rng.uniform(-5, 5, N_SAMPLES)
    x2 = rng.uniform(-5, 5, N_SAMPLES)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    cases_2d = (
        ("Regular", 0.0, 0.0),
        ("High variance on x1", 1.0, 0.0),
        ("High variance on x2", 0.0, 1.0),
        ("High variance on x1 and x2", 1.0, 1.0),
    )
    for axes_item, (title, x1_noise, x2_noise) in zip(axes.flat, cases_2d):
        for variance, color in zip(NOISE_VARIANCES, CASE_COLORS):
            x1_case = x1 + rng.normal(0, x1_noise * np.sqrt(variance), N_SAMPLES)
            x2_case = x2 + rng.normal(0, x2_noise * np.sqrt(variance), N_SAMPLES)
            y_case = x1 + x2 + 1 + rng.normal(0, np.sqrt(variance), N_SAMPLES)
            features_2d = np.column_stack((x1_case, x2_case, np.ones(N_SAMPLES)))
            
            plot_case(
                axes_item,
                x1_case,
                y_case,
                features_2d,
                title,
                color,
                f"Variance {variance}",
            )
        axes_item.set_xlabel("x1")
        axes_item.set_ylabel("y")
    axes[0, 0].legend()
    figure.tight_layout()
    figures.append((figure, Path(__file__).with_name("ex1_2d.png")))

    for figure, output_path in figures:
        figure.savefig(output_path, dpi=150)
        plt.close(figure)


if __name__ == "__main__":
    main()
