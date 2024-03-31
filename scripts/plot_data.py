"""Plot data statistics."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(path: str | Path) -> dict:
    """Load the data from the given path."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def load_synthetic_data(path: str | Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load synthetic data from a JSON file.

    :param path: Path to the JSON file.
    :return: A tuple containing the input data and labels.
    """
    data = load_data(path)
    X = [np.array(x) for x in data["X"]]
    y = [np.array(y) for y in data["y"]]
    return X, y


def get_statistics(X: list[np.ndarray], y: list[np.ndarray]) -> pd.DataFrame:
    """Get statistics of the given data.

    :param X: List of input data.
    :param y: List of labels.
    :return: A DataFrame containing the statistics.
    """
    n_users = len(X)
    n_samples = [len(x) for x in X]
    n_classes = [len(np.unique(y_)) for y_ in y]
    mean_x = [np.mean(x, axis=0) for x in X]
    std_x = [np.std(x, axis=0) for x in X]
    return pd.DataFrame(
        {
            "User": range(n_users),
            "Samples": n_samples,
            "Classes": n_classes,
            "Mean X": mean_x,
            "Std X": std_x,
        }
    )


def main():
    """Plot statistics of synthetic data."""
    data_dir = Path(__file__).parent.parent / "data"
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    for path in data_dir.glob("synthetic*.json"):
        X, y = load_synthetic_data(path)
        stats = get_statistics(X, y)
        save_path = plots_dir / path.stem
        save_path.mkdir(exist_ok=True)

        # Save statistics to a CSV file
        stats.to_csv(save_path / "statistics.csv", index=False)

        # Plot the number of samples per user
        plt.figure(figsize=(8, 6))
        stats.plot.hist(y="Samples", legend=False)
        plt.xlabel("Number of samples")
        plt.ylabel("Number of users")
        plt.title("Number of samples per user")
        plt.savefig(save_path / "samples_per_user.png")
        plt.close()

        # Plot the number of classes per user
        plt.figure(figsize=(8, 6))
        stats.plot.hist(y="Classes", legend=False)
        plt.xlabel("Number of classes")
        plt.ylabel("Number of users")
        plt.title("Number of classes per user")
        plt.savefig(save_path / "classes_per_user.png")
        plt.close()


if __name__ == "__main__":
    main()
