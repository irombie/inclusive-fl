"""Generate synthetic data for federated learning experiments.

This script is based on the GitHub repository fair_flearn. The original code is available at
https://github.com/litian96/fair_flearn/blob/master/data/synthetic/generate_synthetic.py
"""

from pathlib import Path
import argparse
import json

import numpy as np


PROJECT_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT_DIR / "data"
DEFAULT_OUTPUT_PATH: Path = DATA_DIR / "synthetic_data.json"


def get_parser():
    """Get the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output file.",
    )
    parser.add_argument("--n_samples", type=int, help="Total number of samples.")
    parser.add_argument("--n_users", type=int, default=100, help="Number of users.")
    parser.add_argument("--n_classes", type=int, default=10, help="Number of classes.")
    parser.add_argument("--n_dims", type=int, default=60, help="Number of dimensions.")
    parser.add_argument(
        "--majority_proportion",
        type=float,
        help="Proportion of majority class samples.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        help="Overlap between the majority and minority class samples.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Variance of the prior distribution of the weights.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Variance of the prior distribution of the bias.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for the random number generator."
    )
    return parser


def softmax(z: np.ndarray) -> np.ndarray:
    r"""Compute the softmax function.

    :param z: Input vector.
    """
    z_exp = np.exp(z - np.max(z))
    return z_exp / np.sum(z_exp)


def generate_synthetic(
    n_samples: int,
    n_users: int,
    n_classes: int,
    n_dims: int,
    majority_proportion: float,
    overlap: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    seed: int = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    r"""Generate synthetic data for federated learning experiments.

    :param n_samples: Total number of samples.
    :param n_users: Number of users.
    :param n_classes: Number of classes.
    :param n_dims: Number of dimensions.
    :param majority_proportion: Proportion of majority class samples.
    :param overlap: Overlap between the majority and minority class samples.
    :param alpha: Variance of the prior distribution of the weights.
    :param beta: Variance of the prior distribution of the bias.
    :param seed: Seed for the random number generator.
    """
    print("Generating synthetic data with the following parameters:")
    print(f"Number of users: {n_users}")
    print(f"Number of classes: {n_classes}")
    print(f"Number of dimensions: {n_dims}")
    print(f"Proportion of majority class samples: {majority_proportion}")
    print(f"Overlap between the majority and minority class samples: {overlap}")
    print(f"Variance of the prior distribution of the weights: {alpha}")
    print(f"Variance of the prior distribution of the bias: {beta}")
    print(f"Seed: {seed}")

    rng = np.random.default_rng(seed)

    # Compute the number of samples per user
    n_majority_classes = int(n_classes * majority_proportion)
    n_majority_users = int(n_users * majority_proportion)
    n_minority_users = n_users - n_majority_users

    # Get the distribution of samples per user
    dist_uniform = np.full((n_classes, n_users), 1. / n_users)
    dist_majority = np.zeros_like(dist_uniform)
    dist_majority[:n_majority_classes, :n_majority_users] = 1. / n_majority_users
    dist_majority[n_majority_classes:, n_majority_users:] = 1. / n_minority_users
    dist = (1 - overlap) * dist_majority + overlap * dist_uniform

    assert np.allclose(dist.sum(axis=1), 1), "The distribution is not normalized."

    # Define weights and biases
    W = rng.normal(0, alpha, (n_dims, n_classes))
    b = rng.normal(0, beta, (n_classes))
    mean_x = rng.normal(0, beta, n_dims)
    cov_x = np.diag(np.power(np.arange(1, n_dims + 1), -1.2))

    # Generate data
    x = rng.multivariate_normal(mean_x, cov_x, n_samples)
    y: np.ndarray = softmax(x @ W + b).argmax(axis=1)

    # Get samples indices per class
    sample_indices_by_class = [np.where(y == i)[0] for i in range(n_classes)]

    # Split data into users
    X_per_user = [[] for _ in range(n_users)]
    y_per_user = [[] for _ in range(n_users)]

    for sample_indices, cls_dist in zip(sample_indices_by_class, dist):
        user_indices = rng.choice(n_users, len(sample_indices), p=cls_dist)
        print(user_indices.shape)
        for user_idx, sample_idx in zip(user_indices, sample_indices):
            X_per_user[user_idx].append(x[sample_idx].tolist())
            y_per_user[user_idx].append(y[sample_idx].tolist())

    return X_per_user, y_per_user


def save_synthetic_data(X: list[np.ndarray], y: list[np.ndarray], path: str | Path):
    """Save synthetic data to a json file.

    :param X: List of input data.
    :param y: List of labels.
    :param path: Path to the output file.
    """
    path = Path(path).resolve()
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    data = {"X": X, "y": y}
    with open(path, mode="x", encoding="utf-8") as f:
        json.dump(data, f)


def check_output_path(output_path: str):
    """Check if the output path exists and is writable.

    :param output_path: Path to the output file.
    """
    output_path: Path = Path(output_path)
    if output_path.exists():
        raise ValueError(f"{output_path} already exists.")
    if output_path.suffix != ".json":
        raise ValueError(f"Output file must be a JSON file. Got: {output_path}")


def main():
    """Generate synthetic data for federated learning experiments.

    This function is called when the script is run from the command line.
    Generates synthetic data and saves it to a JSON file.
    """
    parser = get_parser()
    args = parser.parse_args()
    check_output_path(args.output_path)
    X, y = generate_synthetic(
        args.n_samples,
        args.n_users,
        args.n_classes,
        args.n_dims,
        args.majority_proportion,
        args.overlap,
        args.alpha,
        args.beta,
        args.seed,
    )
    save_synthetic_data(X, y, args.output_path)


if __name__ == "__main__":
    main()
