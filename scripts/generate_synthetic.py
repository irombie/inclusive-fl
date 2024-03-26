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
    parser.add_argument("--n_users", type=int, default=100, help="Number of users.")
    parser.add_argument("--n_classes", type=int, default=10, help="Number of classes.")
    parser.add_argument("--n_dims", type=int, default=60, help="Number of dimensions.")
    parser.add_argument(
        "--min_samples_per_user",
        type=int,
        default=50,
        help="Minimum number of samples per user.",
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
        "--iid",
        action="store_true",
        help="Whether the weights and biases are generated identically distributed or not.",
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
    n_users: int,
    n_classes: int,
    n_dims: int,
    min_samples_per_user: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    iid: bool = False,
    seed: int = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    r"""Generate synthetic data for federated learning experiments.

    :param n_users: Number of users.
    :param n_classes: Number of classes.
    :param n_dims: Number of dimensions.
    :param min_samples_per_user: Minimum number of samples per user. The actual number of samples per user will be
    sampled as :math:`\text{samples_per_user} = \text{np.random.lognormal}(4, 2, (n\_users)) + \text{min_samples_per_user}`.
    :param alpha: Variance of the prior distribution of the weights.
    :param beta: Variance of the prior distribution of the bias.
    :param iid: Whether the weights and biases are generated identically distributed or not. Note that the weights and
    biases are generated independently in any case.
    :param seed: Seed for the random number generator.
    """
    print("Generating synthetic data with the following parameters:")
    print(f"Number of users: {n_users}")
    print(f"Number of classes: {n_classes}")
    print(f"Number of dimensions: {n_dims}")
    print(f"Minimum number of samples per user: {min_samples_per_user}")
    print(f"Variance of the prior distribution of the weights: {alpha}")
    print(f"Variance of the prior distribution of the bias: {beta}")
    print(f"Identically distributed weights and biases: {iid}")
    print(f"Seed: {seed}")

    rng = np.random.default_rng(seed)

    samples_per_user = min_samples_per_user + rng.lognormal(4, 2, n_users).astype(int)

    # Define weights and biases
    mean_of_mean_x = rng.normal(0, beta, n_users)
    cov_x = np.diag(np.power(np.arange(1, n_dims + 1), -1.2))
    if iid:
        W = rng.normal(0, alpha, (n_users, n_dims, n_classes))
        b = rng.normal(0, beta, (n_users, n_classes))
        mean_x = mean_of_mean_x[:, np.newaxis] * np.ones((n_users, n_dims))
    else:
        mean_W = rng.normal(0, alpha, n_users)
        mean_b = mean_W
        W = rng.normal(
            mean_W[:, np.newaxis, np.newaxis], 1, (n_users, n_dims, n_classes)
        )
        b = rng.normal(mean_b[:, np.newaxis], 1, (n_users, n_classes))
        mean_x = rng.normal(mean_of_mean_x[:, np.newaxis], 1, (n_users, n_dims))

    # Generate data
    X_split, y_split = [], []
    for i in range(n_users):
        x = rng.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        y: np.ndarray = softmax(x @ W[i] + b[i]).argmax(axis=1)
        X_split.append(x.tolist())
        y_split.append(y.tolist())
    return X_split, y_split


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
        args.n_users,
        args.n_classes,
        args.n_dims,
        args.min_samples_per_user,
        args.alpha,
        args.beta,
        args.iid,
        args.seed,
    )
    save_synthetic_data(X, y, args.output_path)


if __name__ == "__main__":
    main()
