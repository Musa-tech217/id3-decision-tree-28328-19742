"""Dataset loading utilities for the ID3 decision tree project."""

import pandas as pd


TENNIS_URL = "https://raw.githubusercontent.com/lmassaron/datasets/master/tennis.csv"


def _clean_column_names(dataframe):
    """Return a copy with whitespace removed from column names."""
    cleaned = dataframe.copy()
    cleaned.columns = cleaned.columns.str.strip()
    return cleaned


def load_tennis():
    """Load the Tennis dataset from the public CSV source."""
    tennis = pd.read_csv(TENNIS_URL)
    return _clean_column_names(tennis)


def load_sunburn():
    """Create and return the Sunburn dataset as a pandas DataFrame."""
    data = [
        ["Sarah", "Blonde", "Average", "Light", "No", "Sunburnt"],
        ["Dana", "Blonde", "Tall", "Medium", "Yes", "None"],
        ["Alex", "Brown", "Short", "Medium", "Yes", "None"],
        ["Annie", "Blonde", "Short", "Medium", "No", "Sunburnt"],
        ["Emily", "Red", "Average", "Heavy", "No", "Sunburnt"],
        ["Pete", "Brown", "Tall", "Heavy", "No", "None"],
        ["John", "Brown", "Medium", "Heavy", "No", "None"],
        ["Kate", "Blonde", "Short", "Light", "Yes", "None"],
    ]
    columns = ["Name", "Hair", "Height", "Weight", "Lotion", "Burnt"]
    return pd.DataFrame(data, columns=columns)


def load_iris_dataset():
    """Load the Iris dataset as a pandas DataFrame."""
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    dataframe = iris.frame.copy()
    dataframe["target_name"] = dataframe["target"].map(
        dict(enumerate(iris.target_names))
    )
    return dataframe


def _bin_labels(requested_bins, actual_bins):
    """Return readable labels for the number of bins qcut produced."""
    if requested_bins == 3:
        three_bin_labels = {
            1: ["low"],
            2: ["low", "high"],
            3: ["low", "medium", "high"],
        }
        if actual_bins in three_bin_labels:
            return three_bin_labels[actual_bins]

    return [f"bin_{index}" for index in range(actual_bins)]


def discretize(data, attributes, num_bins=3):
    """Return a copy with selected numerical attributes converted to bins."""
    discretized = data.copy()

    for attribute in attributes:
        non_missing = discretized[attribute].dropna()
        if non_missing.nunique() <= 1:
            label = _bin_labels(num_bins, 1)[0]
            discretized[attribute] = pd.Categorical(
                [label if pd.notna(value) else pd.NA for value in discretized[attribute]],
                categories=[label],
            )
            continue

        binned = pd.qcut(discretized[attribute], q=num_bins, duplicates="drop")
        actual_bins = len(binned.cat.categories)
        labels = _bin_labels(num_bins, actual_bins)
        discretized[attribute] = pd.Categorical.from_codes(
            binned.cat.codes,
            categories=labels,
        )

    return discretized
