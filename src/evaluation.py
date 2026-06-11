"""Prediction and evaluation utilities for ID3 trees."""

from time import perf_counter

import numpy as np
import pandas as pd

from src.id3 import id3, majority_class, tree_size


def predict(tree, instance, default=None):
    """Predict one label by traversing a nested-dictionary ID3 tree."""
    current_tree = tree

    while isinstance(current_tree, dict):
        if len(current_tree) != 1:
            return default

        attribute = next(iter(current_tree))
        branches = current_tree[attribute]

        if attribute not in instance:
            return default

        value = instance[attribute]
        if value not in branches:
            return default

        current_tree = branches[value]

    return current_tree


def predict_many(tree, data, default=None):
    """Predict labels for each row in a pandas DataFrame."""
    return [predict(tree, row, default=default) for _, row in data.iterrows()]


def compute_accuracy(tree, data, target, default=None):
    """Compute prediction accuracy for a tree on a labeled DataFrame."""
    if data.empty:
        return 0.0

    predictions = predict_many(tree, data, default=default)
    actual = data[target].tolist()
    correct = sum(prediction == label for prediction, label in zip(predictions, actual))
    return correct / len(actual)


def evaluate(tree, test_data, target, default=None):
    """Return weighted accuracy, precision, recall, and F1 metrics."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    if test_data.empty:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    predictions = predict_many(tree, test_data, default=default)
    actual = test_data[target].tolist()

    # None predictions are kept and counted as incorrect rather than excluded,
    # so the metrics reflect missing attributes or unseen branch values.
    metric_predictions = [
        "__missing_prediction__" if prediction is None else prediction
        for prediction in predictions
    ]

    return {
        "accuracy": compute_accuracy(tree, test_data, target, default=default),
        "precision": precision_score(
            actual,
            metric_predictions,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            actual,
            metric_predictions,
            average="weighted",
            zero_division=0,
        ),
        "f1": f1_score(
            actual,
            metric_predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def _manual_kfold_indices(row_count, k, random_state):
    """Create shuffled train/test index splits without sklearn."""
    indices = np.arange(row_count)
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)

    folds = np.array_split(indices, k)
    splits = []
    for fold_index, test_indices in enumerate(folds):
        train_indices = np.concatenate(
            [fold for index, fold in enumerate(folds) if index != fold_index]
        )
        splits.append((train_indices, test_indices))

    return splits


def cross_validate(data, attributes, target, k=5, random_state=42):
    """Run k-fold cross-validation using the custom ID3 implementation."""
    if data.empty:
        return {
            "mean_accuracy": 0.0,
            "std_accuracy": 0.0,
            "fold_results": [],
        }

    row_count = len(data)
    if k < 2:
        raise ValueError("k must be at least 2")
    if k > row_count:
        raise ValueError("k cannot be greater than the number of rows")

    try:
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = splitter.split(data)
    except ImportError:
        splits = _manual_kfold_indices(row_count, k, random_state)

    fold_results = []
    attributes = list(attributes)

    for train_indices, test_indices in splits:
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]
        default_prediction = majority_class(train_data[target])
        tree = id3(
            train_data,
            attributes,
            target,
            default_class=default_prediction,
        )
        accuracy = compute_accuracy(
            tree,
            test_data,
            target,
            default=default_prediction,
        )
        fold_results.append(float(accuracy))

    return {
        "mean_accuracy": float(np.mean(fold_results)),
        "std_accuracy": float(np.std(fold_results)),
        "fold_results": fold_results,
    }


def compare_with_sklearn(
    dataset_name,
    data,
    attributes,
    target,
    test_size=0.2,
    random_state=42,
):
    """Compare custom ID3 with sklearn's entropy decision tree."""
    try:
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for sklearn comparison. "
            "Install with pip install -r requirements.txt"
        ) from exc

    attributes = list(attributes)
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=None,
    )

    default_prediction = majority_class(train_data[target])
    start_time = perf_counter()
    custom_tree = id3(
        train_data,
        attributes,
        target,
        default_class=default_prediction,
    )
    custom_train_time = (perf_counter() - start_time) * 1000
    custom_predictions = predict_many(
        custom_tree,
        test_data,
        default=default_prediction,
    )
    custom_accuracy = accuracy_score(test_data[target], custom_predictions)

    encoded_train = pd.get_dummies(train_data[attributes], dummy_na=True)
    encoded_test = pd.get_dummies(test_data[attributes], dummy_na=True)
    encoded_test = encoded_test.reindex(columns=encoded_train.columns, fill_value=0)

    sklearn_model = DecisionTreeClassifier(
        criterion="entropy",
        random_state=random_state,
    )
    start_time = perf_counter()
    sklearn_model.fit(encoded_train, train_data[target])
    sklearn_train_time = (perf_counter() - start_time) * 1000
    sklearn_predictions = sklearn_model.predict(encoded_test)
    sklearn_accuracy = accuracy_score(test_data[target], sklearn_predictions)

    return pd.DataFrame(
        [
            {
                "Dataset": dataset_name,
                "Algorithm": "Custom ID3",
                "Accuracy": float(custom_accuracy),
                "Tree Size": tree_size(custom_tree),
                "Train Time (ms)": float(custom_train_time),
            },
            {
                "Dataset": dataset_name,
                "Algorithm": "sklearn DecisionTreeClassifier",
                "Accuracy": float(sklearn_accuracy),
                "Tree Size": int(sklearn_model.tree_.node_count),
                "Train Time (ms)": float(sklearn_train_time),
            },
        ],
        columns=[
            "Dataset",
            "Algorithm",
            "Accuracy",
            "Tree Size",
            "Train Time (ms)",
        ],
    )
