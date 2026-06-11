"""Reduced error pruning utilities for custom ID3 trees."""

from copy import deepcopy

import numpy as np
import pandas as pd

from src.evaluation import compute_accuracy, predict_many
from src.id3 import id3, majority_class, tree_size


MIN_VALIDATION_SAMPLES_PER_BRANCH = 2


def count_nodes(tree):
    """Count every decision node and leaf node in a tree."""
    return tree_size(tree)


def _is_leaf(tree):
    return not isinstance(tree, dict)


def _has_strong_tie_pruning_evidence(
    tree,
    validation_data,
    validation_target,
    replacement_class,
):
    """Return True when an accuracy tie is enough evidence to prune."""
    if _is_leaf(tree) or validation_data.empty:
        return False

    attribute = next(iter(tree))
    if attribute not in validation_data.columns:
        return False

    observed_branch_counts = validation_data[attribute].value_counts()
    observed_branch_counts = observed_branch_counts[observed_branch_counts > 0]

    if len(observed_branch_counts) < 2:
        return False
    if observed_branch_counts.min() < MIN_VALIDATION_SAMPLES_PER_BRANCH:
        return False

    subtree_predictions = predict_many(
        tree,
        validation_data,
        default=replacement_class,
    )
    replacement_predictions = [replacement_class] * len(validation_data)

    return subtree_predictions == replacement_predictions


def _prune_node(tree, validation_data, validation_target):
    if _is_leaf(tree) or validation_data.empty:
        return tree

    attribute = next(iter(tree))
    branches = tree[attribute]
    pruned_tree = {attribute: {}}

    for value, subtree in branches.items():
        if attribute in validation_data.columns:
            branch_validation_data = validation_data[validation_data[attribute] == value]
        else:
            branch_validation_data = validation_data.iloc[0:0]

        pruned_tree[attribute][value] = _prune_node(
            subtree,
            branch_validation_data,
            validation_target,
        )

    replacement_class = majority_class(validation_data[validation_target])
    if replacement_class is None:
        return pruned_tree

    default_prediction = replacement_class
    before_accuracy = compute_accuracy(
        pruned_tree,
        validation_data,
        validation_target,
        default=default_prediction,
    )
    after_accuracy = compute_accuracy(
        replacement_class,
        validation_data,
        validation_target,
        default=default_prediction,
    )

    if after_accuracy > before_accuracy:
        return replacement_class
    if after_accuracy == before_accuracy and _has_strong_tie_pruning_evidence(
        pruned_tree,
        validation_data,
        validation_target,
        replacement_class,
    ):
        # Accuracy ties on tiny validation slices are weak evidence and can
        # collapse useful subtrees. On ties, prune only when validation rows
        # cover multiple branches with enough examples per branch and the
        # subtree makes exactly the same validation predictions as the leaf.
        return replacement_class

    return pruned_tree


def prune(tree, validation_data, validation_target, attributes):
    """Prune a custom ID3 tree using reduced error pruning."""
    _ = attributes
    copied_tree = deepcopy(tree)
    return _prune_node(copied_tree, validation_data, validation_target)


def _train_validation_test_split(data, random_state):
    indices = np.arange(len(data))
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)

    test_count = max(1, round(len(data) * 0.15))
    validation_count = max(1, round(len(data) * 0.15))
    train_count = len(data) - validation_count - test_count
    if train_count < 1:
        raise ValueError("At least 3 rows are required for pruning_experiment")

    validation_end = train_count + validation_count
    train_indices = indices[:train_count]
    validation_indices = indices[train_count:validation_end]
    test_indices = indices[validation_end:]

    return (
        data.iloc[train_indices],
        data.iloc[validation_indices],
        data.iloc[test_indices],
    )


def pruning_experiment(data, attributes, target, random_state=42):
    """Run a train/validation/test reduced error pruning experiment."""
    if len(data) < 3:
        raise ValueError("At least 3 rows are required for pruning_experiment")

    attributes = list(attributes)
    train_data, validation_data, test_data = _train_validation_test_split(
        data,
        random_state,
    )

    default_prediction = majority_class(train_data[target])
    unpruned_tree = id3(
        train_data,
        attributes,
        target,
        default_class=default_prediction,
    )
    pruned_tree = prune(
        unpruned_tree,
        validation_data,
        target,
        attributes,
    )

    rows = [
        {
            "Metric": "Training Accuracy",
            "Unpruned": compute_accuracy(
                unpruned_tree,
                train_data,
                target,
                default=default_prediction,
            ),
            "Pruned": compute_accuracy(
                pruned_tree,
                train_data,
                target,
                default=default_prediction,
            ),
        },
        {
            "Metric": "Validation Accuracy",
            "Unpruned": compute_accuracy(
                unpruned_tree,
                validation_data,
                target,
                default=default_prediction,
            ),
            "Pruned": compute_accuracy(
                pruned_tree,
                validation_data,
                target,
                default=default_prediction,
            ),
        },
        {
            "Metric": "Test Accuracy",
            "Unpruned": compute_accuracy(
                unpruned_tree,
                test_data,
                target,
                default=default_prediction,
            ),
            "Pruned": compute_accuracy(
                pruned_tree,
                test_data,
                target,
                default=default_prediction,
            ),
        },
        {
            "Metric": "Number of Nodes",
            "Unpruned": count_nodes(unpruned_tree),
            "Pruned": count_nodes(pruned_tree),
        },
    ]

    return pd.DataFrame(rows, columns=["Metric", "Unpruned", "Pruned"])
