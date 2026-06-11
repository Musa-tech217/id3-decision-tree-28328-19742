"""ID3 decision tree utilities."""

from math import log2

import pandas as pd


def _resolve_column_name(data, column_name):
    """Resolve a column name, accepting case-insensitive matches."""
    if column_name in data.columns:
        return column_name

    normalized = str(column_name).strip().lower()
    matches = [
        column for column in data.columns if str(column).strip().lower() == normalized
    ]
    if matches:
        return matches[0]

    raise KeyError(f"Column not found: {column_name}")


def entropy(labels):
    """Calculate entropy for a sequence of class labels."""
    clean_labels = pd.Series(labels).dropna()
    total = len(clean_labels)
    if total == 0:
        return 0.0

    probabilities = clean_labels.value_counts() / total
    result = -sum(probability * log2(probability) for probability in probabilities)
    return 0.0 if abs(result) < 1e-12 else float(result)


def information_gain(data, attribute, target):
    """Calculate information gain for an attribute against a target column."""
    if data.empty:
        return 0.0

    attribute = _resolve_column_name(data, attribute)
    target = _resolve_column_name(data, target)
    clean_data = data[[attribute, target]].dropna()
    if clean_data.empty:
        return 0.0

    parent_entropy = entropy(clean_data[target])
    total_rows = len(clean_data)
    weighted_child_entropy = 0.0

    for _, subset in clean_data.groupby(attribute):
        subset_weight = len(subset) / total_rows
        weighted_child_entropy += subset_weight * entropy(subset[target])

    return float(parent_entropy - weighted_child_entropy)


def majority_class(labels):
    """Return the most common label, using sorted order to break ties."""
    clean_labels = pd.Series(labels).dropna()
    if clean_labels.empty:
        return None

    counts = clean_labels.value_counts()
    max_count = counts.max()
    tied_classes = counts[counts == max_count].index.tolist()
    return _label_value(sorted(tied_classes, key=lambda label: str(label))[0])


def _sorted_values(values):
    """Sort branch values deterministically, falling back to string order."""
    return sorted(values, key=lambda value: (str(type(value)), str(value)))


def _label_value(label):
    """Convert pandas or numpy scalar labels to plain Python values."""
    return label.item() if hasattr(label, "item") else label


def id3(data, attributes, target, default_class=None):
    """Build an ID3 decision tree as a nested dictionary."""
    if data.empty:
        return default_class

    target = _resolve_column_name(data, target)
    clean_data = data.dropna(subset=[target])
    if clean_data.empty:
        return default_class

    # Baseline variant: keep the recursive ID3 steps explicit so the notebook
    # mirrors the textbook stopping rules.
    class_labels = clean_data[target]
    if class_labels.nunique() == 1:
        return _label_value(class_labels.iloc[0])

    resolved_feature_columns = [
        _resolve_column_name(clean_data, attribute) for attribute in list(attributes)
    ]
    if not resolved_feature_columns:
        return majority_class(class_labels)

    node_default_class = majority_class(class_labels)
    candidate_gains = [
        (attribute, information_gain(clean_data, attribute, target))
        for attribute in resolved_feature_columns
    ]

    # If every gain is zero, keep splitting on the first best attribute from the
    # caller's deterministic attribute order instead of stopping early.
    best_attribute, _ = max(candidate_gains, key=lambda item: item[1])
    remaining_attributes = [
        attribute for attribute in resolved_feature_columns if attribute != best_attribute
    ]

    tree = {best_attribute: {}}
    branch_values = clean_data[best_attribute].dropna().unique()
    for value in _sorted_values(branch_values):
        subset = clean_data[clean_data[best_attribute] == value]
        tree[best_attribute][value] = id3(
            subset,
            remaining_attributes,
            target,
            default_class=node_default_class,
        )

    return tree


def is_leaf(tree):
    """Return True when a tree value is a leaf."""
    return not isinstance(tree, dict)


def tree_size(tree):
    """Count every decision node and leaf node in a tree."""
    if is_leaf(tree):
        return 1

    size = 1
    for branches in tree.values():
        for subtree in branches.values():
            size += tree_size(subtree)
    return size


class ID3DecisionTree:
    """Placeholder for the from-scratch ID3 decision tree classifier."""

    pass
