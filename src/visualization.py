"""Visualization utilities for custom ID3 and sklearn decision trees."""

from pathlib import Path


def _is_leaf(tree):
    return not isinstance(tree, dict)


def _format_subtree(tree, prefix="", connector=""):
    if _is_leaf(tree):
        return [f"{prefix}{tree}"]

    attribute = next(iter(tree))
    branches = tree[attribute]
    lines = [f"{prefix}{connector}{attribute}"]
    branch_items = list(branches.items())
    branch_prefix = prefix
    if connector:
        branch_prefix += "    " if connector == "└── " else "│   "

    for index, (value, subtree) in enumerate(branch_items):
        is_last = index == len(branch_items) - 1
        branch_connector = "└── " if is_last else "├── "
        child_prefix = branch_prefix + ("    " if is_last else "│   ")

        if _is_leaf(subtree):
            lines.append(f"{branch_prefix}{branch_connector}{value} -> {subtree}")
        else:
            lines.append(f"{branch_prefix}{branch_connector}{value}")
            lines.extend(_format_subtree(subtree, child_prefix, "└── "))

    return lines


def format_tree(tree, indent=""):
    """Convert a custom nested-dictionary ID3 tree into readable text."""
    return "\n".join(_format_subtree(tree, indent))


def print_tree(tree):
    """Print a formatted custom ID3 tree."""
    print(format_tree(tree))


def save_text_tree(tree, filepath):
    """Save a formatted custom ID3 tree to a text file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(format_tree(tree), encoding="utf-8")


def plot_sklearn_tree(model, feature_names, class_names, filepath):
    """Save a matplotlib plot of a fitted sklearn decision tree."""
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
    )
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
