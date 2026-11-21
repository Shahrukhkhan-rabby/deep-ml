import math
from collections import Counter


def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    total = len(labels)
    if total == 0:
        return 0.0

    counts = Counter(labels)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob)

    return entropy


def calculate_information_gain(
    examples: list[dict], attr: str, target_attr: str
) -> float:
    """Calculate the information gain of splitting on attr."""
    total = len(examples)
    if total == 0:
        return 0.0

    parent_labels = [ex[target_attr] for ex in examples]
    parent_entropy = calculate_entropy(parent_labels)

    subsets: dict = {}
    for ex in examples:
        subsets.setdefault(ex[attr], []).append(ex[target_attr])

    weighted_child_entropy = 0.0
    for subset_labels in subsets.values():
        weight = len(subset_labels) / total
        weighted_child_entropy += weight * calculate_entropy(subset_labels)

    return parent_entropy - weighted_child_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(ex[target_attr] for ex in examples)
    # Sort by highest count first (-count), then alphabetically by label (label)
    best_label, _ = min(counts.items(), key=lambda item: (-item[1], item[0]))
    return best_label


def learn_decision_tree(
    examples: list[dict], attributes: list[str], target_attr: str
) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    target_values = [ex[target_attr] for ex in examples]

    # Pure node: all examples have the same target class
    if len(set(target_values)) == 1:
        return target_values[0]

    # No attributes left to split on
    if not attributes:
        return majority_class(examples, target_attr)

    best_attr = None
    best_gain = -1.0

    for attr in attributes:
        gain = calculate_information_gain(examples, attr, target_attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    if best_attr is None:
        return majority_class(examples, target_attr)

    remaining_attrs = [a for a in attributes if a != best_attr]
    values = sorted({ex[best_attr] for ex in examples})

    tree: dict = {best_attr: {}}
    for val in values:
        subset = [ex for ex in examples if ex[best_attr] == val]
        if not subset:
            tree[best_attr][val] = majority_class(examples, target_attr)
        else:
            tree[best_attr][val] = learn_decision_tree(
                subset, remaining_attrs, target_attr
            )

    return tree