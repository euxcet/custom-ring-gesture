def get_labels_id(labels: list[str], use_labels: list[str]) -> list[int]:
    return [labels.index(label) for label in use_labels]