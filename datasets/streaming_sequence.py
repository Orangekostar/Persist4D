from numbers import Integral


def causal_windows(scan_indices):
    try:
        indices = tuple(scan_indices)
    except TypeError as error:
        raise ValueError("scan_indices must be an iterable of integers") from error

    if len(indices) < 2:
        raise ValueError("scan_indices must contain at least two indices")

    normalized_indices = []
    for scan_index in indices:
        if isinstance(scan_index, bool) or not isinstance(scan_index, Integral):
            raise ValueError(
                "scan_indices elements must be integral and not boolean"
            )
        normalized_indices.append(int(scan_index))

    if len(set(normalized_indices)) != len(normalized_indices):
        raise ValueError("scan_indices must not contain duplicate indices")

    windows = [(normalized_indices[0],)]
    windows.extend(zip(normalized_indices, normalized_indices[1:]))
    return tuple(windows)
