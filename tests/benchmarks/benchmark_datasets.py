"Setup dummy datasets dicts for benchmarking."

import numpy as np

COLUMNS = ["a", "b", "c"]


def _get_numeric_dict(n_rows: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed=0)
    data = {}
    NULL_THRESH = 0.1
    for col in COLUMNS:
        values = rng.normal(size=n_rows)
        values[rng.random(n_rows) < NULL_THRESH] = np.nan
        data[col] = values
    return data
