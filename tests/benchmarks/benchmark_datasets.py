"Setup dummy datasets dicts for benchmarking."

import numpy as np

COLUMNS = ["a", "b", "c"]
NULL_FRACTION = 0.1
STRING_LEVELS = ["alpha", "beta", "gamma", "delta", "epsilon"]
SEED = 0


def _get_numeric_dict(n_rows: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed=SEED)
    data = {}
    for col in COLUMNS:
        values = rng.normal(size=n_rows)
        values[rng.random(n_rows) < NULL_FRACTION] = np.nan
        data[col] = values
    return data


def _with_nulls(values: np.ndarray, rng: np.random.Generator) -> list:
    """Return values as a python list, with a fraction of entries set to None.

    Used for boolean and string columns, which have no NaN equivalent.
    """
    values = values.astype(object)
    values[rng.random(len(values)) < NULL_FRACTION] = None
    return values.tolist()


def _get_boolean_dict(n_rows: int) -> dict[str, list]:
    rng = np.random.default_rng(seed=SEED)
    HALF = 0.5
    return {col: _with_nulls(rng.random(n_rows) < HALF, rng) for col in COLUMNS}


def _get_string_dict(n_rows: int) -> dict[str, list]:
    rng = np.random.default_rng(seed=SEED)
    return {
        col: _with_nulls(rng.choice(STRING_LEVELS, size=n_rows), rng) for col in COLUMNS
    }
