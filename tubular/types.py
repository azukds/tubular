from typing import Union

import narwhals as nw
import pandas as pd
import polars as pl

DataFrame = Union[
    pd.DataFrame,
    pl.DataFrame,
    pl.LazyFrame,
    nw.DataFrame,
    nw.LazyFrame,
]

Series = Union[
    pd.Series,
    pl.Series,
    nw.Series,
]
