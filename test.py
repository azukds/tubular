import pandas as pd
import polars as pl
import narwhals as nw
import numpy as np
from tubular.nominal import MeanResponseTransformer
import time

n=200000
rng=np.random.default_rng(42)

df=pl.DataFrame({
    "a": rng.choice(['a', 'b', 'c', 'd', 'e'], size=n),
    "b": rng.choice([0,1], size=n)
})

transformer=MeanResponseTransformer(
    columns=['a'],
    prior=2,
    unseen_level_handling=None,
)

times=[]
for i in range(100):
    start=time.time()
    transformer.fit(df[["a"]], df["b"])
    times.append(time.time()-start)

print('polars new ',np.mean(times))