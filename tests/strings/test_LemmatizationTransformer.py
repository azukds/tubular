from tubular.strings import LemmatizationTransformer
import polars as pl

def test():

    df=pl.DataFrame({'a': ['super fascinating text column for reading', 'hello hi howdy']})

    transformer=LemmatizationTransformer(columns='a')

    print(transformer.transform(df))
    assert 1==2