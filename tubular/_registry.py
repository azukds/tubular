from tubular.base import BaseTransformer

CLASS_REGISTRY = {}


def register(cls: BaseTransformer) -> BaseTransformer:
    """Add transformer to registry dict.

    Returns:
    -------
    cls - transformer

    Example:
    -------
    >>> @register
    ... class MyTransformer(BaseTransformer):
    ...     pass
    >>> CLASS_REGISTRY["MyTransformer"]
    <class 'MyTransformer'>

    """
    CLASS_REGISTRY[cls.__name__] = cls
    return cls
