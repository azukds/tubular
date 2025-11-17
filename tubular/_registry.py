"""Class registry for transformers.

This module provides a decorator to register transformer classes by name,
making it easier to look up transformers dynamically.
"""

CLASS_REGISTRY = {}


def register(cls):
    """Decorator to register a transformer class in the registry.

    Parameters
    ----------
    cls : type
        The transformer class to register.

    Returns
    -------
    type
        The same class, for use as a decorator.

    Example
    -------
    >>> @register
    ... class MyTransformer(BaseTransformer):
    ...     pass
    >>> CLASS_REGISTRY["MyTransformer"]
    <class 'MyTransformer'>

    """
    CLASS_REGISTRY[cls.__name__] = cls
    return cls

