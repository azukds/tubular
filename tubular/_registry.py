CLASS_REGISTRY = {}


def register(cls: type) -> type:
    """Add class object to registry(dict).

    Returns
    -------
    cls - class object.

    """
    CLASS_REGISTRY[cls.__name__] = cls
    return cls
