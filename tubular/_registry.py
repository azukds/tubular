CLASS_REGISTRY = {}


def register(cls):
    """Add class object to registry"""

    CLASS_REGISTRY[cls.__name__] = cls
    return cls
