from typing import Callable

_DATASET_REGISTRY: dict[str, Callable] = {}


def register_dataset(name: str):
    def decorator(func: Callable):
        _DATASET_REGISTRY[name] = func
        return func

    return decorator


def load_my_dataset(name: str, **kwargs):
    return _DATASET_REGISTRY[name](**kwargs)
