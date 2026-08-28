



import functools
from typing import Callable

from datasets.utils.logging import disable_progress_bar, enable_progress_bar






def disable_datasets_progress_bar(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        disable_progress_bar()
        ret = func(*args, **kwargs)
        enable_progress_bar()
        return ret

    return wrapper

