from datasets import Dataset, load_dataset

from .load import register_dataset
from .utils import disable_datasets_progress_bar


@register_dataset("alpaca")
def alpaca_dataset() -> Dataset:
    return load_dataset("tatsu-lab/alpaca", split="train")


@register_dataset("alpaca-inst")
def alpaca_inst_dataset() -> Dataset:
    ds = alpaca_dataset()
    ds = ds.rename_column("instruction", "prompt").select_columns(["prompt"])
    return ds


@register_dataset("alpaca-context")
@disable_datasets_progress_bar
def alpaca_context_dataset() -> Dataset:
    ds = alpaca_dataset()
    ds = (
        ds.filter(lambda x: x["input"] != "")
        .rename_column("input", "prompt")
        .select_columns(["prompt"])
    )
    return ds
