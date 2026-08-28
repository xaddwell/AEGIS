from datasets import load_dataset
import datasets

from .load import register_dataset
from .utils import disable_datasets_progress_bar


@register_dataset("dolly")
def dolly_dataset():
    # 增加超时时间，避免网络超时错误
    # 设置更长的超时时间（默认 10 秒，改为 60 秒）
    import os
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
    
    try:
        return load_dataset("databricks/databricks-dolly-15k", split="train")
    except Exception as e:
        print(f"Failed to load dolly dataset from HuggingFace: {e}")
        print("Trying to load from cache...")
        # 尝试从缓存加载
        return load_dataset("databricks/databricks-dolly-15k", split="train", download_mode="reuse_cache_if_exists")


@register_dataset("dolly-inst")
def dolly_inst_dataset():
    ds = dolly_dataset()
    ds = ds.rename_column("instruction", "prompt").select_columns(["prompt"])
    return ds


@register_dataset("dolly-context")
@disable_datasets_progress_bar
def dolly_context_dataset():
    ds = dolly_dataset()
    ds = ds.filter(lambda x: x["context"] != "")
    ds = ds.rename_column("context", "prompt")
    ds = ds.select_columns(["prompt"])
    return ds
