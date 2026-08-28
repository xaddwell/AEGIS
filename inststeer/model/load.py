
import os
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inststeer.utils import jload

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

logger = logging.getLogger(__name__)


def load_model_tokenizer(model_path: str, bf16: bool = True, device_map: str = "cuda:0"):
    # if "/" in model_path:
    #     model_path = model_path.split("/")[1]
    # 动态从环境变量读取 HF_HOME
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    model_path = os.path.join(hf_home, model_path)
    if not os.path.exists(model_path):
        raise ValueError(f"No model in {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        logger.warning("No pad token found, using eos token")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_model_config(model_name: str, bf16: bool = False, device_map: str = "cuda:0"):
    config = jload(CONFIG_PATH)['models'][model_name]
    model, tokenizer = load_model_tokenizer(config['path'], bf16=bf16, device_map=device_map)
    config["model"] = model
    config["tokenizer"] = tokenizer
    config["pad_token_id"] = tokenizer.eos_token_id
    return config