from datasets import Dataset
from transformers import AutoTokenizer

from inststeer.constants import DATASET_N_PROC


def format_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_prompts(
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    prompt_key: str = "prompt",
    formatted_prompt_key: str = "formatted_prompt",
) -> Dataset:
    def format_prompt_fn(example):
        return {formatted_prompt_key: format_prompt(tokenizer, example[prompt_key])}

    return dataset.map(format_prompt_fn, num_proc=DATASET_N_PROC)
