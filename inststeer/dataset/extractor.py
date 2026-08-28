

import os
from inststeer.utils import jload

current_dir = os.path.dirname(os.path.abspath(__file__))

def clean_text(text, tokenizer):
    """
    Clean the input text by replacing special tokens.

    Args:
        text: The input text to be cleaned.

    Returns:
        The cleaned text with special tokens replaced.
    """
    if not text:
        return text

    def insert_vline(token: str) -> str:
        if len(token) < 2:
            return " "
        elif len(token) == 2:
            return f"{token[0]}|{token[1]}"
        else:
            return f"{token[:1]}|{token[1:-1]}|{token[-1:]}"

    if tokenizer.bos_token:
        text = text.replace(tokenizer.bos_token, insert_vline(tokenizer.bos_token))
    if tokenizer.eos_token:
        text = text.replace(tokenizer.eos_token, insert_vline(tokenizer.eos_token))
    if tokenizer.pad_token:
        text = text.replace(tokenizer.pad_token, insert_vline(tokenizer.pad_token))
    if tokenizer.unk_token:
        text = text.replace(tokenizer.unk_token, insert_vline(tokenizer.unk_token))

    return text


# 说明：
# get_formatted_data1：
#   - 使用 examples 中的 instruction 作为 system 提示（或在非 chat_template 模式下拼接在文本前面）
#   - 即：每个样本的 system 提示是可变的、由数据集提供
def get_formatted_data1(examples, tokenizer, use_chat_template=True, use_system_prompt=True):
    # print(f"Processing {len(examples)} examples")
    formatted_dataset = []
    for example in examples:
        if use_chat_template:
            if use_system_prompt:   
                message = [{"role": "system", "content": f"{example['instruction']}"}, {"role": "user", "content": f"{clean_text(example['data_prompt'], tokenizer)}"}]
            else:
                message = [{"role": "user", "content": f"{clean_text(example['data_prompt'], tokenizer)}"}]
            formated_data = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        else:
            formated_data = clean_text(f"{example['instruction']}\n{example['data_prompt']}", tokenizer)
        
        formatted_dataset.append(formated_data)

    # print(f"Processed {len(formatted_dataset)} examples")
    return formatted_dataset



# get_formatted_data2：
#   - 使用固定的 system 提示 "You are a helpful assistant."
#   - 在非 chat_template 模式下只使用 data_prompt，不再拼接 instruction
#   - 并且会打印处理样本数量的日志信息
def get_formatted_data2(examples, tokenizer, use_chat_template=True, use_system_prompt=True):
    print(f"Processing {len(examples)} examples")
    formatted_dataset = []
    for example in examples:
        if use_chat_template:
            if use_system_prompt:       
                message = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": clean_text(example['data_prompt'], tokenizer)}]
            else:
                message = [{"role": "user", "content": clean_text(example['data_prompt'], tokenizer)}]
            formated_data = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        else:
            formated_data = clean_text(example['data_prompt'], tokenizer)
        
        formatted_dataset.append(formated_data)

    print(f"Processed {len(formatted_dataset)} examples")
    return formatted_dataset




def get_formatted_data(customized_instruction, path, tokenizer, use_chat_template, use_system_prompt):
    json_data = jload(os.path.join(path, 'data.json'))
    label_data = jload(os.path.join(path, 'label.json'))
    
    # We will modify the functions to also return masks
    # For compatibility, we'll iterate and construct them here or modify the helper functions.
    # Modifying the helpers is cleaner.
    
    if customized_instruction:
        return get_formatted_data1(json_data, tokenizer, use_chat_template, use_system_prompt), label_data
    else:
        return get_formatted_data2(json_data, tokenizer, use_chat_template, use_system_prompt), label_data


def get_formatted_data_with_mask(customized_instruction, path, tokenizer, use_chat_template, use_system_prompt):
    """
    简化版：通过占位符定位 untrusted 区域
    
    核心思路：
    1. 用占位符构建 chat template
    2. 找到占位符的字符位置
    3. 用实际文本替换占位符，位置即确定
    """
    json_data = jload(os.path.join(path, 'data.json'))
    label_data = jload(os.path.join(path, 'label.json'))
    
    formatted_dataset = []
    masks = []
    
    # 获取特殊 token IDs，用于排除
    special_token_ids = set()
    if hasattr(tokenizer, 'all_special_ids'):
        special_token_ids = set(tokenizer.all_special_ids)

    # 占位符
    PLACEHOLDER = "<<<UNTRUSTED>>>"

    for example in json_data:
        # 使用 strip() 去除首尾空白，保持一致性
        untrusted_text = clean_text(example['data_prompt'], tokenizer).strip()
        instruction_text = clean_text(example.get('instruction', ''), tokenizer)
        
        # 1. 用占位符构造模板，找到 untrusted 区域的字符位置
        if use_chat_template:
            system_content = instruction_text if customized_instruction else "You are a helpful assistant."
            if use_system_prompt and system_content:
                template_messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": PLACEHOLDER}
                ]
            else:
                template_messages = [{"role": "user", "content": PLACEHOLDER}]
            template_prompt = tokenizer.apply_chat_template(template_messages, tokenize=False, add_generation_prompt=True)
            # 找到占位符位置
            placeholder_start = template_prompt.find(PLACEHOLDER)
            # 用实际文本替换占位符
            full_prompt = template_prompt.replace(PLACEHOLDER, untrusted_text)
            start_char = placeholder_start
            end_char = placeholder_start + len(untrusted_text)
        else:
            # 非 chat_template 模式，直接拼接
            if customized_instruction:
                prefix = f"{instruction_text}\n"
                full_prompt = prefix + untrusted_text
                start_char = len(prefix)
            else:
                full_prompt = untrusted_text
                start_char = 0
            end_char = start_char + len(untrusted_text)

        # 2. Tokenize 并获取 offset mapping
        encoding = tokenizer(full_prompt, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = encoding.input_ids
        offsets = encoding.offset_mapping
        
        # 3. 根据字符位置生成 mask（排除特殊 token）
        mask = [0] * len(full_ids)
        for idx, (tok_start, tok_end) in enumerate(offsets):
            # 检查 token 是否在 untrusted 区间内
            if tok_start < end_char and tok_end > start_char:
                # 排除特殊 token
                if full_ids[idx] not in special_token_ids:
                    mask[idx] = 1
        
        formatted_dataset.append(full_prompt)
        masks.append(mask)

    return formatted_dataset, masks, label_data



def pad_masks_for_batch(masks, max_len, padding_side="left"):
    """
    Args:
        masks: 原始 mask 列表 (List[List[int]])
        max_len: 目标长度 (通常是 batch 中最长序列的长度)
        padding_side: "left" 或 "right"，应与 tokenizer.padding_side 一致
    
    Returns:
        padded_masks: padding 后的 mask 列表
    
    Example:
        # 在 batch 推理时使用
        batch_encoding = tokenizer(prompts, padding=True, return_tensors="pt")
        max_len = batch_encoding.input_ids.shape[1]
        padded_masks = pad_masks_for_batch(masks, max_len, tokenizer.padding_side)
    """
    padded_masks = []
    for mask in masks:
        pad_len = max_len - len(mask)
        if pad_len < 0:
            raise ValueError(f"Mask length {len(mask)} exceeds max_len {max_len}")
        
        if padding_side == "left":
            # Left padding: [0, 0, ..., 0, original_mask]
            padded_mask = [0] * pad_len + mask
        else:
            # Right padding: [original_mask, 0, 0, ..., 0]
            padded_mask = mask + [0] * pad_len
        
        padded_masks.append(padded_mask)
    
    return padded_masks


def get_train_data(customized_instruction, path, tokenizer, use_chat_template, use_system_prompt):
    json_data = jload(os.path.join(path, 'data.json'))
    label_data = jload(os.path.join(path, 'label.json'))
    new_label_data = []
    new_json_data = []
    for (label, example) in zip(label_data, json_data):
        if label == 1:
            continue
        instruction_message = [{"role": "user", "content": f"{clean_text(example['instruction'], tokenizer)}"}]
        knowledge_message = [{"role": "user", "content": f"{clean_text(example['data_prompt'], tokenizer)}"}]
        instruction_formatted_data = tokenizer.apply_chat_template(instruction_message, tokenize=False, add_generation_prompt=True)
        knowledge_formatted_data = tokenizer.apply_chat_template(knowledge_message, tokenize=False, add_generation_prompt=True)
        new_label_data.append(1)
        new_json_data.append(instruction_formatted_data)
        new_label_data.append(0)
        new_json_data.append(knowledge_formatted_data)

    return new_json_data, new_label_data