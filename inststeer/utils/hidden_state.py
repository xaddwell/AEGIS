

import typing
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


@torch.inference_mode()
def get_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Dataset | str,
    prompt_key: str = "formatted_prompt",
    max_new_tokens: int = 1,
    output_device: str = "cpu",
    batch_size: int = 8,
    show_progress: bool = True,
    extract_token_position: int | None = None,
):
    """
    Extract hidden states from model with batch processing to avoid OOM.
    Note: This version uses model.generate() which is slower. Consider using get_hidden_states_fast() instead.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: Dataset, list of strings, or single string
        prompt_key: Key to extract prompts from Dataset
        max_new_tokens: Number of new tokens to generate (set to 1 for minimal generation)
        output_device: Device to store outputs (use "cpu" to save GPU memory)
        batch_size: Number of samples to process at once
        show_progress: Whether to show progress bar
        extract_token_position: If set, extract only this token position from each sample.
                                -1 for last non-padding token, None to keep all tokens.
                                Setting this significantly reduces memory usage.
    
    Returns:
        hidden_states: Tuple of tensors, one per layer
                      Shape: (num_samples, hidden_size) if extract_token_position is set
                             (num_samples, seq_len, hidden_size) otherwise
        output_ids: Generated token IDs
    """
    # Convert input to list of prompts
    if isinstance(prompts, str):
        prompts = [prompts]
    elif isinstance(prompts, Dataset):
        prompts = list(prompts[prompt_key])
    else:
        raise NotImplementedError(f"Unsupported type: {type(prompts)}")

    num_samples = len(prompts)
    all_hidden_states = None
    
    # Process in batches
    iterator = range(0, num_samples, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting hidden states", total=(num_samples + batch_size - 1) // batch_size)
    
    for i in iterator:
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=2048  # Prevent extremely long sequences
        ).to(model.device)
        
        # Generate (minimal, just 1 token to get hidden states)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Get hidden states
        outputs = model(
            output_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Process hidden states
        if extract_token_position is not None:
            # Extract specific token position for each sample
            if extract_token_position == -1:
                # Extract last non-padding token using attention mask
                attention_mask = inputs['attention_mask']
                # Get the index of the last non-padding token for each sample
                seq_lengths = attention_mask.sum(dim=1) - 1  # -1 to convert to 0-indexed
                
                # Extract the hidden state at the last token position for each sample
                batch_hidden_states = tuple(
                    torch.stack([
                        layer[sample_idx, seq_lengths[sample_idx], :]
                        for sample_idx in range(layer.shape[0])
                    ]).to(output_device)
                    for layer in outputs.hidden_states
                )
            else:
                # Extract fixed position (excluding the last generated token)
                batch_hidden_states = tuple(
                    layer[:, extract_token_position, :].to(output_device)
                    for layer in outputs.hidden_states
                )
        else:
            # Keep all tokens except the last generated one
            batch_hidden_states = tuple(
                layer[:, :-1, :].to(output_device) for layer in outputs.hidden_states
            )
        
        # Accumulate results
        if all_hidden_states is None:
            # Initialize with the first batch
            all_hidden_states = batch_hidden_states
        else:
            # Concatenate along batch dimension
            all_hidden_states = tuple(
                torch.cat([existing, new], dim=0)
                for existing, new in zip(all_hidden_states, batch_hidden_states)
            )
        
        # Clear GPU cache after each batch
        del inputs, output_ids, outputs, batch_hidden_states
        torch.cuda.empty_cache()
    
    return all_hidden_states

@torch.inference_mode()
def get_hidden_states_fast(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Dataset | str,
    prompt_key: str = "text",
    batch_size: int = 8,
    show_progress: bool = True,
    extract_token_position: typing.Union[int, str] = 1,
    extract_layer_ids: list[int] = list(range(32)),
):
    """
    Fast version: Extract hidden states without generation (forward pass only).
    This is much faster and more memory efficient when you don't need to generate text.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: Dataset, list of strings, or single string
        prompt_key: Key to extract prompts from Dataset
        output_device: Device to store outputs (use "cpu" to save GPU memory)
        batch_size: Number of samples to process at once
        show_progress: Whether to show progress bar
        extract_token_position: If set, extract only this token position from each sample.
                                -1 for last non-padding token, None to keep all tokens.
                                Setting this significantly reduces memory usage.
    
    Returns:
        hidden_states: Tuple of tensors, one per layer
                      Shape: (num_samples, hidden_size) if extract_token_position is set
                             (num_samples, seq_len, hidden_size) otherwise
        input_ids: Input token IDs
    """
    # Convert input to list of prompts
    if isinstance(prompts, str):
        prompts = [prompts]
    elif isinstance(prompts, Dataset):
        prompts = list(prompts[prompt_key])
    else:
        raise NotImplementedError(f"Unsupported type: {type(prompts)}")

    num_samples = len(prompts)
    hidden_states = {layer: [] for layer in extract_layer_ids}
    
    # Process in batches
    iterator = range(0, num_samples, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting hidden states", total=(num_samples + batch_size - 1) // batch_size)
    
    with torch.no_grad():
        for batch_idx, i in enumerate(iterator):
            inputs = tokenizer(
                prompts[i:i + batch_size], 
                return_tensors="pt", 
                max_length=100, 
                padding=True, 
                truncation=True
            ).to(model.device)

            outputs = model(**inputs, return_dict=True, output_hidden_states=True)
            # Extract and immediately move to CPU for each layer
            for layer_id in extract_layer_ids:
                hidden_idx = layer_id + 1 if layer_id >= 0 else layer_id

                for i, batch_hidden in enumerate(outputs.hidden_states[hidden_idx]):
                    # Extract specific token position for each sample
                    if extract_token_position == "last":
                        # Get the index of the last non-padding token for each sample
                        batch_hs = batch_hidden[-1, :]
                    elif extract_token_position == "all":
                        batch_hs = torch.mean(batch_hidden, dim=0)
                    elif isinstance(extract_token_position, int):
                        batch_hs = torch.mean(batch_hidden[-extract_token_position:, :], dim=0)
                    else:
                        raise ValueError(f"Invalid extract_token_position: {extract_token_position}")
                    
                    hidden_states[layer_id].append(batch_hs.squeeze().cpu().numpy())
            
            del outputs
            torch.cuda.empty_cache()
    
    return {k: np.vstack(v) for k, v in hidden_states.items()}

@torch.inference_mode()
def get_hidden_states_full(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Dataset | str,
    prompt_key: str = "text",
    batch_size: int = 8,
    show_progress: bool = True,
):

    """
    Outputs: 
        hidden_states: numpy array of shape (num_samples, num_layers, seq_len, hidden_size)
    """
    # Convert input to list of prompts
    if isinstance(prompts, str):
        prompts = [prompts]
    elif isinstance(prompts, Dataset):
        prompts = list(prompts[prompt_key])
    else:
        raise NotImplementedError(f"Unsupported type: {type(prompts)}")

    num_samples = len(prompts)
    hidden_states = {}
    
    # Process in batches
    iterator = range(0, num_samples, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting hidden states", total=(num_samples + batch_size - 1) // batch_size)
    
    with torch.no_grad():
        for batch_idx, i in enumerate(iterator):
            inputs = tokenizer(
                prompts[i:i + batch_size], 
                return_tensors="pt", 
                max_length=100,
                truncation=True,
                padding='max_length',
            ).to(model.device)
            outputs = model(**inputs, return_dict=True, output_hidden_states=True)
            for layer_id, layer_hs in enumerate(outputs.hidden_states):
                if layer_id not in hidden_states.keys():
                    hidden_states[layer_id] = []
                hidden_states[layer_id].append(layer_hs.cpu().numpy())
    
    return {k: np.vstack(v) for k, v in hidden_states.items()}

def extract_hidden_states_layer_token(
    hidden_states: np.ndarray,
    layer_ids: list[int],
    token_position: typing.Union[int, str] = 1,
):
    """
    Args:
        hidden_states: numpy array of shape (num_samples, num_layers, seq_len, hidden_size)
        layer_ids: list of layer IDs to extract hidden states from
        token_position: token position to extract hidden states from
            - "last": extract the last token
            - "all": extract all tokens
            - int: extract the last x tokens
    Returns:
        numpy array of shape (num_samples, hidden_size)
    """
    extracted_hidden_states = []
    for layer_id in layer_ids:

        if token_position == "last":
            extracted_hidden_states.append(hidden_states[layer_id][ :, -1, :])
        elif token_position == "all":
            extracted_hidden_states.append(np.mean(hidden_states[layer_id][:, :, :], axis=0))
        elif isinstance(token_position, int):
            extracted_hidden_states.append(np.mean(hidden_states[layer_id][:, -token_position:, :], axis=1))
        else:
            raise ValueError(f"Invalid token_position: {token_position}")
    
    return np.stack(extracted_hidden_states, axis=1)


def get_tokens(tokenizer: AutoTokenizer, token_ids: torch.Tensor) -> list[str]:
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = [token.replace("Ġ", "▁") for token in tokens]
    tokens = [token.replace("ĊĊ", "⏎⏎") for token in tokens]
    return tokens


