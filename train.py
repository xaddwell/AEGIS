
import os
import json
import sklearn
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datasets import Dataset, concatenate_datasets

from inststeer.model import load_model_config
from inststeer.dataset import get_formatted_data, get_train_data
from inststeer.utils.steer import Steerer, AsymmetricLDA, condition_similarity, obtain_direction
from inststeer.utils.hidden_state import get_hidden_states_full, extract_hidden_states_layer_token
from inststeer.utils import seed_everything, jdump, jload, load_pickle, save_pickle


seed_everything(42)


def gen_dataset():
    datasets = []
    # get opi datasets
    for attack_pos in ['end']:
        for attack_strategy in ['naive', 'escape', 'ignore', 'fake_comp', 'combine', 'neural_exec', 'pleak', 'universal']:
            for target_task in ['sms_spam', 'sst2', 'mrpc', 'hsol', 'rte', 'jfleg','gigaword']:
                clean_dataset_name = f"opi_clean/{target_task}"
                for inject_task in ['sms_spam', 'sst2', 'mrpc', 'hsol', 'rte', 'jfleg','gigaword']:
                    malicious_dataset_name = f"opi_malicious/{attack_pos}/{attack_strategy}/{target_task}_{inject_task}"
                    datasets.append((clean_dataset_name, malicious_dataset_name))
    # get other datasets
    other_datasets={'dolly': ['dolly','dolly'],
                    'mmlu': ['mmlu','mmlu'],
                    'boolq': ['boolq','boolq'],
                    'hotelreview': ['hotelreview','close']
                    }
    for other_data, tasks_name in other_datasets.items():
        clean_dataset_name = f"{other_data}_clean/{other_data}"
        target_task, inject_task = tasks_name[0], tasks_name[1]
        for attack_pos in ['end']:
            for attack_strategy in ['naive', 'escape', 'ignore', 'fake_comp', 'combine', 'neural_exec', 'pleak', 'universal']:
                malicious_dataset_name = f"{other_data}_malicious/{attack_pos}/{attack_strategy}/{target_task}_{inject_task}"
                datasets.append((clean_dataset_name, malicious_dataset_name))

    return datasets



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama3.1-8b")
    parser.add_argument("--num_samples_per_class", type=int, default=200)
    parser.add_argument("--extract_layer_position", type=float, default=4/5)
    parser.add_argument("--extract_token_position", type=str, default="last")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()



if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    train_data_dir = os.path.join(data_dir, 'TrainData/data')
    test_data_dir = os.path.join(data_dir, 'TestData')
    hidden_states_dir = os.path.join(data_dir, 'hidden_states')
    models_dir = os.path.join(data_dir, 'models')
    dataset_candidates_dir = os.path.join(test_data_dir, 'dataset_candidates.json')
    if os.path.exists(dataset_candidates_dir):
        dataset_candidates = jload(dataset_candidates_dir)
    
    args = parse_args()

    cfg = load_model_config(args.model_name, device_map=f"cuda:{args.device}")
    model = cfg['model'].eval().to(f"cuda:{args.device}")
    tokenizer = cfg['tokenizer']
    pad_token_id = cfg['pad_token_id']
    use_system_prompt = cfg['use_system_prompt']
    use_chat_template = cfg['use_chat_template']
    extract_layer_ids = int(cfg['num_hidden_layers']*args.extract_layer_position)
    extract_token_position = args.extract_token_position
    num_samples_per_class = args.num_samples_per_class
    

    train_data, train_label = get_formatted_data(
        customized_instruction=False, 
        path=train_data_dir, 
        tokenizer=tokenizer, 
        use_chat_template=use_chat_template, 
        use_system_prompt=False
    )

    full_dataset = Dataset.from_dict({"text": train_data, "label": train_label})
    dataset_label_0 = full_dataset.filter(lambda x: x['label'] == 0)
    dataset_label_1 = full_dataset.filter(lambda x: x['label'] == 1)

    sample_0 = dataset_label_0.shuffle(seed=42).select(range(min(num_samples_per_class, len(dataset_label_0))))
    sample_1 = dataset_label_1.shuffle(seed=42).select(range(min(num_samples_per_class, len(dataset_label_1))))
    train_dataset = concatenate_datasets([sample_0, sample_1]).shuffle(seed=42)
    train_labels = np.array(train_dataset['label'])
    print(f"Final balanced dataset: {len(train_dataset)} samples")

    hs_file_path = os.path.join(hidden_states_dir, f"hs_{args.model_name}_{args.num_samples_per_class}")
    label_file_path = os.path.join(hidden_states_dir, f"label_{args.model_name}_{args.num_samples_per_class}")
    if os.path.exists(hs_file_path+".pkl"):
        hs_train_full = load_pickle(hs_file_path)
        train_labels = load_pickle(label_file_path)
    else:
        hs_train_full = get_hidden_states_full(
            model, tokenizer, 
            train_dataset, 
            prompt_key="text",
            batch_size=32,
            show_progress=True,
        )
        save_pickle(hs_train_full, hs_file_path)
        save_pickle(train_labels, label_file_path)
    
    hs_train_layer_token = extract_hidden_states_layer_token(
        hs_train_full,
        layer_ids=[extract_layer_ids],
        token_position=extract_token_position
    ).squeeze(axis=1)
    lda = AsymmetricLDA(concentrated_label=1, layer_id=extract_layer_ids, token_idx=extract_token_position)
    lda.fit(hs_train_layer_token, train_labels)
    lda.save(os.path.join(models_dir, f'{args.model_name}_num:{args.num_samples_per_class}_layer:{extract_layer_ids}_token:{args.extract_token_position}.pkl'))