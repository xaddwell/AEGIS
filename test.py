

import os
import json
import sklearn
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datasets import Dataset, concatenate_datasets

from inststeer.model import load_model_config
from inststeer.dataset import load_my_dataset
from inststeer.dataset import get_formatted_data, get_train_data
from inststeer.utils.steer import Steerer, AsymmetricLDA, condition_similarity, obtain_direction
from inststeer.utils import format_prompts, get_hidden_states, seed_everything, jdump, jload
from inststeer.utils.hidden_state import get_hidden_states_fast


seed_everything(42)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama3.1-8b")
    parser.add_argument("--num_samples_per_class", type=int, default=200)
    parser.add_argument("--extract_layer_position", type=float, default=4/5)
    parser.add_argument("--extract_token_position", type=str, default="last")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()

def plot_hits(scores, labels, save_path):
    # 2. 分离两类数据
    scores_0 = scores[labels == 0]
    scores_1 = scores[labels == 1]

    # 3. 创建统一的 bins (分箱)
    # 这一步很重要，确保两类数据的柱子宽度一致，方便对比
    # 如果数据极差为0 (所有分都一样)，就强制设少量 bins
    if np.ptp(scores) == 0:
        bins = 10 
    else:
        bins = np.linspace(scores.min(), scores.max(), 60)

    # 4. 绘图
    plt.figure(figsize=(10, 6))

    # 绘制 Class 0 (蓝色)
    # density=True: 显示“密度”而不是“数量”，这样即使两类样本数差距很大也能公平对比分布形状
    plt.hist(scores_0, bins=bins, color='blue', alpha=0.5, 
             label='Class 0 (Label 0)', density=False, edgecolor='black', linewidth=0.5)

    # 绘制 Class 1 (橙色)
    plt.hist(scores_1, bins=bins, color='orange', alpha=0.6, 
             label='Class 1 (Label 1)', density=False, edgecolor='black', linewidth=0.5)

    # 5. 装饰图表
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Decision Boundary (0)')
    plt.title("Distribution of Decision Scores by Class", fontsize=14)
    plt.xlabel("Score (Distance to Hyperplane)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='best')
    plt.grid(axis='y', alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')


if __name__ == "__main__":

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    test_data_dir = os.path.join(data_dir, 'TestData')
    models_dir = os.path.join(data_dir, 'models')
    figures_dir = os.path.join(data_dir, 'figures')
    dataset_candidates_dir = os.path.join(test_data_dir, 'dataset_candidates.json')
    dataset_candidates = jload(dataset_candidates_dir)

    args = parse_args()

    cfg = load_model_config(args.model_name)
    model = cfg['model'].eval().to(f"cuda:{args.device}")
    tokenizer = cfg['tokenizer']
    pad_token_id = cfg['pad_token_id']
    use_system_prompt = cfg['use_system_prompt']
    use_chat_template = cfg['use_chat_template']
    extract_layer_ids = int(cfg['num_hidden_layers']*args.extract_layer_position)
    extract_token_position = args.extract_token_position
    num_samples_per_class = args.num_samples_per_class

    lda = AsymmetricLDA(concentrated_label=1, layer_id=extract_layer_ids, token_idx=extract_token_position)
    lda.load(os.path.join(models_dir, f'{args.model_name}_num:{args.num_samples_per_class}_layer:{extract_layer_ids}_token:{args.extract_token_position}.pkl'))

    for (clean_dataset_name, malicious_dataset_name) in dataset_candidates:
        clean_dataset_dir = os.path.join(test_data_dir, clean_dataset_name)
        malicious_dataset_dir = os.path.join(test_data_dir, malicious_dataset_name)
        clean_formatted_data, clean_label_data = get_formatted_data(
            customized_instruction=False, 
            path=clean_dataset_dir, 
            tokenizer=tokenizer, 
            use_chat_template=use_chat_template, 
            use_system_prompt=use_system_prompt
        )
        malicious_formatted_data, malicious_label_data = get_formatted_data(
            customized_instruction=False, 
            path=malicious_dataset_dir, 
            tokenizer=tokenizer, 
            use_chat_template=use_chat_template, 
            use_system_prompt=use_system_prompt
        )
        min_num_samples = min(len(clean_formatted_data), len(malicious_formatted_data))
        test_dataset = Dataset.from_dict({
            "text": clean_formatted_data[:min_num_samples] + malicious_formatted_data[:min_num_samples],
            "label": clean_label_data[:min_num_samples] + malicious_label_data[:min_num_samples]
        })
        hs_test_layer_token = get_hidden_states_fast(
            model, tokenizer, 
            test_dataset, 
            prompt_key="text",
            batch_size=128,
            show_progress=True,
            extract_token_position=extract_token_position,
            extract_layer_ids=[extract_layer_ids]
        )
        hs_test_layer_token = torch.from_numpy(
            hs_test_layer_token[extract_layer_ids]
        )

        scores_test = lda.decision_function(hs_test_layer_token)
        test_labels = np.array(test_dataset['label'])
        figure_name = f'{args.model_name}_{malicious_dataset_name.replace("/", "_")}_layer:{extract_layer_ids}_token:{extract_layer_ids}.png'
        plot_hits(scores_test, test_labels, save_path=os.path.join(figures_dir, figure_name))