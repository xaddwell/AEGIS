"""
测试 MultiLayerVotingDetector 的检测效果
函数式编程风格，便于大规模实验
"""

import os

import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Tuple, Any, Optional

from inststeer.model import load_model_config
from inststeer.dataset import get_formatted_data
from inststeer.utils.steer import MultiLayerVotingDetector
from inststeer.utils.hidden_state import get_hidden_states_full, extract_hidden_states_layer_token
from inststeer.utils import seed_everything, jload, load_pickle, save_pickle

seed_everything(42)


# =============================================================================
# 基础工具函数
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        positive_label: 正类标签
    
    Returns:
        包含 TP, TN, FP, FN, TPR, FPR, FNR, Precision, F1, Accuracy 的字典
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        if len(np.unique(y_true)) == 1:
            if y_true[0] == positive_label:
                tp = int(np.sum(y_pred == positive_label))
                fn = int(np.sum(y_pred != positive_label))
                tn, fp = 0, 0
            else:
                tn = int(np.sum(y_pred != positive_label))
                fp = int(np.sum(y_pred == positive_label))
                tp, fn = 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    
    return {
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'TPR': float(tpr), 'FPR': float(fpr), 'FNR': float(fnr),
        'Precision': float(precision), 'F1': float(f1), 'Accuracy': float(accuracy),
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    聚合多个数据集的指标，计算平均值
    
    Args:
        metrics_list: 指标字典列表
    
    Returns:
        包含各指标平均值的字典
    """
    if not metrics_list:
        return {}
    
    keys = ['TPR', 'FPR', 'FNR', 'Precision', 'F1', 'Accuracy']
    return {
        f'avg_{key}': float(np.mean([m[key] for m in metrics_list]))
        for key in keys
    }


# =============================================================================
# 数据加载函数
# =============================================================================

def load_model_and_config(model_name: str, device: int) -> Dict[str, Any]:
    """
    加载模型和配置
    
    Returns:
        包含 model, tokenizer, use_system_prompt, use_chat_template, num_hidden_layers 的字典
    """
    cfg = load_model_config(model_name, device_map=f"cuda:{device}")
    model = cfg['model'].eval().to(f"cuda:{device}")
    
    return {
        'model': model,
        'tokenizer': cfg['tokenizer'],
        'use_system_prompt': cfg['use_system_prompt'],
        'use_chat_template': cfg['use_chat_template'],
        'num_hidden_layers': cfg['num_hidden_layers'],
    }


def load_or_extract_train_hidden_states(
    model, tokenizer,
    train_data_dir: str,
    hidden_states_dir: str,
    model_name: str,
    num_samples_per_class: int,
    use_chat_template: bool,
    use_system_prompt: bool = False,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    加载或提取训练集的 hidden states
    
    Returns:
        (hs_train_full, train_labels) 元组
    """
    hs_file_path = os.path.join(hidden_states_dir, f"hs_{model_name}_{num_samples_per_class}")
    label_file_path = os.path.join(hidden_states_dir, f"label_{model_name}_{num_samples_per_class}")
    
    if os.path.exists(hs_file_path + ".pkl"):
        hs_train_full = load_pickle(hs_file_path)
        train_labels = load_pickle(label_file_path)
    else:
        train_data, train_label = get_formatted_data(
            customized_instruction=False,
            path=train_data_dir,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
            use_system_prompt=use_system_prompt
        )
        
        full_dataset = Dataset.from_dict({"text": train_data, "label": train_label})
        dataset_label_0 = full_dataset.filter(lambda x: x['label'] == 0)
        dataset_label_1 = full_dataset.filter(lambda x: x['label'] == 1)
        
        sample_0 = dataset_label_0.shuffle(seed=42).select(range(min(num_samples_per_class, len(dataset_label_0))))
        sample_1 = dataset_label_1.shuffle(seed=42).select(range(min(num_samples_per_class, len(dataset_label_1))))
        train_dataset = concatenate_datasets([sample_0, sample_1]).shuffle(seed=42)
        train_labels = np.array(train_dataset['label'])
        
        hs_train_full = get_hidden_states_full(
            model, tokenizer,
            train_dataset,
            prompt_key="text",
            batch_size=32,
            show_progress=True
        )
        
        save_pickle(hs_train_full, hs_file_path)
        save_pickle(train_labels, label_file_path)
    
    return hs_train_full, train_labels


def extract_multilayer_hidden_states(
    model, tokenizer,
    clean_dir: str,
    malicious_dir: str,
    use_chat_template: bool,
    use_system_prompt: bool,
    layer_ids: List[int],
    extract_token_position: str,
    num_samples_per_class: int = 100,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    提取多层的 hidden states
    
    Returns:
        包含 hidden_states_dict, labels, texts 的字典
    """
    clean_data, _ = get_formatted_data(
        customized_instruction=False,
        path=clean_dir,
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
        use_system_prompt=use_system_prompt
    )
    
    malicious_data, _ = get_formatted_data(
        customized_instruction=False,
        path=malicious_dir,
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
        use_system_prompt=use_system_prompt
    )
    
    clean_dataset = Dataset.from_dict({"text": clean_data, "label": [0] * len(clean_data)})
    malicious_dataset = Dataset.from_dict({"text": malicious_data, "label": [1] * len(malicious_data)})
    
    n_clean = min(num_samples_per_class, len(clean_dataset))
    n_malicious = min(num_samples_per_class, len(malicious_dataset))
    
    clean_sample = clean_dataset.shuffle(seed=42).select(range(n_clean))
    malicious_sample = malicious_dataset.shuffle(seed=42).select(range(n_malicious))
    
    test_dataset = concatenate_datasets([clean_sample, malicious_sample])
    test_labels = np.array(test_dataset['label'])
    
    hs_full = get_hidden_states_full(
        model, tokenizer,
        test_dataset,
        prompt_key="text",
        batch_size=batch_size,
        show_progress=False
    )
    
    hidden_states_dict = {}
    for layer_id in layer_ids:
        hs_layer = extract_hidden_states_layer_token(
            hs_full,
            layer_ids=[layer_id],
            token_position=extract_token_position
        ).squeeze(axis=1)
        hidden_states_dict[layer_id] = hs_layer
    
    return {
        'hidden_states_dict': hidden_states_dict,
        'labels': test_labels,
        'texts': list(test_dataset['text']),
    }


def extract_layer_hidden_states(
    hs_full: Dict[int, np.ndarray],
    layer_ids: List[int],
    token_position: str
) -> Dict[int, np.ndarray]:
    """
    从完整的 hidden states 中提取指定层的数据
    
    Returns:
        {layer_id: hidden_states} 字典
    """
    result = {}
    for layer_id in layer_ids:
        hs_layer = extract_hidden_states_layer_token(
            hs_full,
            layer_ids=[layer_id],
            token_position=token_position
        ).squeeze(axis=1)
        result[layer_id] = hs_layer
    return result


# =============================================================================
# 评估函数
# =============================================================================

def evaluate_on_dataset(
    detector: MultiLayerVotingDetector,
    hidden_states_dict: Dict[int, np.ndarray],
    labels: np.ndarray,
    layer_ids: List[int],
    soft_threshold: float,
    hard_threshold: int
) -> Dict[str, Any]:
    """
    在单个数据集上评估检测器
    
    Returns:
        包含预测结果和指标的字典:
        - results: predict 的原始结果
        - metrics_soft: soft 模式指标
        - metrics_hard: hard 模式指标
        - metrics_per_layer: 每层的指标 {layer_id: metrics}
    """
    results = detector.predict(
        hidden_states_dict,
        soft_threshold=soft_threshold,
        hard_threshold=hard_threshold
    )
    
    metrics_soft = compute_metrics(labels, results['soft_pred'], positive_label=1)
    metrics_hard = compute_metrics(labels, results['hard_pred'], positive_label=1)
    
    metrics_per_layer = {}
    for j, layer_id in enumerate(layer_ids):
        layer_pred = results['stack_decisions'][j]
        metrics_per_layer[layer_id] = compute_metrics(labels, layer_pred, positive_label=1)
    
    return {
        'results': results,
        'metrics_soft': metrics_soft,
        'metrics_hard': metrics_hard,
        'metrics_per_layer': metrics_per_layer,
    }


def evaluate_on_multiple_datasets(
    detector: MultiLayerVotingDetector,
    model, tokenizer,
    dataset_candidates: List[Tuple[str, str]],
    test_data_dir: str,
    layer_ids: List[int],
    use_chat_template: bool,
    use_system_prompt: bool,
    extract_token_position: str,
    soft_threshold: float,
    hard_threshold: int,
    num_samples_per_class: int,
    log_file: Optional[str] = None,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    在多个数据集上评估检测器
    
    Returns:
        包含所有评估结果的字典:
        - results_soft: soft 模式每个数据集的结果列表
        - results_hard: hard 模式每个数据集的结果列表
        - results_per_layer: 每层每个数据集的结果 {layer_id: [results]}
        - all_labels: 所有标签
        - all_preds_soft: soft 模式所有预测
        - all_preds_hard: hard 模式所有预测
        - all_preds_per_layer: 每层所有预测 {layer_id: preds}
        - overall_soft: soft 模式总体指标
        - overall_hard: hard 模式总体指标
        - overall_per_layer: 每层总体指标 {layer_id: metrics}
        - avg_soft: soft 模式平均指标
        - avg_hard: hard 模式平均指标
        - avg_per_layer: 每层平均指标 {layer_id: avg_metrics}
    """
    results_soft = []
    results_hard = []
    results_per_layer = {layer_id: [] for layer_id in layer_ids}
    
    all_labels = []
    all_preds_soft = []
    all_preds_hard = []
    all_preds_per_layer = {layer_id: [] for layer_id in layer_ids}
    all_scores_soft = []
    all_scores_per_layer = {layer_id: [] for layer_id in layer_ids}
    
    # 按数据集收集详细 scores（用于后续绘图）
    detailed_scores = []
    
    iterator = dataset_candidates
    if show_progress:
        iterator = tqdm(dataset_candidates, desc="Testing datasets")
    
    for clean_name, malicious_name in iterator:
        clean_dir = os.path.join(test_data_dir, clean_name)
        malicious_dir = os.path.join(test_data_dir, malicious_name)
        
        if not os.path.exists(clean_dir) or not os.path.exists(malicious_dir):
            continue
        
        try:
            data = extract_multilayer_hidden_states(
                model, tokenizer,
                clean_dir, malicious_dir,
                use_chat_template, use_system_prompt,
                layer_ids, extract_token_position,
                num_samples_per_class=num_samples_per_class,
                batch_size=16
            )
        except Exception as e:
            print(f"\n  [Warning] Failed to process {malicious_name}: {e}")
            continue
        
        eval_result = evaluate_on_dataset(
            detector, data['hidden_states_dict'], data['labels'],
            layer_ids, soft_threshold, hard_threshold
        )
        
        # 注：不再写入每个样本的详细日志，改为在 summary.json 中保存每个数据集的结果
        
        # 收集结果
        results_soft.append({'dataset': malicious_name, **eval_result['metrics_soft']})
        results_hard.append({'dataset': malicious_name, **eval_result['metrics_hard']})
        
        for layer_id in layer_ids:
            results_per_layer[layer_id].append({
                'dataset': malicious_name,
                **eval_result['metrics_per_layer'][layer_id]
            })
            all_preds_per_layer[layer_id].extend(
                eval_result['results']['stack_decisions'][layer_ids.index(layer_id)].tolist()
            )
        
        all_labels.extend(data['labels'].tolist())
        all_preds_soft.extend(eval_result['results']['soft_pred'].tolist())
        all_preds_hard.extend(eval_result['results']['hard_pred'].tolist())
        all_scores_soft.extend(eval_result['results']['soft_aggregate_score'].tolist())
        for j, layer_id in enumerate(layer_ids):
            all_scores_per_layer[layer_id].extend(
                eval_result['results']['stack_scores'][j].tolist()
            )
        
        # 收集详细的每个样本 scores（按数据集）
        n_samples = len(data['labels'])
        dataset_scores = {
            'dataset': malicious_name,
            'n_samples': n_samples,
            'labels': data['labels'].tolist(),
            'soft_scores': eval_result['results']['soft_aggregate_score'].tolist(),
            'soft_preds': eval_result['results']['soft_pred'].tolist(),
            'hard_votes': eval_result['results']['hard_votes'].tolist(),
            'hard_preds': eval_result['results']['hard_pred'].tolist(),
            'layer_scores': {},
            'layer_preds': {},
        }
        for j, layer_id in enumerate(layer_ids):
            dataset_scores['layer_scores'][layer_id] = eval_result['results']['stack_scores'][j].tolist()
            dataset_scores['layer_preds'][layer_id] = eval_result['results']['stack_decisions'][j].tolist()
        detailed_scores.append(dataset_scores)
    
    # 计算总体和平均指标
    overall_soft = compute_metrics(np.array(all_labels), np.array(all_preds_soft), positive_label=1)
    overall_hard = compute_metrics(np.array(all_labels), np.array(all_preds_hard), positive_label=1)
    overall_per_layer = {
        layer_id: compute_metrics(np.array(all_labels), np.array(all_preds_per_layer[layer_id]), positive_label=1)
        for layer_id in layer_ids
    }
    
    avg_soft = aggregate_metrics(results_soft)
    avg_hard = aggregate_metrics(results_hard)
    avg_per_layer = {
        layer_id: aggregate_metrics(results_per_layer[layer_id])
        for layer_id in layer_ids
    }
    
    return {
        'results_soft': results_soft,
        'results_hard': results_hard,
        'results_per_layer': results_per_layer,
        'all_labels': all_labels,
        'all_preds_soft': all_preds_soft,
        'all_preds_hard': all_preds_hard,
        'all_preds_per_layer': all_preds_per_layer,
        'all_scores_soft': all_scores_soft,
        'all_scores_per_layer': all_scores_per_layer,
        'detailed_scores': detailed_scores,  # 按数据集的详细 scores
        'overall_soft': overall_soft,
        'overall_hard': overall_hard,
        'overall_per_layer': overall_per_layer,
        'avg_soft': avg_soft,
        'avg_hard': avg_hard,
        'avg_per_layer': avg_per_layer,
    }


# =============================================================================
# 日志和输出函数
# =============================================================================

def write_detailed_log(
    log_file: str,
    dataset_name: str,
    sample_texts: List[str],
    labels: np.ndarray,
    results: Dict[str, Any],
    layer_ids: List[int]
) -> None:
    """将详细的检测结果写入日志文件"""
    n_samples = len(labels)
    
    stack_scores = results['stack_scores']
    stack_decisions = results['stack_decisions']
    soft_pred = results['soft_pred']
    hard_pred = results['hard_pred']
    soft_aggregate_score = results['soft_aggregate_score']
    hard_votes = results['hard_votes']
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*100}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Layers: {layer_ids}\n")
        f.write(f"Total samples: {n_samples}\n")
        f.write(f"Soft threshold: {results['soft_threshold']}, Hard threshold: {results['hard_threshold']}\n")
        f.write(f"{'='*100}\n\n")
        
        for i in range(n_samples):
            f.write(f"--- Sample {i+1}/{n_samples} ---\n")
            f.write(f"True Label: {labels[i]} ({'malicious' if labels[i] == 1 else 'benign'})\n")
            f.write(f"\n[Soft Voting]\n")
            f.write(f"  Aggregate Score: {soft_aggregate_score[i]:.4f}\n")
            f.write(f"  Prediction: {soft_pred[i]} ({'malicious' if soft_pred[i] == 1 else 'benign'})\n")
            f.write(f"  Correct: {'✓' if labels[i] == soft_pred[i] else '✗'}\n")
            f.write(f"\n[Hard Voting]\n")
            f.write(f"  Votes: {int(hard_votes[i])}/{len(layer_ids)}\n")
            f.write(f"  Prediction: {hard_pred[i]} ({'malicious' if hard_pred[i] == 1 else 'benign'})\n")
            f.write(f"  Correct: {'✓' if labels[i] == hard_pred[i] else '✗'}\n")
            f.write(f"\nPer-layer results:\n")
            
            for j, layer_id in enumerate(layer_ids):
                score = stack_scores[j, i]
                decision = stack_decisions[j, i]
                correct = '✓' if labels[i] == decision else '✗'
                f.write(f"  Layer {layer_id:3d}: Score={score:+8.4f}, Decision={decision} ({'attack' if decision == 1 else 'benign'}) {correct}\n")
            
            text_preview = sample_texts[i][:200] + "..." if len(sample_texts[i]) > 200 else sample_texts[i]
            text_preview = text_preview.replace('\n', ' ')
            f.write(f"\nText preview: {text_preview}\n")
            f.write(f"\n{'-'*80}\n")


def write_log_header(log_file: str, config: Dict[str, Any]) -> None:
    """写入日志头部"""
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"MultiLayerVotingDetector Evaluation Log\n")
        f.write(f"{'='*100}\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write(f"{'='*100}\n")


def write_log_summary(log_file: str, eval_results: Dict[str, Any], layer_ids: List[int]) -> None:
    """写入日志汇总"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n{'='*100}\n")
        f.write(f"SUMMARY\n")
        f.write(f"{'='*100}\n")
        
        avg_soft = eval_results['avg_soft']
        avg_hard = eval_results['avg_hard']
        
        f.write(f"Aggregated - Soft Mode:\n")
        f.write(f"  Avg TPR: {avg_soft['avg_TPR']:.4f}, Avg FPR: {avg_soft['avg_FPR']:.4f}, ")
        f.write(f"Avg F1: {avg_soft['avg_F1']:.4f}, Avg Acc: {avg_soft['avg_Accuracy']:.4f}\n")
        
        f.write(f"Aggregated - Hard Mode:\n")
        f.write(f"  Avg TPR: {avg_hard['avg_TPR']:.4f}, Avg FPR: {avg_hard['avg_FPR']:.4f}, ")
        f.write(f"Avg F1: {avg_hard['avg_F1']:.4f}, Avg Acc: {avg_hard['avg_Accuracy']:.4f}\n")
        
        f.write(f"\nPer-Layer Results:\n")
        for layer_id in layer_ids:
            avg = eval_results['avg_per_layer'][layer_id]
            f.write(f"  Layer {layer_id}: Avg TPR={avg['avg_TPR']:.4f}, Avg FPR={avg['avg_FPR']:.4f}, ")
            f.write(f"Avg F1={avg['avg_F1']:.4f}, Avg Acc={avg['avg_Accuracy']:.4f}\n")


def save_detailed_scores(
    eval_results: Dict[str, Any],
    save_path: str,
    config: Dict[str, Any]
) -> None:
    """
    保存所有详细的 scores 数据为 JSON 格式（用于后续绘图分析）
    
    保存的数据结构:
    {
        'config': {...},  # 实验配置
        'layer_ids': [...],  # 检测层列表
        'datasets': [  # 按数据集组织的详细 scores
            {
                'dataset': 'xxx',
                'n_samples': N,
                'labels': [...],  # 真实标签
                'soft_scores': [...],  # soft 聚合分数
                'soft_preds': [...],  # soft 预测
                'hard_votes': [...],  # hard 投票数
                'hard_preds': [...],  # hard 预测
                'layer_scores': {layer_id: [...]},  # 每层分数
                'layer_preds': {layer_id: [...]},  # 每层预测
            },
            ...
        ]
    }
    """
    # 转换 layer_scores 和 layer_preds 的 key 为字符串（JSON 不支持 int key）
    datasets_for_json = []
    for ds in eval_results['detailed_scores']:
        ds_copy = ds.copy()
        ds_copy['layer_scores'] = {str(k): v for k, v in ds['layer_scores'].items()}
        ds_copy['layer_preds'] = {str(k): v for k, v in ds['layer_preds'].items()}
        datasets_for_json.append(ds_copy)
    
    data_to_save = {
        'config': config,
        'layer_ids': config.get('layer_ids', []),
        'datasets': datasets_for_json,
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    print(f"\n[详细 scores 已保存到: {save_path}]")


def plot_roc_curves(
    eval_results: Dict[str, Any],
    layer_ids: List[int],
    save_path: str,
    title: str = "ROC Curves"
) -> Dict[str, float]:
    """
    绘制 ROC 曲线并保存为 SVG 格式
    
    Args:
        eval_results: evaluate_on_multiple_datasets 的返回结果
        layer_ids: 层 ID 列表
        save_path: SVG 文件保存路径
        title: 图表标题
    
    Returns:
        包含各方法 AUC 值的字典
    """
    labels = np.array(eval_results['all_labels'])
    scores_soft = np.array(eval_results['all_scores_soft'])
    scores_per_layer = eval_results['all_scores_per_layer']
    
    # 设置绘图风格
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
    })
    
    auc_results = {}
    
    # 使用颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(layer_ids) + 1))
    
    # 绘制 Soft Voting 的 ROC 曲线
    fpr_soft, tpr_soft, _ = roc_curve(labels, scores_soft)
    auc_soft = auc(fpr_soft, tpr_soft)
    auc_results['Soft Voting'] = auc_soft
    plt.plot(fpr_soft, tpr_soft, 
             color='darkred', linewidth=2.5, linestyle='-',
             label=f'Soft Voting (AUC = {auc_soft:.4f})')
    
    # 绘制每层的 ROC 曲线
    for idx, layer_id in enumerate(layer_ids):
        scores_layer = np.array(scores_per_layer[layer_id])
        fpr_layer, tpr_layer, _ = roc_curve(labels, scores_layer)
        auc_layer = auc(fpr_layer, tpr_layer)
        auc_results[f'Layer {layer_id}'] = auc_layer
        plt.plot(fpr_layer, tpr_layer, 
                 color=colors[idx], linewidth=1.5, linestyle='--',
                 label=f'Layer {layer_id} (AUC = {auc_layer:.4f})', alpha=0.8)
    
    # 绘制对角线（随机分类器）
    plt.plot([0, 1], [0, 1], color='gray', linewidth=1, linestyle=':', label='Random')
    
    # 设置图表样式
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # 保存为 SVG
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"\n[ROC 曲线已保存到: {save_path}]")
    
    return auc_results


def print_results_table(results: List[Dict], title: str) -> None:
    """打印结果表格"""
    print(f"\n[{title}]")
    print("-"*110)
    print(f"{'Dataset':<50} {'TPR':>8} {'FPR':>8} {'FNR':>8} {'F1':>8} {'Acc':>8}")
    print("-"*110)
    for result in results:
        dataset_short = result['dataset'].split('/')[-1]
        print(f"{dataset_short:<50} {result['TPR']:>8.4f} {result['FPR']:>8.4f} {result['FNR']:>8.4f} {result['F1']:>8.4f} {result['Accuracy']:>8.4f}")


def print_summary(eval_results: Dict[str, Any], layer_ids: List[int], config: Dict[str, Any]) -> None:
    """打印汇总统计"""
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    
    overall_soft = eval_results['overall_soft']
    overall_hard = eval_results['overall_hard']
    avg_soft = eval_results['avg_soft']
    avg_hard = eval_results['avg_hard']
    
    print("\n[Aggregated - Soft Mode]")
    print(f"  Layers: {layer_ids}, Threshold: {config['soft_threshold']}")
    print(f"  Overall: TPR={overall_soft['TPR']:.4f}, FPR={overall_soft['FPR']:.4f}, F1={overall_soft['F1']:.4f}, Acc={overall_soft['Accuracy']:.4f}")
    print(f"  Average: TPR={avg_soft['avg_TPR']:.4f}, FPR={avg_soft['avg_FPR']:.4f}, F1={avg_soft['avg_F1']:.4f}, Acc={avg_soft['avg_Accuracy']:.4f}")
    
    print("\n[Aggregated - Hard Mode]")
    print(f"  Layers: {layer_ids}, Threshold: {config['hard_threshold']} votes")
    print(f"  Overall: TPR={overall_hard['TPR']:.4f}, FPR={overall_hard['FPR']:.4f}, F1={overall_hard['F1']:.4f}, Acc={overall_hard['Accuracy']:.4f}")
    print(f"  Average: TPR={avg_hard['avg_TPR']:.4f}, FPR={avg_hard['avg_FPR']:.4f}, F1={avg_hard['avg_F1']:.4f}, Acc={avg_hard['avg_Accuracy']:.4f}")
    
    print("\n[Per-Layer Results]")
    for layer_id in layer_ids:
        avg = eval_results['avg_per_layer'][layer_id]
        overall = eval_results['overall_per_layer'][layer_id]
        print(f"  Layer {layer_id}: Overall F1={overall['F1']:.4f}, Avg F1={avg['avg_F1']:.4f}, Avg TPR={avg['avg_TPR']:.4f}, Avg FPR={avg['avg_FPR']:.4f}")


def print_comparison(eval_results: Dict[str, Any], layer_ids: List[int]) -> None:
    """打印对比分析"""
    print("\n" + "="*70)
    print("Comparison Analysis")
    print("="*70)
    
    avg_soft = eval_results['avg_soft']
    avg_hard = eval_results['avg_hard']
    avg_per_layer = eval_results['avg_per_layer']
    
    print(f"\n{'Method':<20} {'Avg TPR':>10} {'Avg FPR':>10} {'Avg FNR':>10} {'Avg F1':>10} {'Avg Acc':>10}")
    print("-"*80)
    print(f"{'Soft Voting':<20} {avg_soft['avg_TPR']:>10.4f} {avg_soft['avg_FPR']:>10.4f} {avg_soft['avg_FNR']:>10.4f} {avg_soft['avg_F1']:>10.4f} {avg_soft['avg_Accuracy']:>10.4f}")
    print(f"{'Hard Voting':<20} {avg_hard['avg_TPR']:>10.4f} {avg_hard['avg_FPR']:>10.4f} {avg_hard['avg_FNR']:>10.4f} {avg_hard['avg_F1']:>10.4f} {avg_hard['avg_Accuracy']:>10.4f}")
    
    for layer_id in layer_ids:
        avg = avg_per_layer[layer_id]
        print(f"{f'Layer {layer_id}':<20} {avg['avg_TPR']:>10.4f} {avg['avg_FPR']:>10.4f} {avg['avg_FNR']:>10.4f} {avg['avg_F1']:>10.4f} {avg['avg_Accuracy']:>10.4f}")
    
    # 分析
    print("\n  Analysis:")
    best_layer = max(layer_ids, key=lambda l: avg_per_layer[l]['avg_F1'])
    best_layer_f1 = avg_per_layer[best_layer]['avg_F1']
    
    print(f"    - Best single layer: Layer {best_layer}, F1={best_layer_f1:.4f}")
    print(f"    - Soft aggregation F1={avg_soft['avg_F1']:.4f}, {'improved' if avg_soft['avg_F1'] > best_layer_f1 else 'decreased'} by {abs(avg_soft['avg_F1'] - best_layer_f1)*100:.2f}%")
    print(f"    - Hard aggregation F1={avg_hard['avg_F1']:.4f}, {'improved' if avg_hard['avg_F1'] > best_layer_f1 else 'decreased'} by {abs(avg_hard['avg_F1'] - best_layer_f1)*100:.2f}%")


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="评估 MultiLayerVotingDetector 检测效果")
    parser.add_argument("--model_name", type=str, default="llama3.1-8b")
    parser.add_argument("--num_samples_per_class", type=int, default=200, help="训练集每类样本数")
    parser.add_argument("--start_layer_ratio", type=float, default=2/3, help="起始层位置比例")
    parser.add_argument("--num_layers", type=int, default=5, help="检测使用的层数")
    parser.add_argument("--extract_token_position", type=str, default="last", help="提取 token 位置")
    parser.add_argument("--soft_threshold", type=float, default=0.0, help="Soft 模式聚合阈值")
    parser.add_argument("--hard_threshold", type=int, default=None, help="Hard 模式投票阈值")
    parser.add_argument("--target_fpr_per_layer", type=float, default=0.05, help="每层目标 FPR")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_test_datasets", type=int, default=10, help="测试数据集数量")
    parser.add_argument("--test_samples_per_class", type=int, default=50, help="每类测试样本数")
    parser.add_argument("--log_dir", type=str, default="logs/multilayer_voting", help="日志输出目录")
    return parser.parse_args()


def run_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行完整的评估流程
    
    Args:
        config: 配置字典，包含所有参数
    
    Returns:
        包含所有评估结果的字典
    """
    device = f"cuda:{config['device']}"
    
    # 路径设置
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    train_data_dir = os.path.join(data_dir, 'TrainData/data')
    test_data_dir = os.path.join(data_dir, 'TestData')
    hidden_states_dir = os.path.join(data_dir, 'hidden_states')
    
    # 创建带配置-时间的实验文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_ratio_str = f"{config['start_layer_ratio']:.2f}".replace('.', 'p')  # 0.67 -> 0p67
    experiment_name = (
        f"{config['model_name']}_"
        f"s{start_ratio_str}_"
        f"n{config['num_layers']}_"
        f"t{config['num_test_datasets']}_"
        f"{timestamp}"
    )
    experiment_dir = os.path.join(config['log_dir'], experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 设置输出文件路径
    log_file = os.path.join(experiment_dir, "eval.log")
    
    # 1. 加载模型
    print("="*70)
    print("Loading model...")
    print("="*70)
    
    model_cfg = load_model_and_config(config['model_name'], config['device'])
    num_hidden_layers = model_cfg['num_hidden_layers']
    
    # 计算检测层索引
    start_layer = int(num_hidden_layers * config['start_layer_ratio'])
    layer_ids = list(range(start_layer, min(start_layer + config['num_layers'], num_hidden_layers)))
    
    # 设置 hard 模式阈值
    hard_threshold = config['hard_threshold'] if config['hard_threshold'] is not None else len(layer_ids) // 2 + 1
    config['hard_threshold'] = hard_threshold
    config['layer_ids'] = layer_ids
    config['timestamp'] = timestamp
    config['experiment_dir'] = experiment_dir
    
    print(f"  Model: {config['model_name']}")
    print(f"  Detection layers: {layer_ids}")
    print(f"  Soft threshold: {config['soft_threshold']}, Hard threshold: {hard_threshold}")
    print(f"  Experiment dir: {experiment_dir}")
    
    # 写入日志头部
    write_log_header(log_file, config)
    
    # 2. 加载训练集 hidden states
    print("\n" + "="*70)
    print("Loading training data...")
    print("="*70)
    
    hs_train_full, train_labels = load_or_extract_train_hidden_states(
        model_cfg['model'], model_cfg['tokenizer'],
        train_data_dir, hidden_states_dir,
        config['model_name'], config['num_samples_per_class'],
        model_cfg['use_chat_template'], model_cfg['use_system_prompt']
    )
    
    train_hidden_states_dict = extract_layer_hidden_states(
        hs_train_full, layer_ids, config['extract_token_position']
    )
    
    print(f"  Training samples: {len(train_labels)}")
    
    # 3. 训练检测器
    print("\n" + "="*70)
    print("Training MultiLayerVotingDetector...")
    print("="*70)
    
    detector = MultiLayerVotingDetector(layers=layer_ids, threshold=config['soft_threshold'])
    detector.fit(train_hidden_states_dict, train_labels, target_fpr_per_layer=config['target_fpr_per_layer'])
    
    # 4. 在训练集上验证
    print("\n" + "="*70)
    print("Evaluating on training set...")
    print("="*70)
    
    train_eval = evaluate_on_dataset(
        detector, train_hidden_states_dict, train_labels,
        layer_ids, config['soft_threshold'], hard_threshold
    )
    
    print(f"\n  [Soft Mode] TPR={train_eval['metrics_soft']['TPR']:.4f}, FPR={train_eval['metrics_soft']['FPR']:.4f}, F1={train_eval['metrics_soft']['F1']:.4f}")
    print(f"  [Hard Mode] TPR={train_eval['metrics_hard']['TPR']:.4f}, FPR={train_eval['metrics_hard']['FPR']:.4f}, F1={train_eval['metrics_hard']['F1']:.4f}")
    for layer_id in layer_ids:
        m = train_eval['metrics_per_layer'][layer_id]
        print(f"  [Layer {layer_id}] TPR={m['TPR']:.4f}, FPR={m['FPR']:.4f}, F1={m['F1']:.4f}")
    
    # 5. 在测试集上评估
    print("\n" + "="*70)
    print("Evaluating on test sets...")
    print("="*70)
    
    dataset_candidates = jload(os.path.join(test_data_dir, 'dataset_candidates.json'))
    num_datasets = min(config['num_test_datasets'], len(dataset_candidates))
    
    eval_results = evaluate_on_multiple_datasets(
        detector, model_cfg['model'], model_cfg['tokenizer'],
        dataset_candidates[:num_datasets], test_data_dir,
        layer_ids, model_cfg['use_chat_template'], model_cfg['use_system_prompt'],
        config['extract_token_position'], config['soft_threshold'], hard_threshold,
        config['test_samples_per_class'], log_file, show_progress=True
    )
    
    # 6. 打印结果
    print("\n" + "="*70)
    print("Detailed Results per Dataset")
    print("="*70)
    
    print_results_table(eval_results['results_soft'], "Aggregated - Soft Mode")
    print_results_table(eval_results['results_hard'], "Aggregated - Hard Mode")
    
    print_summary(eval_results, layer_ids, config)
    print_comparison(eval_results, layer_ids)
    
    print(f"\n  Detailed log saved to: {log_file}")
    
    # 写入日志汇总
    write_log_summary(log_file, eval_results, layer_ids)
    
    # 7. 绘制 ROC 曲线
    roc_save_path = os.path.join(experiment_dir, "roc_curve.svg")
    roc_title = f"ROC Curves - {config['model_name']} (Layers {layer_ids[0]}-{layer_ids[-1]})"
    auc_results = plot_roc_curves(eval_results, layer_ids, roc_save_path, roc_title)
    
    # 打印 AUC 结果
    print("\n[AUC Scores]")
    print("-"*40)
    for method, auc_val in auc_results.items():
        print(f"  {method}: {auc_val:.4f}")
    
    # 8. 保存详细 scores（JSON 格式，用于后续绘图）
    scores_save_path = os.path.join(experiment_dir, "detailed_scores.json")
    save_detailed_scores(eval_results, scores_save_path, config)
    
    # 9. 保存配置文件
    config_save_path = os.path.join(experiment_dir, "config.json")
    config_for_save = {k: v for k, v in config.items() if k != 'detector'}  # 排除不可序列化的对象
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config_for_save, f, indent=2, ensure_ascii=False)
    print(f"[配置已保存到: {config_save_path}]")
    
    # 10. 保存汇总结果（包含每个数据集的完整评估结果）
    summary_save_path = os.path.join(experiment_dir, "summary.json")
    summary_data = {
        'auc_results': auc_results,
        # 每个数据集的评估结果
        'results_per_dataset': {
            'soft': eval_results['results_soft'],  # 每个数据集的 soft 模式结果
            'hard': eval_results['results_hard'],  # 每个数据集的 hard 模式结果
            'per_layer': {str(k): v for k, v in eval_results['results_per_layer'].items()},  # 每层每个数据集的结果
        },
        # 平均指标
        'avg_soft': eval_results['avg_soft'],
        'avg_hard': eval_results['avg_hard'],
        'avg_per_layer': {str(k): v for k, v in eval_results['avg_per_layer'].items()},
        # 总体指标（所有样本聚合）
        'overall_soft': eval_results['overall_soft'],
        'overall_hard': eval_results['overall_hard'],
        'overall_per_layer': {str(k): v for k, v in eval_results['overall_per_layer'].items()},
    }
    with open(summary_save_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"[汇总结果已保存到: {summary_save_path}]")
    
    print(f"\n{'='*70}")
    print(f"所有结果已保存到: {experiment_dir}")
    print(f"{'='*70}")
    
    return {
        'config': config,
        'train_eval': train_eval,
        'test_eval': eval_results,
        'detector': detector,
        'layer_ids': layer_ids,
        'experiment_dir': experiment_dir,
        'log_file': log_file,
        'roc_path': roc_save_path,
        'scores_path': scores_save_path,
        'summary_path': summary_save_path,
        'auc_results': auc_results,
    }


def main():
    args = parse_args()
    
    config = {
        'model_name': args.model_name,
        'num_samples_per_class': args.num_samples_per_class,
        'start_layer_ratio': args.start_layer_ratio,
        'num_layers': args.num_layers,
        'extract_token_position': args.extract_token_position,
        'soft_threshold': args.soft_threshold,
        'hard_threshold': args.hard_threshold,
        'target_fpr_per_layer': args.target_fpr_per_layer,
        'device': args.device,
        'num_test_datasets': args.num_test_datasets,
        'test_samples_per_class': args.test_samples_per_class,
        'log_dir': args.log_dir,
    }
    
    results = run_evaluation(config)
    return results


if __name__ == "__main__":
    main()
