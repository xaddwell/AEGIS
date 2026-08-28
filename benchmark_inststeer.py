"""
AEGIS 检测方法性能基准测试脚本

使用方式:
    # 测试检测模式（LDA分类器）
    python benchmark_inststeer.py --mode detection --model_name llama3.1-8b
    
    # 测试多层投票检测器
    python benchmark_inststeer.py --mode multilayer --model_name llama3.1-8b
    
    # 指定样本数量
    python benchmark_inststeer.py --mode detection --num_samples 500
    
    # 指定LDA模型路径
    python benchmark_inststeer.py --mode detection --lda_model /path/to/lda.pkl
"""

import os

import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from tqdm import tqdm
from pathlib import Path

import torch
import psutil

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("[Warning] GPUtil not found, will use torch.cuda for GPU monitoring")

from inststeer.model import load_model_config
from inststeer.utils.steer import AsymmetricLDA, MultiLayerVotingDetector
from inststeer.dataset import get_formatted_data
from inststeer.utils.hidden_state import get_hidden_states_full, extract_hidden_states_layer_token
from inststeer.utils import seed_everything, load_pickle, save_pickle
from datasets import Dataset, concatenate_datasets

seed_everything(42)


# =============================================================================
# 显存和内存监控工具（与baseline相同）
# =============================================================================

def get_gpu_memory_usage(device_id: int = 0) -> Dict[str, float]:
    """获取GPU显存使用情况（单位: MB）"""
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            if device_id < len(gpus):
                gpu = gpus[device_id]
                return {
                    'used_mb': gpu.memoryUsed,
                    'total_mb': gpu.memoryTotal,
                    'utilization_percent': gpu.memoryUtil * 100,
                }
        except Exception as e:
            print(f"[Warning] Failed to get GPU memory via GPUtil: {e}")
    
    # 备用方案：使用 torch
    if torch.cuda.is_available():
        try:
            used = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
            return {
                'used_mb': used,
                'reserved_mb': reserved,
                'total_mb': torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 2),
            }
        except Exception as e:
            print(f"[Warning] Failed to get GPU memory via torch: {e}")
    
    return {'used_mb': 0, 'total_mb': 0}


def get_cpu_memory_usage() -> Dict[str, float]:
    """获取CPU内存使用情况（单位: MB）"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / (1024 ** 2),  # 实际物理内存
        'vms_mb': mem_info.vms / (1024 ** 2),  # 虚拟内存
    }


def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# 数据准备
# =============================================================================

def load_test_data(
    clean_dir: str,
    malicious_dir: str,
    tokenizer,
    use_chat_template: bool,
    use_system_prompt: bool,
    num_samples_per_class: int = 100,
) -> Dict[str, Any]:
    """
    使用项目标准的 get_formatted_data 函数加载测试数据
    
    Returns:
        包含 texts 和 labels 的字典
    """
    # 使用 get_formatted_data 加载数据（这是项目的标准方式）
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
    
    # 创建数据集并采样
    clean_dataset = Dataset.from_dict({"text": clean_data, "label": [0] * len(clean_data)})
    malicious_dataset = Dataset.from_dict({"text": malicious_data, "label": [1] * len(malicious_data)})
    
    n_clean = min(num_samples_per_class, len(clean_dataset))
    n_malicious = min(num_samples_per_class, len(malicious_dataset))
    
    clean_sample = clean_dataset.shuffle(seed=42).select(range(n_clean))
    malicious_sample = malicious_dataset.shuffle(seed=42).select(range(n_malicious))
    
    # 合并数据
    combined_dataset = concatenate_datasets([clean_sample, malicious_sample])
    texts = list(combined_dataset['text'])
    labels = np.array(combined_dataset['label'])
    
    return {
        'texts': texts,
        'labels': labels,
    }


def prepare_benchmark_data(
    model_name: str,
    num_samples: int = 200,
    test_data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'TestData'),
) -> Dict[str, Any]:
    """准备基准测试数据"""
    print(f"\n[准备基准测试数据: {num_samples} 个样本]")
    
    # 加载模型配置（需要tokenizer）
    cfg = load_model_config(model_name)
    tokenizer = cfg['tokenizer']
    use_chat_template = cfg['use_chat_template']
    use_system_prompt = cfg['use_system_prompt']
    
    # 加载 dataset_candidates
    candidates_path = os.path.join(test_data_dir, 'dataset_candidates.json')
    with open(candidates_path, 'r', encoding='utf-8') as f:
        dataset_candidates = json.load(f)
    
    # 选择第一个数据集
    clean_name, malicious_name = dataset_candidates[0]
    clean_dir = os.path.join(test_data_dir, clean_name)
    malicious_dir = os.path.join(test_data_dir, malicious_name)
    
    print(f"  使用数据集: {malicious_name}")
    
    # 加载数据（每类一半）
    samples_per_class = num_samples // 2
    data = load_test_data(
        clean_dir, malicious_dir,
        tokenizer, use_chat_template, use_system_prompt,
        samples_per_class
    )
    
    print(f"  实际加载: {len(data['texts'])} 个样本")
    print(f"    - Clean: {sum(data['labels'] == 0)}")
    print(f"    - Malicious: {sum(data['labels'] == 1)}")
    
    return data


# =============================================================================
# 检测模式：LDA分类器
# =============================================================================

def benchmark_lda_detection(
    model_name: str,
    lda_model_path: str,
    texts: List[str],
    device_id: int = 0,
    extract_layer_position: float = 0.8,
    extract_token_position: str = "last",
    warmup_runs: int = 3,
) -> Dict[str, Any]:
    """
    测试 LDA 检测器的性能
    
    这个模式只测试检测性能，不进行文本生成
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: LDA Detection Mode")
    print(f"{'='*70}")
    
    # 清理缓存
    clear_gpu_cache()
    time.sleep(1)
    
    # 记录初始显存
    initial_gpu_mem = get_gpu_memory_usage(device_id)
    initial_cpu_mem = get_cpu_memory_usage()
    
    print(f"\n[初始状态]")
    print(f"  GPU显存: {initial_gpu_mem.get('used_mb', 0):.2f} MB")
    print(f"  CPU内存: {initial_cpu_mem['rss_mb']:.2f} MB")
    
    # 加载模型
    print(f"\n[加载模型...]")
    load_start = time.time()
    
    try:
        cfg = load_model_config(model_name)
        model = cfg['model'].eval().to(f"cuda:{device_id}")
        tokenizer = cfg['tokenizer']
        
        # 加载 LDA 模型
        lda = AsymmetricLDA()
        lda.load(lda_model_path)
        
        extract_layer_ids = lda.layer_id
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return {
            'method': 'lda_detection',
            'error': str(e),
            'status': 'failed',
        }
    
    load_time = time.time() - load_start
    
    # 记录加载后显存
    after_load_gpu_mem = get_gpu_memory_usage(device_id)
    after_load_cpu_mem = get_cpu_memory_usage()
    
    model_gpu_memory = after_load_gpu_mem.get('used_mb', 0) - initial_gpu_mem.get('used_mb', 0)
    model_cpu_memory = after_load_cpu_mem['rss_mb'] - initial_cpu_mem['rss_mb']
    
    print(f"  加载时间: {load_time:.2f}s")
    print(f"  模型显存占用: {model_gpu_memory:.2f} MB")
    print(f"  模型CPU内存占用: {model_cpu_memory:.2f} MB")
    print(f"  检测层: Layer {extract_layer_ids}")
    
    # 预热运行
    print(f"\n[预热运行 {warmup_runs} 次...]")
    warmup_texts = texts[:min(5, len(texts))]
    
    for i in range(warmup_runs):
        try:
            for text in warmup_texts:
                inputs = tokenizer(text, return_tensors="pt").to(f"cuda:{device_id}")
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[extract_layer_ids]
                    last_token_hidden = hidden_states[:, -1, :].cpu().numpy()
                    _ = lda.decision_function(last_token_hidden)
            clear_gpu_cache()
        except Exception as e:
            print(f"  [Warning] Warmup run {i+1} failed: {e}")
    
    # 正式推理测试
    print(f"\n[推理测试: {len(texts)} 个样本...]")
    
    inference_times = []
    peak_gpu_memory = after_load_gpu_mem.get('used_mb', 0)
    peak_cpu_memory = after_load_cpu_mem['rss_mb']
    
    all_scores = []
    all_predictions = []
    
    for text in tqdm(texts, desc="检测中"):
        sample_start = time.time()
        
        try:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(f"cuda:{device_id}")
            
            # 前向传播获取隐藏状态
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[extract_layer_ids]
                last_token_hidden = hidden_states[:, -1, :].cpu().numpy()
            
            # LDA 检测
            score = lda.decision_function(last_token_hidden)[0]
            pred = 1 if score > 0 else 0
            
            sample_time = time.time() - sample_start
            inference_times.append(sample_time)
            
            all_scores.append(score)
            all_predictions.append(pred)
            
            # 记录峰值显存
            current_gpu_mem = get_gpu_memory_usage(device_id)
            current_cpu_mem = get_cpu_memory_usage()
            
            peak_gpu_memory = max(peak_gpu_memory, current_gpu_mem.get('used_mb', 0))
            peak_cpu_memory = max(peak_cpu_memory, current_cpu_mem['rss_mb'])
            
        except Exception as e:
            print(f"\n[ERROR] Sample failed: {e}")
            inference_times.append(0)
            all_scores.append(0)
            all_predictions.append(0)
    
    # 计算统计指标
    total_inference_time = sum(inference_times)
    avg_time_per_sample = total_inference_time / len(texts) if len(texts) > 0 else 0
    throughput = len(texts) / total_inference_time if total_inference_time > 0 else 0
    
    # 计算推理时的显存增量
    inference_gpu_memory = peak_gpu_memory - after_load_gpu_mem.get('used_mb', 0)
    inference_cpu_memory = peak_cpu_memory - after_load_cpu_mem['rss_mb']
    
    # 总显存占用
    total_gpu_memory = peak_gpu_memory - initial_gpu_mem.get('used_mb', 0)
    total_cpu_memory = peak_cpu_memory - initial_cpu_mem['rss_mb']
    
    # 打印结果
    print(f"\n[性能统计]")
    print(f"  总推理时间: {total_inference_time:.2f}s")
    print(f"  平均每样本时间: {avg_time_per_sample*1000:.2f}ms")
    print(f"  吞吐量: {throughput:.2f} samples/s")
    print(f"\n[显存占用]")
    print(f"  模型加载显存: {model_gpu_memory:.2f} MB")
    print(f"  推理峰值增量: {inference_gpu_memory:.2f} MB")
    print(f"  总显存占用: {total_gpu_memory:.2f} MB")
    print(f"\n[CPU内存占用]")
    print(f"  模型加载内存: {model_cpu_memory:.2f} MB")
    print(f"  推理峰值增量: {inference_cpu_memory:.2f} MB")
    print(f"  总内存占用: {total_cpu_memory:.2f} MB")
    
    # 清理
    del model, tokenizer, lda
    clear_gpu_cache()
    time.sleep(1)
    
    return {
        'method': 'lda_detection',
        'mode': 'detection',
        'status': 'success',
        'num_samples': len(texts),
        'model_name': model_name,
        'detection_layer': extract_layer_ids,
        
        # 时间指标
        'load_time_s': load_time,
        'total_inference_time_s': total_inference_time,
        'avg_time_per_sample_ms': avg_time_per_sample * 1000,
        'throughput_samples_per_s': throughput,
        
        # GPU显存指标
        'model_gpu_memory_mb': model_gpu_memory,
        'inference_gpu_memory_mb': inference_gpu_memory,
        'total_gpu_memory_mb': total_gpu_memory,
        'peak_gpu_memory_mb': peak_gpu_memory,
        
        # CPU内存指标
        'model_cpu_memory_mb': model_cpu_memory,
        'inference_cpu_memory_mb': inference_cpu_memory,
        'total_cpu_memory_mb': total_cpu_memory,
        'peak_cpu_memory_mb': peak_cpu_memory,
    }


# =============================================================================
# 多层投票检测模式
# =============================================================================

def benchmark_multilayer_detection(
    model_name: str,
    lda_models_dir: str,
    texts: List[str],
    device_id: int = 0,
    layer_range: tuple = (0.5, 0.6, 0.7, 0.8),
    extract_token_position: str = "last",
    batch_size: int = 16,
    warmup_runs: int = 3,
    train_samples_per_class: int = 200,
    target_fpr_per_layer: float = 0.05,
) -> Dict[str, Any]:
    """
    测试多层投票检测器的性能
    
    Args:
        model_name: 模型名称
        lda_models_dir: LDA模型目录
        texts: 测试文本列表
        device_id: GPU设备ID
        layer_range: 要使用的层位置比例列表
        extract_token_position: token提取位置
        warmup_runs: 预热次数
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: Multi-Layer Voting Detection Mode")
    print(f"{'='*70}")
    
    # 清理缓存
    clear_gpu_cache()
    time.sleep(1)
    
    # 记录初始显存
    initial_gpu_mem = get_gpu_memory_usage(device_id)
    initial_cpu_mem = get_cpu_memory_usage()
    
    print(f"\n[初始状态]")
    print(f"  GPU显存: {initial_gpu_mem.get('used_mb', 0):.2f} MB")
    print(f"  CPU内存: {initial_cpu_mem['rss_mb']:.2f} MB")
    
    # 加载模型
    print(f"\n[加载模型和多层LDA检测器...]")
    load_start = time.time()
    
    try:
        cfg = load_model_config(model_name)
        model = cfg['model'].eval().to(f"cuda:{device_id}")
        tokenizer = cfg['tokenizer']
        num_layers = cfg['num_hidden_layers']
        
        # 计算检测层索引
        layer_ids = [int(num_layers * pos) for pos in layer_range]
        print(f"  检测层: {layer_ids}")
        
        # 尝试加载预训练的LDA模型
        print(f"\n  尝试加载预训练的LDA模型...")
        lda_detectors = {}
        missing_models = []
        
        for layer_id in layer_ids:
            lda_path = os.path.join(
                lda_models_dir,
                f'{model_name}_num:200_layer:{layer_id}_token:{extract_token_position}.pkl'
            )
            
            if os.path.exists(lda_path):
                try:
                    lda = AsymmetricLDA()
                    lda.load(lda_path)
                    lda_detectors[layer_id] = lda
                    print(f"    ✓ Layer {layer_id}: 已加载预训练模型")
                except Exception as e:
                    print(f"    ✗ Layer {layer_id}: 加载失败 - {e}")
                    missing_models.append(layer_id)
            else:
                missing_models.append(layer_id)
        
        # 如果缺少模型，临时训练
        if missing_models:
            print(f"\n  未找到预训练模型的层: {missing_models}")
            print(f"  将临时训练这些层的LDA检测器...")
            
            # 加载训练数据
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            train_data_dir = os.path.join(data_dir, 'TrainData/data')
            hidden_states_dir = os.path.join(data_dir, 'hidden_states')
            
            print(f"\n  加载训练数据...")
            # 检查是否有缓存的hidden states
            hs_file_path = os.path.join(hidden_states_dir, f"hs_{model_name}_{train_samples_per_class}")
            label_file_path = os.path.join(hidden_states_dir, f"label_{model_name}_{train_samples_per_class}")
            
            if os.path.exists(hs_file_path + ".pkl"):
                print(f"    从缓存加载 hidden states...")
                hs_train_full = load_pickle(hs_file_path)
                train_labels = load_pickle(label_file_path)
            else:
                print(f"    提取训练集 hidden states...")
                train_data, train_label = get_formatted_data(
                    customized_instruction=False,
                    path=train_data_dir,
                    tokenizer=tokenizer,
                    use_chat_template=cfg['use_chat_template'],
                    use_system_prompt=cfg['use_system_prompt']
                )
                
                full_dataset = Dataset.from_dict({"text": train_data, "label": train_label})
                dataset_label_0 = full_dataset.filter(lambda x: x['label'] == 0)
                dataset_label_1 = full_dataset.filter(lambda x: x['label'] == 1)
                
                sample_0 = dataset_label_0.shuffle(seed=42).select(range(min(train_samples_per_class, len(dataset_label_0))))
                sample_1 = dataset_label_1.shuffle(seed=42).select(range(min(train_samples_per_class, len(dataset_label_1))))
                train_dataset = concatenate_datasets([sample_0, sample_1]).shuffle(seed=42)
                train_labels = np.array(train_dataset['label'])
                
                hs_train_full = get_hidden_states_full(
                    model, tokenizer,
                    train_dataset,
                    prompt_key="text",
                    batch_size=32,
                    show_progress=True
                )
                
                # 保存缓存
                os.makedirs(hidden_states_dir, exist_ok=True)
                save_pickle(hs_train_full, hs_file_path)
                save_pickle(train_labels, label_file_path)
            
            print(f"    训练样本数: {len(train_labels)}")
            
            # 为缺失的层训练LDA
            print(f"\n  训练缺失层的LDA检测器...")
            for layer_id in missing_models:
                print(f"    训练 Layer {layer_id}...")
                # 提取该层的hidden states
                hs_layer = extract_hidden_states_layer_token(
                    hs_train_full,
                    layer_ids=[layer_id],
                    token_position=extract_token_position
                ).squeeze(axis=1)
                
                # 训练LDA
                lda = AsymmetricLDA(concentrated_label=1, layer_id=layer_id, token_idx=extract_token_position)
                lda.fit(hs_layer, train_labels)
                lda_detectors[layer_id] = lda
                print(f"      ✓ Layer {layer_id}: 训练完成")
        
        # 使用MultiLayerVotingDetector进行训练（设置阈值）
        print(f"\n  初始化多层投票检测器...")
        voting_detector = MultiLayerVotingDetector(
            layers=layer_ids,
            threshold=0.0
        )
        
        # 如果有临时训练的模型，需要设置阈值
        if missing_models:
            print(f"  为临时训练的模型设置阈值 (target FPR={target_fpr_per_layer})...")
            # 提取训练集的hidden states
            train_hidden_states_dict = {}
            hs_file_path = os.path.join(hidden_states_dir, f"hs_{model_name}_{train_samples_per_class}")
            label_file_path = os.path.join(hidden_states_dir, f"label_{model_name}_{train_samples_per_class}")
            
            hs_train_full = load_pickle(hs_file_path)
            train_labels = load_pickle(label_file_path)
            
            for layer_id in layer_ids:
                hs_layer = extract_hidden_states_layer_token(
                    hs_train_full,
                    layer_ids=[layer_id],
                    token_position=extract_token_position
                ).squeeze(axis=1)
                train_hidden_states_dict[layer_id] = hs_layer
            
            # 设置检测器
            voting_detector.detectors = lda_detectors
            voting_detector.fit(train_hidden_states_dict, train_labels, target_fpr_per_layer=target_fpr_per_layer)
        else:
            # 全部使用预训练模型
            voting_detector.detectors = lda_detectors
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return {
            'method': 'multilayer_detection',
            'error': str(e),
            'status': 'failed',
        }
    
    load_time = time.time() - load_start
    
    # 记录加载后显存
    after_load_gpu_mem = get_gpu_memory_usage(device_id)
    after_load_cpu_mem = get_cpu_memory_usage()
    
    model_gpu_memory = after_load_gpu_mem.get('used_mb', 0) - initial_gpu_mem.get('used_mb', 0)
    model_cpu_memory = after_load_cpu_mem['rss_mb'] - initial_cpu_mem['rss_mb']
    
    print(f"  加载时间: {load_time:.2f}s")
    print(f"  模型显存占用: {model_gpu_memory:.2f} MB")
    print(f"  模型CPU内存占用: {model_cpu_memory:.2f} MB")
    print(f"  检测层: {layer_ids}")
    
    # 预热运行
    print(f"\n[预热运行 {warmup_runs} 次...]")
    warmup_texts = texts[:min(batch_size, len(texts))]
    
    for i in range(warmup_runs):
        try:
            # 批量预热
            inputs = tokenizer(warmup_texts, padding=True, truncation=True, 
                             return_tensors="pt").to(f"cuda:{device_id}")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            clear_gpu_cache()
        except Exception as e:
            print(f"  [Warning] Warmup run {i+1} failed: {e}")
    
    # 正式推理测试
    print(f"\n[推理测试: {len(texts)} 个样本, batch_size={batch_size}]")
    
    inference_times = []
    peak_gpu_memory = after_load_gpu_mem.get('used_mb', 0)
    peak_cpu_memory = after_load_cpu_mem['rss_mb']
    
    all_scores = []
    all_predictions = []
    
    # 分批推理
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="多层检测中"):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]
        batch_start = time.time()
        
        try:
            # 批量 Tokenize
            inputs = tokenizer(batch_texts, padding=True, truncation=True,
                             return_tensors="pt").to(f"cuda:{device_id}")
            
            # 批量前向传播获取多层隐藏状态
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # 对批次中的每个样本进行检测
                for j in range(len(batch_texts)):
                    # 提取该样本的多层隐藏状态
                    hidden_states_dict = {}
                    for layer_id in layer_ids:
                        hidden_states = outputs.hidden_states[layer_id]
                        # 取第 j 个样本的最后一个 token
                        last_token_hidden = hidden_states[j:j+1, -1, :].cpu().numpy()
                        hidden_states_dict[layer_id] = last_token_hidden
                    
                    # 多层投票检测
                    result = voting_detector.predict(hidden_states_dict)
                    score = result['soft_aggregate_score'][0]
                    pred = result['soft_pred'][0]
                    
                    all_scores.append(score)
                    all_predictions.append(pred)
            
            batch_time = time.time() - batch_start
            inference_times.append(batch_time)
            
            # 记录峰值显存
            current_gpu_mem = get_gpu_memory_usage(device_id)
            current_cpu_mem = get_cpu_memory_usage()
            
            peak_gpu_memory = max(peak_gpu_memory, current_gpu_mem.get('used_mb', 0))
            peak_cpu_memory = max(peak_cpu_memory, current_cpu_mem['rss_mb'])
            
        except Exception as e:
            print(f"\n[ERROR] Batch {i} failed: {e}")
            import traceback
            traceback.print_exc()
            # 为该批次的所有样本填充默认值
            batch_time = time.time() - batch_start
            inference_times.append(batch_time)
            for _ in range(len(batch_texts)):
                all_scores.append(0)
                all_predictions.append(0)
    
    # 计算统计指标
    total_inference_time = sum(inference_times)
    avg_time_per_sample = total_inference_time / len(texts) if len(texts) > 0 else 0
    avg_time_per_batch = np.mean(inference_times) if inference_times else 0
    throughput = len(texts) / total_inference_time if total_inference_time > 0 else 0
    
    # 计算推理时的显存增量
    inference_gpu_memory = peak_gpu_memory - after_load_gpu_mem.get('used_mb', 0)
    inference_cpu_memory = peak_cpu_memory - after_load_cpu_mem['rss_mb']
    
    # 总显存占用
    total_gpu_memory = peak_gpu_memory - initial_gpu_mem.get('used_mb', 0)
    total_cpu_memory = peak_cpu_memory - initial_cpu_mem['rss_mb']
    
    # 打印结果
    print(f"\n[性能统计]")
    print(f"  总推理时间: {total_inference_time:.2f}s")
    print(f"  平均每样本时间: {avg_time_per_sample*1000:.2f}ms")
    print(f"  平均每批次时间: {avg_time_per_batch:.2f}s")
    print(f"  吞吐量: {throughput:.2f} samples/s")
    print(f"\n[显存占用]")
    print(f"  模型加载显存: {model_gpu_memory:.2f} MB")
    print(f"  推理峰值增量: {inference_gpu_memory:.2f} MB")
    print(f"  总显存占用: {total_gpu_memory:.2f} MB")
    print(f"\n[CPU内存占用]")
    print(f"  模型加载内存: {model_cpu_memory:.2f} MB")
    print(f"  推理峰值增量: {inference_cpu_memory:.2f} MB")
    print(f"  总内存占用: {total_cpu_memory:.2f} MB")
    
    # 清理
    del model, tokenizer, lda_detectors, voting_detector
    clear_gpu_cache()
    time.sleep(1)
    
    return {
        'method': 'multilayer_detection',
        'mode': 'multilayer',
        'status': 'success',
        'num_samples': len(texts),
        'batch_size': batch_size,
        'num_batches': num_batches,
        'model_name': model_name,
        'detection_layers': layer_ids,
        'num_layers': len(layer_ids),
        
        # 时间指标
        'load_time_s': load_time,
        'total_inference_time_s': total_inference_time,
        'avg_time_per_sample_ms': avg_time_per_sample * 1000,
        'avg_time_per_batch_s': avg_time_per_batch,
        'throughput_samples_per_s': throughput,
        
        # GPU显存指标
        'model_gpu_memory_mb': model_gpu_memory,
        'inference_gpu_memory_mb': inference_gpu_memory,
        'total_gpu_memory_mb': total_gpu_memory,
        'peak_gpu_memory_mb': peak_gpu_memory,
        
        # CPU内存指标
        'model_cpu_memory_mb': model_cpu_memory,
        'inference_cpu_memory_mb': inference_cpu_memory,
        'total_cpu_memory_mb': total_cpu_memory,
        'peak_cpu_memory_mb': peak_cpu_memory,
    }


# =============================================================================
# 结果保存
# =============================================================================

def save_benchmark_results(
    results: Dict[str, Any],
    output_dir: str = 'logs/benchmark_inststeer',
) -> None:
    """保存基准测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = results.get('mode', 'unknown')
    
    # 保存完整的 JSON 结果
    json_path = os.path.join(output_dir, f'benchmark_{mode}_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[完整结果已保存到: {json_path}]")
    
    if results.get('status') != 'success':
        print("[WARNING] Benchmark failed, no summary generated!")
        return
    
    # 打印汇总表格
    print("\n" + "="*100)
    print("AEGIS DETECTION BENCHMARK SUMMARY")
    print("="*100)
    print(f"\nMethod: {results['method']}")
    print(f"Mode: {results['mode']}")
    print(f"Model: {results['model_name']}")
    print(f"Samples: {results['num_samples']}")
    print(f"\nPerformance:")
    print(f"  - Load Time: {results['load_time_s']:.2f}s")
    print(f"  - Total Inference Time: {results['total_inference_time_s']:.2f}s")
    print(f"  - Avg Time per Sample: {results['avg_time_per_sample_ms']:.2f}ms")
    print(f"  - Throughput: {results['throughput_samples_per_s']:.2f} samples/s")
    print(f"\nMemory:")
    print(f"  - Model GPU Memory: {results['model_gpu_memory_mb']:.2f} MB")
    print(f"  - Total GPU Memory: {results['total_gpu_memory_mb']:.2f} MB")
    print(f"  - Peak GPU Memory: {results['peak_gpu_memory_mb']:.2f} MB")
    print(f"  - Total CPU Memory: {results['total_cpu_memory_mb']:.2f} MB")
    print("="*100)
    
    # 保存 Markdown 格式的报告
    md_path = os.path.join(output_dir, f'benchmark_{mode}_{timestamp}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# AEGIS Detection Performance Benchmark\n\n")
        f.write(f"**测试时间**: {timestamp}\n\n")
        f.write(f"**方法**: {results['method']}\n\n")
        f.write(f"**模式**: {results['mode']}\n\n")
        f.write(f"**模型**: {results['model_name']}\n\n")
        f.write(f"**样本数量**: {results['num_samples']}\n\n")
        
        if results['mode'] == 'detection':
            f.write(f"**检测层**: Layer {results['detection_layer']}\n\n")
        elif results['mode'] == 'multilayer':
            f.write(f"**检测层**: {results['detection_layers']}\n\n")
            f.write(f"**层数**: {results['num_layers']}\n\n")
            f.write(f"**批次大小**: {results.get('batch_size', 1)}\n\n")
            f.write(f"**批次数量**: {results.get('num_batches', len(results.get('texts', [])))}\n\n")
        
        f.write("## 性能指标\n\n")
        f.write(f"- **加载时间**: {results['load_time_s']:.2f}s\n")
        f.write(f"- **总推理时间**: {results['total_inference_time_s']:.2f}s\n")
        f.write(f"- **平均每样本时间**: {results['avg_time_per_sample_ms']:.2f}ms\n")
        if 'avg_time_per_batch_s' in results:
            f.write(f"- **平均每批次时间**: {results['avg_time_per_batch_s']:.2f}s\n")
        f.write(f"- **吞吐量**: {results['throughput_samples_per_s']:.2f} samples/s\n\n")
        
        f.write("## 显存占用\n\n")
        f.write(f"- **模型加载显存**: {results['model_gpu_memory_mb']:.2f} MB\n")
        f.write(f"- **推理显存增量**: {results['inference_gpu_memory_mb']:.2f} MB\n")
        f.write(f"- **总显存占用**: {results['total_gpu_memory_mb']:.2f} MB\n")
        f.write(f"- **峰值显存**: {results['peak_gpu_memory_mb']:.2f} MB\n\n")
        
        f.write("## CPU内存占用\n\n")
        f.write(f"- **模型加载内存**: {results['model_cpu_memory_mb']:.2f} MB\n")
        f.write(f"- **推理内存增量**: {results['inference_cpu_memory_mb']:.2f} MB\n")
        f.write(f"- **总内存占用**: {results['total_cpu_memory_mb']:.2f} MB\n")
    
    print(f"[Markdown报告已保存到: {md_path}]")


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AEGIS 检测性能基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试检测模式（单层）
  python benchmark_inststeer.py --mode detection --model_name llama3.1-8b
  
  # 测试多层投票检测模式（自动生成层范围）
  python benchmark_inststeer.py --mode multilayer --model_name llama3.1-8b --start_layer 0.5 --num_layers 4
  
  # 测试多层投票检测模式（手动指定层范围）
  python benchmark_inststeer.py --mode multilayer --model_name llama3.1-8b --layer_range 0.5,0.6,0.7,0.8
  
  # 指定样本数量
  python benchmark_inststeer.py --mode detection --num_samples 500
        """
    )
    parser.add_argument("--mode", type=str, default="detection",
                        choices=['detection', 'multilayer'],
                        help="测试模式: detection(单层检测), multilayer(多层投票检测)")
    parser.add_argument("--model_name", type=str, default="llama3.1-8b",
                        help="模型名称 (默认: llama3.1-8b)")
    parser.add_argument("--lda_model", type=str, default=None,
                        help="LDA模型路径 (默认: 自动查找)")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="测试样本数量 (默认: 200)")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU 设备编号 (默认: 0)")
    parser.add_argument("--output_dir", type=str, default="logs/benchmark_inststeer",
                        help="输出目录 (默认: logs/benchmark_inststeer)")
    parser.add_argument("--warmup_runs", type=int, default=3,
                        help="预热运行次数 (默认: 3)")
    parser.add_argument("--extract_layer_position", type=float, default=0.5,
                        help="提取层位置比例 (默认: 0.8)")
    parser.add_argument("--extract_token_position", type=str, default="last",
                        help="提取token位置 (默认: last)")
    parser.add_argument("--layer_range", type=str, default=None,
                        help="多层检测的层位置比例，逗号分隔 (仅multilayer模式, 例如: 0.5,0.6,0.7,0.8)")
    parser.add_argument("--start_layer", type=float, default=0.5,
                        help="起始层位置比例 (仅multilayer模式, 默认: 0.5)")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="使用的层数 (仅multilayer模式, 默认: 4)")
    parser.add_argument("--layer_step", type=float, default=0.1,
                        help="层间隔比例 (仅multilayer模式, 默认: 0.1)")
    parser.add_argument("--train_samples_per_class", type=int, default=200,
                        help="训练LDA时每类样本数 (仅multilayer模式, 默认: 200)")
    parser.add_argument("--target_fpr_per_layer", type=float, default=0.05,
                        help="每层目标FPR (仅multilayer模式, 默认: 0.05)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小 (默认: 32)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("AEGIS DETECTION PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"模式: {args.mode}")
    print(f"模型: {args.model_name}")
    print(f"样本数量: {args.num_samples}")
    print(f"批次大小: {args.batch_size}")
    print(f"GPU设备: cuda:{args.device}")
    print(f"输出目录: {args.output_dir}")
    
    # 准备测试数据
    data = prepare_benchmark_data(model_name=args.model_name, num_samples=args.num_samples)
    texts = data['texts']
    
    # 确定 LDA 模型路径
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    lda_models_dir = os.path.join(data_dir, 'models')
    
    if args.mode == 'multilayer':
        # 多层模式使用模型目录
        print(f"LDA模型目录: {lda_models_dir}")
        
        # 解析层范围：优先使用 layer_range，否则使用 start_layer + num_layers
        if args.layer_range is not None:
            # 手动指定层范围
            layer_range = tuple([float(x.strip()) for x in args.layer_range.split(',')])
            print(f"层位置比例（手动指定）: {layer_range}")
        else:
            # 根据起始层和层数自动生成
            layer_range = tuple([
                min(args.start_layer + i * args.layer_step, 1.0) 
                for i in range(args.num_layers)
            ])
            print(f"层位置比例（自动生成）:")
            print(f"  起始层: {args.start_layer}")
            print(f"  层数: {args.num_layers}")
            print(f"  间隔: {args.layer_step}")
            print(f"  实际层范围: {layer_range}")
        
        results = benchmark_multilayer_detection(
            model_name=args.model_name,
            lda_models_dir=lda_models_dir,
            texts=texts,
            device_id=args.device,
            batch_size=args.batch_size,
            layer_range=layer_range,
            extract_token_position=args.extract_token_position,
            warmup_runs=args.warmup_runs,
            train_samples_per_class=args.train_samples_per_class,
            target_fpr_per_layer=args.target_fpr_per_layer,
        )
    else:
        # 单层模式使用单个模型文件
        if args.lda_model is None:
            from inststeer.model import load_model_config
            cfg = load_model_config(args.model_name)
            extract_layer_ids = int(cfg['num_hidden_layers'] * args.extract_layer_position)
            
            lda_model_path = os.path.join(
                lda_models_dir,
                f'{args.model_name}_num:200_layer:{extract_layer_ids}_token:{args.extract_token_position}.pkl'
            )
        else:
            lda_model_path = args.lda_model
        
        print(f"LDA模型路径: {lda_model_path}")
        
        # 运行基准测试
        if args.mode == 'detection':
            results = benchmark_lda_detection(
                model_name=args.model_name,
                lda_model_path=lda_model_path,
                texts=texts,
                device_id=args.device,
                extract_layer_position=args.extract_layer_position,
                extract_token_position=args.extract_token_position,
                warmup_runs=args.warmup_runs,
            )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    
    # 保存结果
    save_benchmark_results(results, output_dir=args.output_dir)
    
    print("\n" + "="*70)
    print("基准测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()

