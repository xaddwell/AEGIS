<div align="center">

# 🛡️ AEGIS: Beyond Over-Refusal

### Defending Indirect Prompt Injection via Latent Instruction Manifolds

[![arXiv](https://img.shields.io/badge/arXiv-2608.22248-B31B1B.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.22248)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)](#license)

**EMNLP 2026 (Findings)** · Official code release

📄 **Paper:** [arXiv:2608.22248](https://arxiv.org/abs/2608.22248) — *Beyond Over-Refusal: Defending Indirect Prompt Injection via Latent Instruction Manifolds*

</div>

---

## Overview

**AEGIS** (*Adaptive Ensemble Guard for Injection Shielding*) is a lightweight, **training-free** defense against indirect prompt injection (IPI). It exploits the spectral asymmetry between the compact *instruction manifold* and the high-entropy *knowledge manifold* induced by instruction tuning:

| Component | Idea |
|---|---|
| 🎯 **Instruction-Sensitive Projector** | Closed-form direction `w ∝ Σ_I⁻¹(μ_I − μ_K)` (Asymmetric LDA + Ledoit–Wolf shrinkage) that isolates "imperative intent" from background knowledge |
| ⚖️ **Distribution-Aware Safety Calibration** | Decision boundary anchored to the empirical quantile of benign scores, mathematically bounding the false positive rate |
| 🗳️ **Unified Multi-Layer Consensus** | Hard/soft voting across network depths — a sample is flagged only when the malicious signal persists across layers (exponential FPR bound via Hoeffding) |

**Headline results:** average accuracy **> 98%** against 8 IPI attacks with a consistent **0.29%** false positive rate on benign workloads, at **~120 ms** overhead — Pareto-optimal among ten baselines.

## Table of contents

- [Repository structure](#repository-structure)
- [Quickstart](#quickstart)
- [Usage](#usage)
  - [1. Train — fit the projector](#1-train--fit-the-projector)
  - [2. Test — inference & benchmarking](#2-test--inference--benchmarking)
  - [3. Demo — end-to-end detection](#3-demo--end-to-end-detection)
- [Training data](#training-data)
- [Data format](#data-format)
- [Citation](#citation)

## Repository structure

```
.
├── inststeer/                    # core package
│   ├── utils/
│   │   ├── steer.py              # AsymmetricLDA, CalibratedAsymmetricLDA, MultiLayerVotingDetector  ← core algorithm
│   │   ├── hidden_state.py       # hidden-state extraction
│   │   └── module.py             # weight-editing helpers
│   ├── model/                    # model registry (config.json) + loader
│   └── dataset/                  # data loading & prompt formatting
├── train.py                      # training: fit the instruction-sensitive projector
├── test.py                       # inference/test: evaluate the fitted projector
├── steering.py                   # SteeringDefense / SteeringConfig (defense mode)
├── benchmark_inststeer.py        # main benchmark (detection / defense / multilayer)
├── eval_multilayer_voting.py     # multi-layer consensus evaluation
├── demo.py                       # quick end-to-end inference demo
└── data/
    ├── TrainData/data/           # demo calibration set (50 benign + 50 injected)
    └── TestData/                 # demo test set (20 benign + 20 injected)
```

## Quickstart

```bash
# Python >= 3.10 with a CUDA-capable GPU recommended
pip install -e .
```

Models are loaded from `$HF_HOME/<repo-id>` (flat layout; `HF_HOME` defaults to `~/.cache/huggingface`):

```bash
HF_HOME=${HF_HOME:-~/.cache/huggingface}
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir $HF_HOME/Qwen/Qwen3-4B-Instruct-2507
# Llama models require accepting the license on Hugging Face first:
# huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir $HF_HOME/meta-llama/Llama-3.1-8B-Instruct
```

> 💡 If huggingface.co is unreachable from your network, set `HF_ENDPOINT=https://hf-mirror.com` before downloading.

## Usage

### 1. Train — fit the projector

```bash
python train.py \
    --model_name qwen3-4b \
    --num_samples_per_class 50 \
    --extract_layer_position 0.5 \
    --extract_token_position last \
    --device 0
```

Formats the calibration set (`data/TrainData/data`), extracts hidden states at the selected layer, fits `AsymmetricLDA`, and saves the projector to `data/models/`.

**Parameter selection (as used in the paper):**

| Parameter | Value | Note |
|---|---|---|
| `num_samples_per_class` | 200 | performance converges already at ~100 per class |
| `extract_layer_position` | 0.5 | middle layers give the best ACC / lowest FPR (layers 18–27 on Qwen3-4B) |
| voting window `\|L\|` | 3 (Qwen3-4B), 7 (Llama-3/3.1-8B) | window starts at 50% network depth |
| per-layer target FPR | 0.05 | distribution-aware calibration; global FPR 0.29% after consensus |
| aggregation | soft voting | see `eval_multilayer_voting.py` for hard/soft modes |

### 2. Test — inference & benchmarking

```bash
python test.py --model_name qwen3-4b --device 0
python benchmark_inststeer.py --mode detection --model_name qwen3-4b
python benchmark_inststeer.py --mode multilayer --model_name qwen3-4b
python benchmark_inststeer.py --mode defense --model_name qwen3-4b
python eval_multilayer_voting.py --model_name qwen3-4b
```

Reported metrics: accuracy (ACC), false positive rate (FPR), false negative rate (FNR), true positive rate (TPR), F1, plus inference latency and memory overhead.

### 3. Demo — end-to-end detection

```bash
python demo.py                     # fits the projector on the packaged calibration set, then detects
python demo.py --model_name llama3.1-8b
```

Prints per-example projection scores and aggregate ACC/FPR/FNR on the packaged test set.

> ⚠️ **Demo-only:** the numbers printed here come from the small packaged set and are **not**
> the numbers reported in the paper (ACC > 98%, FPR 0.29%).

## Training data

The training in this repository is **only for testing the pipeline**. To reproduce the
paper results, use the balanced calibration corpus described in the paper:

- **Benign samples** from processed [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) and
  [Natural Questions](https://huggingface.co/datasets/google-research-datasets/natural_questions);
- **Malicious samples** generated with the Naive Attack and NeuralExec configurations;

from which **200 samples per class** are randomly sampled to estimate the instruction-sensitive projector `w*`.

## Data format

Each dataset directory contains:

- `data.json` — a list of examples `[{"instruction": "", "data_prompt": "<text>"}]`
  (`data_prompt` is the untrusted external content);
- `label.json` — aligned list of labels (`0` = benign, `1` = injected).

`data/TestData/dataset_candidates.json` lists `[clean_dir, malicious_dir]` pairs.

## Citation

If you find AEGIS useful, please cite:

```bibtex
@misc{chen2026overrefusaldefendingindirectprompt,
      title={Beyond Over-Refusal: Defending Indirect Prompt Injection via Latent Instruction Manifolds},
      author={Jiahao Chen and Rui Yin and Xinfeng Li and Qianli Ma and Tianyu Du and Zhihui Fu and Jun Wang and Zhaoxiang Wang and Shouling Ji},
      year={2026},
      eprint={2608.22248},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2608.22248},
}
```

## License

This repository contains research code; a license will be added soon. It ships with a small
demonstration calibration set only — model weights, hidden states, and internal
infrastructure are intentionally **not** included.
